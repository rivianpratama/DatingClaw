"""Stateless release scorer. Controller state changes only through commit()."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import FeatureBlock
from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.decision.calibration import Calibrator
from bumbleclaw_v6.decision.controller import OnlineQuotaController
from bumbleclaw_v6.features.extract import FeatureBundle, extract_image, load_models
from bumbleclaw_v6.features.preprocess import (
    LEGACY_FULL,
    PHOTO_TIGHT,
    PHOTO_TIGHT_FLIP,
    PHOTO_V1,
    PHOTO_WIDE,
)
from bumbleclaw_v6.features.recipes import FeatureRecipe
from bumbleclaw_v6.models.linear import LinearModel
from bumbleclaw_v6.provenance import read_json, stable_hash
from bumbleclaw_v6.release.validate import validate_release
from bumbleclaw_v6.schemas import FeatureStatus

FeatureFn = Callable[[str | Path], FeatureBundle]


class V6Scorer:
    def __init__(
        self,
        release_dir: str | Path,
        *,
        controller: bool = True,
        feature_fn: FeatureFn | None = None,
    ):
        report = validate_release(release_dir)
        if not report["PASS"]:
            raise ValueError(f"release validation failed: {report['checks']}")
        root = Path(release_dir)
        self.release_dir = root
        self.manifest = read_json(root / "manifest.json")
        self.decision = read_json(root / "decision.json")
        self.keys = tuple(str(k) for k in self.manifest["keys"])
        self.model = LinearModel.load(root / "model.npz")
        self.transform = FeatureTransform.from_dict(read_json(root / "transform.json"))
        self.calibrator = Calibrator.from_dict(self.decision["calibrator"])
        self.lam = float(self.decision["lambda"])
        self.unknown_cost_policy = str(self.decision.get("unknown_cost_policy", "abstain"))
        self.ctrl = None
        self._pending_utility: dict[str, float] = {}
        if controller:
            c = self.decision.get("controller", {})
            if not c.get("warm_start"):
                raise ValueError("live controller requires a non-empty warm_start in decision.json "
                                 "(otherwise the first threshold is -inf and early cards all swipe right)")
            self.ctrl = OnlineQuotaController(
                target=float(self.decision.get("quota", C.QUOTA)),
                window=int(c.get("window", 300)),
                half_life=float(c.get("half_life", 50.0)),
                kp=float(c.get("kp", 2.0)),
                warm_start=np.asarray(c.get("warm_start", []), dtype=np.float64)
                if c.get("warm_start") else None,
            )
        self.feature_fn = feature_fn
        self._models = None
        if feature_fn is None:
            self._models = self._load_models()

    def _load_models(self):
        views = []
        if "sf" in self.keys:
            views.append(LEGACY_FULL)
        if any(k in self.keys for k in ("sp", "d3", "sp_flip", "stop")):
            views.append(PHOTO_V1)
        if any(k in self.keys for k in ("sp_tight", "sp_tight_tta")):
            views.append(PHOTO_TIGHT)
        if any(k in self.keys for k in ("sp_tight_flip", "sp_tight_tta")):
            views.append(PHOTO_TIGHT_FLIP)
        if "sp_wide" in self.keys:
            views.append(PHOTO_WIDE)
        self.recipe = FeatureRecipe(
            f"release_{self.manifest['name']}", self.keys, tuple(views)
        )
        return load_models(
            self.recipe,
            need_arc="arc" in self.keys,
            need_siglip=any(k in self.keys for k in ("sf", "sp", "d3", "sp_flip", "sp_tight", "sp_tight_flip", "sp_tight_tta", "sp_wide", "stop")),
            need_dinov3="d3" in self.keys,
        )

    def _extract(self, path: str | Path) -> FeatureBundle:
        if self.feature_fn is not None:
            return self.feature_fn(path)
        return extract_image(path, self.recipe, self._models)

    def extract_array(self, img_bgr: np.ndarray) -> FeatureBundle:
        """Runtime extraction from an in-memory BGR frame (no temp PNG round-trip)."""
        if self.feature_fn is not None or self._models is None:
            raise RuntimeError("extract_array requires a release loaded with its own models")
        return extract_image(None, self.recipe, self._models, image=img_bgr)

    def _bundle_blocks(self, bundle: FeatureBundle) -> dict[str, FeatureBlock]:
        blocks: dict[str, FeatureBlock] = {}
        for key in self.keys:
            if key not in bundle.values:
                raise KeyError(f"feature bundle is missing {key}")
            values = np.asarray(bundle.values[key], dtype=np.float32).reshape(1, -1)
            mask = np.asarray([bundle.masks[key]], dtype=bool)
            status = np.asarray([int(bundle.status[key])], dtype=np.uint8)
            blocks[key] = FeatureBlock(key, values, mask, status)
        return blocks

    def _cost(self, old_score: float | None) -> tuple[float, bool, str | None]:
        if old_score is None or not np.isfinite(float(old_score)):
            policy = self.unknown_cost_policy
            if policy == "zero":
                return 0.0, False, None
            if policy == "one":
                return 1.0, False, None
            return 0.0, False, "unknown_cost"
        return float(float(old_score) < float(self.decision["cat_fp_threshold"])), True, None

    def score_bundle(self, bundle: FeatureBundle, old_score: float | None) -> dict[str, Any]:
        required_failures = []
        for key in self.keys:
            status = bundle.status.get(key, FeatureStatus.NOT_EXTRACTED)
            if status in (FeatureStatus.DECODE_ERROR, FeatureStatus.UNSUPPORTED_LAYOUT,
                          FeatureStatus.EXTRACTION_ERROR, FeatureStatus.NOT_EXTRACTED):
                required_failures.append(f"{key}:{FeatureStatus(status).name}")
        if required_failures:
            return {"right": None, "abstain": True, "reason": "feature_failure",
                    "feature_failures": required_failures, "p": None, "u": None,
                    "cost": None, "cost_known": False, "threshold": None}
        blocks = self._bundle_blocks(bundle)
        x, _ = self.transform.transform(blocks)
        if x.shape[1] != len(self.model.weights):
            raise ValueError("design matrix width does not match release model")
        logit = float(self.model.predict(x)[0])
        p = float(self.calibrator.apply(
            np.array([logit]),
            model_hash=self.model.hash,
            transform_hash=stable_hash(self.transform.to_dict()),
            recipe_hash=self.manifest["recipe_hash"],
        )[0])
        cost, cost_known, cost_error = self._cost(old_score)
        if cost_error:
            return {"right": None, "abstain": True, "reason": cost_error,
                    "p": p, "u": None, "cost": None, "cost_known": False,
                    "threshold": None, "logit": logit}
        utility = p - self.lam * (1.0 - p) * cost
        threshold = None
        right = None
        if self.ctrl is not None:
            threshold = float(self.ctrl.threshold())
            right = bool(self.ctrl.propose(utility))
        return {
            "right": right,
            "abstain": right is None,
            "reason": None if right is not None else "no_controller",
            "logit": logit,
            "p": p,
            "u": float(utility),
            "cost": float(cost),
            "cost_known": bool(cost_known),
            "threshold": threshold,
            "model_hash": self.model.hash,
            "release": self.manifest["name"],
        }

    def decide(self, path: str | Path, old_score: float | None = None,
               event_id: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        bundle = self._extract(path)
        result = self.score_bundle(bundle, old_score)
        result["event_id"] = event_id
        if event_id is not None and result.get("u") is not None and result.get("right") is not None:
            self._pending_utility[str(event_id)] = float(result["u"])
        result["timings_ms"] = {"total": round((time.perf_counter() - started) * 1000.0, 3)}
        result["timings_ms"].update({k: round(float(v), 3) for k, v in bundle.meta.get("timings_ms", {}).items()})
        return result

    def commit(self, event_id: str, action: bool) -> None:
        if self.ctrl is None:
            raise RuntimeError("release was loaded without a controller")
        if str(event_id) not in self._pending_utility:
            raise KeyError(f"commit for unknown/undecided event {event_id}")
        # The controller history must hold utilities (not 0/1 actions) for its quantile threshold.
        self.ctrl.commit(event_id, bool(action), value=self._pending_utility[str(event_id)])  # kept: duplicate commits stay idempotent

"""Photo-only LoRA seed-ensemble release: build, validate, score.

Release layout (immutable once built):
  manifest.json   kind=lora_ensemble_v1, seeds, sha256 of every file, lora config
  transform.json  side-feature transform (arc, d3, fv), identical across seeds
  seed{i}.pt      trainable LoRA/head/clf state for seed i
  decision.json   calibrator (or null => calibration PENDING => always abstain)

The score is the mean logit over seeds. Without a calibrator the scorer logs the
logit and abstains with reason ``uncalibrated``; it never fabricates p(like).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from bumbleclaw_v6.data.load import FeatureBlock
from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.provenance import stable_hash
from bumbleclaw_v6.schemas import FeatureStatus

KIND = "lora_ensemble_v1"
SIDE_KEYS = ("arc", "d3", "fv")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(runs: list[Path], out: Path, name: str) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    transforms = {_sha(r / "transform.json") for r in runs}
    if len(transforms) != 1:
        raise ValueError("seed runs use different side transforms")
    configs = []
    for r in runs:
        if not (r / "COMPLETE.json").exists():
            raise ValueError(f"run not complete: {r}")
        summary = json.loads((r / "summary.json").read_text())
        if not summary.get("full_fit") or summary["side_keys"] != list(SIDE_KEYS):
            raise ValueError(f"run is not a photo-only full fit: {r}")
        configs.append(summary["config"])
    keys = ("r", "alpha", "first_layer", "targets", "dropout", "epochs")
    if any({k: c[k] for k in keys} != {k: configs[0][k] for k in keys} for c in configs):
        raise ValueError("seed runs differ in LoRA config")
    out.mkdir(parents=True)
    shutil.copy2(runs[0] / "transform.json", out / "transform.json")
    files = {"transform.json": _sha(out / "transform.json")}
    seeds = []
    import torch
    for i, r in enumerate(runs):
        state = torch.load(r / "models" / "full_final_lora.pt", map_location="cpu")
        target = out / f"seed{i}.pt"
        torch.save(state, target)
        files[target.name] = _sha(target)
        seeds.append({"file": target.name, "seed": configs[i]["seed"], "run": r.name})
    (out / "decision.json").write_text(json.dumps({
        "calibrator": None,
        "calibration_status": "PENDING",
        "note": "No honest held-out logits exist (all seeds fit on all labels). "
                "Scorer abstains until a calibrator from fold refits or R22-independent labels is added.",
    }, indent=2))
    files["decision.json"] = _sha(out / "decision.json")
    manifest = {
        "schema": "bumbleclaw_release_v1", "kind": KIND, "name": name,
        "side_keys": list(SIDE_KEYS), "photo_only": True, "uses_sf": False,
        "lora": {k: configs[0][k] for k in keys}, "portrait_scale": 1.8,
        "seeds": seeds, "aggregate": "mean_logit", "files": files,
        "box_source_training": "legacy detector bboxes (C.TRAIN_D3_NPZ/HOLDOUT_NPZ)",
        "box_source_runtime": "insightface buffalo_l via bumbleclaw_v6.features.extract",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def validate(root: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    manifest = json.loads((root / "manifest.json").read_text())
    checks["kind"] = manifest.get("kind") == KIND
    checks["photo_only"] = manifest.get("photo_only") is True and "sf" not in manifest.get("side_keys", [])
    for name, digest in manifest["files"].items():
        checks[f"sha:{name}"] = (root / name).exists() and _sha(root / name) == digest
    return {"PASS": all(checks.values()), "checks": checks}


class LoRAEnsembleScorer:
    def __init__(self, release_dir: str | Path, *, controller: bool = False,
                 load_side_models: bool = True):
        import torch
        from bumbleclaw_v6.features.extract import load_models
        from bumbleclaw_v6.features.preprocess import PHOTO_V1
        from bumbleclaw_v6.features.recipes import FeatureRecipe
        from bumbleclaw_v6.finetune.lora_siglip import LoRASigLIP

        root = Path(release_dir)
        report = validate(root)
        if not report["PASS"]:
            raise ValueError(f"LoRA release validation failed: {report['checks']}")
        if controller:
            raise ValueError("LoRA ensemble release is shadow-only (no controller)")
        self.torch = torch
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.decision = json.loads((root / "decision.json").read_text())
        self.transform = FeatureTransform.from_dict(json.loads((root / "transform.json").read_text()))
        self.transform_hash = stable_hash(self.transform.to_dict())
        cfg = self.manifest["lora"]
        from bumbleclaw_v6.features.extract import _dim
        probe = {k: FeatureBlock(k, np.zeros((1, _dim(k)), np.float32), np.ones(1, bool),
                                 np.zeros(1, np.uint8)) for k in SIDE_KEYS}
        d_side = int(self.transform.transform(probe)[0].shape[1])
        self.model = LoRASigLIP(int(cfg["first_layer"]), int(cfg["r"]), int(cfg["alpha"]),
                                float(cfg["dropout"]), d_side, tuple(cfg["targets"]))
        self.model.eval()
        dev = next(self.model.clf.parameters()).device
        self.states = [{k: v.to(dev) for k, v in torch.load(root / s["file"], map_location="cpu").items()}
                       for s in self.manifest["seeds"]]
        self.recipe = FeatureRecipe("lora_side_v1", SIDE_KEYS, (PHOTO_V1,))
        # `logits()` only needs the LoRA model; side models are only for `decide()`.
        self.models = (load_models(self.recipe, need_arc=True, need_siglip=False, need_dinov3=True)
                       if load_side_models else None)
        self.scale = float(self.manifest["portrait_scale"])
        # One seed: load its weights once instead of copying the state dict every card.
        self._single_state = len(self.states) == 1
        if self._single_state:
            self.model.load_state_dict(self.states[0], strict=False)

    def _pixels(self, img, bbox):
        import cv2
        from bumbleclaw_v6.features.preprocess import crop_portrait
        crop, _ = crop_portrait(img, bbox, 384, self.scale)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
        return self.torch.from_numpy(np.ascontiguousarray(x))

    def logits(self, img_bgr, bbox, side_blocks: dict[str, FeatureBlock]) -> list[float]:
        torch = self.torch
        x, _ = self.transform.transform(side_blocks)
        dev = next(self.model.clf.parameters()).device
        side = torch.tensor(x, dtype=torch.float32, device=dev)
        px = self._pixels(img_bgr, bbox).to(dev).half()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if self._single_state:
                return [float(self.model(px, side).float().cpu().numpy()[0])]
            for state in self.states:
                self.model.load_state_dict(state, strict=False)
                out.append(float(self.model(px, side).float().cpu().numpy()[0]))
        return out

    def decide(self, path: str | Path, old_score: float | None = None,
               event_id: str | None = None) -> dict[str, Any]:
        from bumbleclaw_v6.features.extract import extract_image
        from bumbleclaw_v6.features.preprocess import load_bgr
        if self.models is None:
            raise RuntimeError("decide() requires load_side_models=True")
        started = time.perf_counter()
        bundle = extract_image(path, self.recipe, self.models)
        base = {"event_id": event_id, "release": self.manifest["name"], "kind": KIND,
                "right": None, "abstain": True, "p": None, "u": None}
        bad = [f"{k}:{FeatureStatus(bundle.status[k]).name}" for k in SIDE_KEYS
               if bundle.status[k] in (FeatureStatus.DECODE_ERROR, FeatureStatus.UNSUPPORTED_LAYOUT,
                                       FeatureStatus.EXTRACTION_ERROR, FeatureStatus.NOT_EXTRACTED)]
        if bad:
            base.update(reason="feature_failure", feature_failures=bad)
        else:
            blocks = {k: FeatureBlock(k, np.asarray(bundle.values[k], np.float32).reshape(1, -1),
                                      np.asarray([bundle.masks[k]], bool),
                                      np.asarray([int(bundle.status[k])], np.uint8)) for k in SIDE_KEYS}
            t = time.perf_counter()
            seed_logits = self.logits(load_bgr(path), bundle.meta.get("bbox"), blocks)
            base["lora_ms"] = round((time.perf_counter() - t) * 1e3, 3)
            base["seed_logits"] = seed_logits
            base["logit"] = float(np.mean(seed_logits))
            base["reason"] = "uncalibrated" if self.decision.get("calibrator") is None else "no_controller"
        base["timings_ms"] = {"total": round((time.perf_counter() - started) * 1e3, 3)}
        return base


def load_any_scorer(release_dir: str | Path, *, controller: bool = False):
    """Factory used by phone shadow mode: linear v6 releases or LoRA ensembles."""
    for marker in ("DEFECT.md", "SUPERSEDED.md"):
        if (Path(release_dir) / marker).exists():
            raise ValueError(f"release is marked {marker[:-3]}: {Path(release_dir) / marker}")
    manifest = json.loads((Path(release_dir) / "manifest.json").read_text())
    if manifest.get("kind") == KIND:
        return LoRAEnsembleScorer(release_dir, controller=controller)
    from bumbleclaw_v6.runtime.scorer import V6Scorer
    return V6Scorer(release_dir, controller=controller)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--runs", nargs="+", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--name", required=True)
    v = sub.add_parser("validate")
    v.add_argument("release", type=Path)
    args = parser.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.runs, args.out, args.name), indent=2))
    else:
        report = validate(args.release)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["PASS"] else 1)


if __name__ == "__main__":
    main()

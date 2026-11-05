"""Package one exact fitted pipeline: model + transform + calibrator + policy."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.decision.calibration import Calibrator
from bumbleclaw_v6.eval.artifacts import load_run
from bumbleclaw_v6.models.linear import LinearModel
from bumbleclaw_v6.provenance import (
    atomic_write_json,
    ensure_writable_new,
    read_json,
    sha256_file,
    stable_hash,
)


def _verify_bindings(selection: dict[str, Any], policy: dict[str, Any], model: LinearModel,
                     transform: FeatureTransform) -> tuple[Calibrator, str]:
    transform_hash = stable_hash(transform.to_dict())
    if selection.get("transform_hash") != transform_hash:
        raise ValueError("selection transform hash does not match stored transform")
    if selection.get("model_hash") != model.hash:
        raise ValueError("selection model hash does not match stored model")
    calibrator = Calibrator.from_dict(policy["calibrator"])
    if calibrator.model_hash != model.hash:
        raise ValueError("calibrator is bound to a different model")
    if calibrator.transform_hash != transform_hash:
        raise ValueError("calibrator is bound to a different transform")
    if calibrator.recipe_hash != selection.get("feature_recipe_hash"):
        raise ValueError("calibrator recipe hash does not match the incumbent recipe")
    if not policy.get("policy", {}).get("feasible", False):
        raise ValueError("policy is not feasible")
    lam = policy["policy"].get("lambda")
    if lam is None or lam < 0:
        raise ValueError("release policy has no valid lambda")
    return calibrator, transform_hash


def build_release(
    *,
    base_run: str | Path,
    policy_run: str | Path,
    name: str,
    fold: str = "F5",
    seed: int = 0,
    releases_root: str | Path | None = None,
    kind: str = "development_release",
    extra_manifest: dict[str, Any] | None = None,
    warm_start: list[float] | None = None,
) -> Path:
    base = load_run(base_run)
    policy = load_run(policy_run)
    selection = read_json(Path(base["path"]) / f"selection_{fold}_s{seed}.json")
    policy_record = read_json(Path(policy["path"]) / f"policy_{fold}.json")
    if selection.get("fold") != fold or int(selection.get("seed", -1)) != seed:
        raise ValueError("selection fold/seed mismatch")
    model_path = Path(base["path"]) / "models" / selection["model_file"]
    model = LinearModel.load(model_path)
    transform = FeatureTransform.from_dict(selection["transform"])
    calibrator, transform_hash = _verify_bindings(selection, policy_record, model, transform)
    keys = list(selection["keys"])
    root = Path(releases_root or C.RELEASES_DIR)
    root.mkdir(parents=True, exist_ok=True)
    out = ensure_writable_new(root / name, root)
    out.mkdir(parents=True)
    model.save(out / "model.npz")
    atomic_write_json(out / "transform.json", transform.to_dict())
    atomic_write_json(out / "calibration.json", calibrator.to_dict())
    decision = {
        "schema": C.POLICY_SCHEMA,
        "calibrator": calibrator.to_dict(),
        "lambda": float(policy_record["policy"]["lambda"]),
        "budget_per_4540": int(policy["manifest"].get("budget_per_4540", 0)),
        "quota": C.QUOTA,
        "cat_fp_threshold": C.CAT_FP_THRESHOLD,
        "unknown_cost_policy": "abstain",
        "controller": {"window": 300, "half_life": 50.0, "kp": 2.0, "warm_start": [float(v) for v in (warm_start or [])]},
    }
    atomic_write_json(out / "decision.json", decision)
    atomic_write_json(out / "policy.json", policy_record)
    manifest = {
        "schema": "bumbleclaw_release_v1",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "fold": fold,
        "seed": seed,
        "keys": keys,
        "model_hash": model.hash,
        "transform_hash": transform_hash,
        "recipe_hash": selection["feature_recipe_hash"],
        "role_fingerprint": selection.get("role_fingerprint"),
        "base_run": str(Path(base_run)),
        "policy_run": str(Path(policy_run)),
        "snapshot": base["manifest"].get("snapshot"),
        "snapshot_hash": base["manifest"].get("snapshot_hash"),
        "base_run_manifest": base["manifest"],
        "policy_run_manifest": policy["manifest"],
        "files": {
            "model.npz": sha256_file(out / "model.npz"),
            "transform.json": sha256_file(out / "transform.json"),
            "calibration.json": sha256_file(out / "calibration.json"),
            "decision.json": sha256_file(out / "decision.json"),
            "policy.json": sha256_file(out / "policy.json"),
        },
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    atomic_write_json(out / "manifest.json", manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--policy-run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", default="F5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--releases-root", type=Path, default=None)
    parser.add_argument("--kind", default="development_release")
    args = parser.parse_args()
    print(build_release(
        base_run=args.base_run,
        policy_run=args.policy_run,
        name=args.name,
        fold=args.fold,
        seed=args.seed,
        releases_root=args.releases_root,
        kind=args.kind,
    ))


if __name__ == "__main__":
    main()

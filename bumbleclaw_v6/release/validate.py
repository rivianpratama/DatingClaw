"""Startup validation for a packaged release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.decision.calibration import Calibrator
from bumbleclaw_v6.models.linear import LinearModel
from bumbleclaw_v6.provenance import read_json, sha256_file, stable_hash


def validate_release(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    required = ("manifest.json", "model.npz", "transform.json", "calibration.json", "decision.json", "policy.json")
    report: dict[str, Any] = {"path": str(root), "checks": {}, "PASS": True}
    for name in required:
        exists = (root / name).is_file()
        report["checks"][f"file:{name}"] = exists
        report["PASS"] &= exists
    if not report["PASS"]:
        return report
    manifest = read_json(root / "manifest.json")
    model = LinearModel.load(root / "model.npz")
    transform = FeatureTransform.from_dict(read_json(root / "transform.json"))
    decision = read_json(root / "decision.json")
    calibrator = Calibrator.from_dict(decision["calibrator"])
    transform_hash = stable_hash(transform.to_dict())
    checks = {
        "model_hash": manifest.get("model_hash") == model.hash,
        "transform_hash": manifest.get("transform_hash") == transform_hash,
        "calibrator_model_binding": calibrator.model_hash == model.hash,
        "calibrator_transform_binding": calibrator.transform_hash == transform_hash,
        "calibrator_recipe_binding": calibrator.recipe_hash == manifest.get("recipe_hash"),
        "lambda_nonnegative": float(decision.get("lambda", -1)) >= 0,
        "budget_nonnegative": int(decision.get("budget_per_4540", -1)) >= 0,
        "files": all(
            manifest.get("files", {}).get(name) == sha256_file(root / name)
            for name in ("model.npz", "transform.json", "calibration.json", "decision.json", "policy.json")
        ),
    }
    report["checks"].update(checks)
    report["PASS"] = bool(report["PASS"] and all(checks.values()))
    report["model_hash"] = model.hash
    report["transform_hash"] = transform_hash
    report["lambda"] = decision.get("lambda")
    report["budget_per_4540"] = decision.get("budget_per_4540")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    report = validate_release(args.release)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["PASS"] else 1)


if __name__ == "__main__":
    main()

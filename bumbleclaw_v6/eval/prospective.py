"""R22 prospective evaluation manifest: freeze first, collect labels later.

This module never fabricates labels. It records the release hash, sampling plan,
and hidden-label contract so a later analysis can run exactly the registered
evaluation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bumbleclaw_v6.provenance import atomic_write_json, read_json, sha256_file
from bumbleclaw_v6.release.validate import validate_release

SCHEMA = "bumbleclaw_prospective_v1"
LABEL_SCHEMA = [
    "trial_id",
    "sample_id",
    "capture_ts",
    "image_path",
    "human_action",
    "old_score",
    "consent",
    "notes",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def release_metadata(release_dir: str | Path) -> dict[str, Any]:
    root = Path(release_dir)
    report = validate_release(root)
    if not report["PASS"]:
        raise ValueError(f"release validation failed: {report['checks']}")
    manifest = read_json(root / "manifest.json")
    freeze = read_json(root / "FREEZE.json") if (root / "FREEZE.json").exists() else {}
    return {
        "release": manifest["name"],
        "release_dir": str(root.resolve()),
        "release_manifest_sha256": sha256_file(root / "manifest.json"),
        "model_hash": manifest["model_hash"],
        "transform_hash": manifest["transform_hash"],
        "recipe_hash": manifest["recipe_hash"],
        "keys": list(manifest["keys"]),
        "frozen_utc": freeze.get("frozen_utc"),
        "freeze_status": freeze.get("status"),
    }


def build_manifest(
    release: dict[str, Any],
    *,
    plan_name: str,
    target_cards: int,
    owner: str = "owner",
    notes: str = "",
) -> dict[str, Any]:
    if target_cards <= 0:
        raise ValueError("target_cards must be positive")
    if not release.get("release_manifest_sha256"):
        raise ValueError("release manifest hash is required")
    return {
        "schema": SCHEMA,
        "plan_name": plan_name,
        "created_utc": _now(),
        "status": "collecting",
        "owner": owner,
        "release": release,
        "target_cards": int(target_cards),
        "collected_cards": 0,
        "labels_hidden": True,
        "labels_revealed": False,
        "sampling": {
            "method": "consecutive_future_cards",
            "selection_by_model_score": False,
            "selection_by_old_score": False,
            "include_no_face": True,
            "include_decode_or_layout_failures": True,
            "notes": "Collect the next consecutive eligible cards after the freeze; do not cherry-pick.",
        },
        "label_schema": LABEL_SCHEMA,
        "evaluation": {
            "primary": "recall@20.8% and severe-FP count on the frozen release",
            "secondary": "calibration/Brier, coverage, abstain rate, latency",
            "tuning_after_reveal": "forbidden",
        },
        "notes": notes,
        "labels": [],
    }


def init_manifest(
    release_dir: str | Path,
    out_path: str | Path,
    *,
    plan_name: str,
    target_cards: int,
    owner: str = "owner",
    notes: str = "",
) -> Path:
    out = Path(out_path)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite prospective manifest: {out}")
    payload = build_manifest(
        release_metadata(release_dir), plan_name=plan_name,
        target_cards=target_cards, owner=owner, notes=notes,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, payload)
    return out


def validate_manifest(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported prospective manifest schema")
    if payload.get("labels_hidden") is not True or payload.get("labels_revealed") is not False:
        raise ValueError("prospective manifest must keep labels hidden")
    if int(payload.get("target_cards", 0)) <= 0:
        raise ValueError("target_cards must be positive")
    release = payload.get("release", {})
    root = Path(release.get("release_dir", ""))
    if not root.is_dir():
        raise ValueError("release directory is missing")
    if sha256_file(root / "manifest.json") != release.get("release_manifest_sha256"):
        raise ValueError("release manifest changed after prospective freeze")
    return payload


def seal_collection(path: str | Path, *, collected_cards: int) -> Path:
    payload = validate_manifest(path)
    if collected_cards < int(payload["target_cards"]):
        raise ValueError("target card count not reached")
    payload["status"] = "sealed"
    payload["collected_cards"] = int(collected_cards)
    payload["sealed_utc"] = _now()
    atomic_write_json(path, payload)
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--release", type=Path, required=True)
    p_init.add_argument("--out", type=Path, required=True)
    p_init.add_argument("--plan-name", default="r22_photo_tight_prospective")
    p_init.add_argument("--target-cards", type=int, default=2000)
    p_init.add_argument("--owner", default="owner")
    p_init.add_argument("--notes", default="")
    p_val = sub.add_parser("validate")
    p_val.add_argument("manifest", type=Path)
    p_seal = sub.add_parser("seal")
    p_seal.add_argument("manifest", type=Path)
    p_seal.add_argument("--collected-cards", type=int, required=True)
    args = parser.parse_args()
    if args.command == "init":
        print(init_manifest(args.release, args.out, plan_name=args.plan_name,
                            target_cards=args.target_cards, owner=args.owner, notes=args.notes))
    elif args.command == "validate":
        print(json.dumps(validate_manifest(args.manifest), indent=2))
    else:
        print(seal_collection(args.manifest, collected_cards=args.collected_cards))


if __name__ == "__main__":
    main()

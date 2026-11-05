"""R19 full-data photo-only release: all-label fit + grouped OOF calibration.

This is intentionally not the matched A/H/C/P/T protocol. It answers the release
question: fit one exact model on all available labels, then calibrate it without
using in-sample logits. The calibrator is fitted on grouped out-of-fold logits and
bound to the final model/transform/recipe hashes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.decision.calibration import fit_platt
from bumbleclaw_v6.eval.artifacts import RunArtifacts
from bumbleclaw_v6.eval.metrics import deck_metrics
from bumbleclaw_v6.experiments.baselines import CONFIGS, specs_for
from bumbleclaw_v6.features.bank import FeatureStoreBank
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.models.linear import fit_linear_logreg
from bumbleclaw_v6.provenance import hash_id_list, stable_hash
from bumbleclaw_v6.release.build import build_release
from bumbleclaw_v6.release.validate import validate_release


def _groups(meta) -> np.ndarray:
    if "group_id" not in meta.columns:
        return np.array(["g:" + str(v) for v in meta["sample_id"]], dtype=object)
    raw = meta["group_id"].astype(str).to_numpy()
    ids = meta["sample_id"].astype(str).to_numpy()
    return np.array([g if g and g not in {"-1", "nan", "None"} else "g:" + s
                     for g, s in zip(raw, ids)], dtype=object)


def grouped_folds(groups: np.ndarray, n_folds: int, seed: int = 0) -> np.ndarray:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    unique = unique[rng.permutation(len(unique))]
    fold_of = {g: i % n_folds for i, g in enumerate(unique)}
    return np.array([fold_of[g] for g in groups], dtype=np.int16)


def fit_full_release(
    *,
    snapshot_dir: str | Path,
    feature_store: str | Path,
    store_recipe: str = "crop_v1",
    release_recipe: str = "photo_tight_v1",
    config: str = "no_sf_tight",
    wd: float = 1.0,
    seed: int = 0,
    n_folds: int = 5,
    lam: float = 1.0,
    run_prefix: str = "e19r_full",
    policy_prefix: str = "e19p_full",
    results_root: str | Path | None = None,
    releases_root: str | Path | None = None,
    release_name: str = "v6r2_photo_tight_full",
    build: bool = True,
) -> dict[str, Any]:
    C.setup_env()
    if config not in CONFIGS:
        raise KeyError(f"unknown config {config}")
    keys = CONFIGS[config]
    snapshot = load_snapshot(snapshot_dir)
    meta = snapshot["samples"]
    y = meta["y"].to_numpy(dtype=np.int8)
    ids = meta["sample_id"].astype(str).to_numpy()
    old_score = meta["old_score"].to_numpy(dtype=np.float64)
    n = len(ids)
    bank = FeatureStoreBank(snapshot_dir, feature_store, get_recipe(store_recipe))
    missing = set(keys) - set(bank.store.manifest["dims"])
    if missing:
        raise KeyError(f"feature store is missing blocks: {sorted(missing)}")
    blocks = bank.blocks(keys)
    specs = specs_for(keys)
    all_idx = np.arange(n, dtype=np.int64)
    transform = FeatureTransform.fit(blocks, all_idx, specs)
    X, _ = transform.transform(blocks)
    final = fit_linear_logreg(X, y, weight_decay=wd, seed=seed)
    final_logits = final.predict(X)
    transform_hash = stable_hash(transform.to_dict())
    release_recipe_obj = get_recipe(release_recipe)
    recipe_hash = stable_hash({
        "recipe": release_recipe,
        "recipe_hash": release_recipe_obj.hash,
        "keys": list(keys),
    })

    # Grouped OOF calibration: no validation group appears in its fold's fit.
    groups = _groups(meta)
    fold_ids = grouped_folds(groups, n_folds=n_folds, seed=seed)
    oof_logits = np.full(n, np.nan, dtype=np.float64)
    for fold in range(n_folds):
        train = np.flatnonzero(fold_ids != fold)
        val = np.flatnonzero(fold_ids == fold)
        if len(val) == 0:
            continue
        fold_transform = FeatureTransform.fit(blocks, train, specs)
        X_fold, _ = fold_transform.transform(blocks)
        fold_model = fit_linear_logreg(X_fold[train], y[train], weight_decay=wd, seed=seed)
        oof_logits[val] = fold_model.predict(X_fold[val])
    if not np.isfinite(oof_logits).all():
        raise FloatingPointError("OOF logits contain non-finite values")
    calibrator = fit_platt(
        oof_logits, y,
        model_hash=final.hash,
        transform_hash=transform_hash,
        recipe_hash=recipe_hash,
        calibration_ids=ids,
    )

    in_sample = deck_metrics(final_logits, y, ids, old_score=old_score)
    oof_metrics = deck_metrics(oof_logits, y, ids, old_score=old_score)
    p_final = calibrator.apply(final_logits, model_hash=final.hash,
                               transform_hash=transform_hash, recipe_hash=recipe_hash)

    root = Path(results_root or C.RESULTS_DIR)
    run_id = f"{run_prefix}_{config}"
    manifest = {
        "kind": "full_data_linear_probe",
        "config": config,
        "keys": list(keys),
        "protocol": "full_data_release_v1",
        "snapshot": str(snapshot_dir),
        "snapshot_hash": snapshot["manifest"]["snapshot_hash"],
        "feature_store": str(feature_store),
        "feature_store_recipe": store_recipe,
        "release_recipe": release_recipe,
        "release_recipe_hash": release_recipe_obj.hash,
        "folds": ["ALL"],
        "seeds": [seed],
        "weight_decay": float(wd),
        "n_rows": int(n),
        "n_likes": int(y.sum()),
        "calibration": "grouped_out_of_fold_platt",
        "calibration_folds": int(n_folds),
    }
    run = RunArtifacts.create(run_id, root, manifest)
    model_path = run.save_model("ALL_s0", final)
    run.save_predictions("ALL", "ALL", seed, ids, final_logits, p=p_final)
    np.savez_compressed(
        run.root / "oof_calibration.npz",
        sample_id=ids.astype("U"),
        logit=oof_logits.astype(np.float32),
        fold=fold_ids,
        y=y.astype(np.int8),
    )
    selection = {
        "config": config,
        "keys": list(keys),
        "fold": "ALL",
        "seed": int(seed),
        "selected_weight_decay": float(wd),
        "selection_hits": int(in_sample["hits"]),
        "model_file": model_path.name,
        "model_hash": final.hash,
        "transform": transform.to_dict(),
        "transform_hash": transform_hash,
        "feature_recipe": release_recipe,
        "feature_recipe_hash": recipe_hash,
        "role_fingerprint": None,
        "metrics": {"ALL_in_sample": in_sample, "ALL_oof": oof_metrics},
    }
    run.save_json("selection_ALL_s0.json", selection)
    summary = {
        "run_id": run_id,
        "config": config,
        "keys": list(keys),
        "protocol": "full_data_release_v1",
        "weight_decay": float(wd),
        "seed": int(seed),
        "n_rows": int(n),
        "n_likes": int(y.sum()),
        "model_hash": final.hash,
        "transform_hash": transform_hash,
        "feature_recipe": release_recipe,
        "feature_recipe_hash": recipe_hash,
        "in_sample": in_sample,
        "oof": oof_metrics,
        "calibrator": calibrator.to_dict(),
        "fold_sizes": {str(i): int((fold_ids == i).sum()) for i in range(n_folds)},
        "calibration_note": "Platt fit on grouped OOF logits; calibrator bound to final model/transform/recipe hashes.",
    }
    run.finalize(summary)

    policy_id = f"{policy_prefix}_{config}"
    policy_manifest = {
        "kind": "full_data_policy",
        "base_run": str(run.root),
        "base_run_id": run_id,
        "budget_per_4540": int(C.DIAGNOSTIC_BUDGET_PER_DECK),
        "folds": ["ALL"],
        "protocol": "full_data_release_v1",
        "snapshot": str(snapshot_dir),
        "unknown_cost_policy": "abstain",
        "policy_note": "Fixed conservative lambda=1.0 after R15 stop; no further F2-F5 tuning.",
    }
    policy_run = RunArtifacts.create(policy_id, root, policy_manifest)
    policy_record = {
        "calibrator": calibrator.to_dict(),
        "policy": {
            "lambda": float(lam),
            "feasible": True,
            "reason": "fixed conservative release lambda; R15 stopped; no further F2-F5 experiments",
            "sweeps": [],
            "budget_p": int(C.DIAGNOSTIC_BUDGET_PER_DECK),
        },
        "test": {"scope": "ALL", "in_sample": in_sample, "oof": oof_metrics},
    }
    policy_run.save_json("policy_ALL.json", policy_record)
    policy_summary = {
        "run_id": policy_id,
        "base_run": str(run.root),
        "budget": int(C.DIAGNOSTIC_BUDGET_PER_DECK),
        "lambda": float(lam),
        "calibrator": calibrator.to_dict(),
        "oof": oof_metrics,
        "in_sample": in_sample,
        "policy_note": "fixed conservative lambda=1.0",
    }
    policy_run.finalize(policy_summary)

    release_path = None
    if build:
        release_path = build_release(
            base_run=run.root,
            policy_run=policy_run.root,
            name=release_name,
            fold="ALL",
            seed=seed,
            releases_root=releases_root,
            kind="photo_only_full_data_release",
            extra_manifest={
                "feature_store": str(feature_store),
                "store_recipe": store_recipe,
                "release_recipe": release_recipe,
                "training_rows": int(n),
                "training_likes": int(y.sum()),
                "training_weight_decay": float(wd),
                "calibration_method": "grouped_out_of_fold_platt",
                "calibration_folds": int(n_folds),
                "calibration_note": "Calibrator fit on grouped OOF logits; not in-sample.",
                "policy_note": "Fixed lambda=1.0; no further F2-F5 tuning.",
                "oof_metrics": oof_metrics,
                "in_sample_metrics": in_sample,
            },
        )
        validation = validate_release(release_path)
        if not validation["PASS"]:
            raise RuntimeError(f"release validation failed: {validation['checks']}")
    return {
        "run": str(run.root),
        "policy_run": str(policy_run.root),
        "release": None if release_path is None else str(release_path),
        "model_hash": final.hash,
        "transform_hash": transform_hash,
        "recipe_hash": recipe_hash,
        "in_sample": in_sample,
        "oof": oof_metrics,
        "calibrator": calibrator.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--store-recipe", default="crop_v1")
    parser.add_argument("--release-recipe", default="photo_tight_v1")
    parser.add_argument("--config", default="no_sf_tight")
    parser.add_argument("--wd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--run-prefix", default="e19r_full")
    parser.add_argument("--policy-prefix", default="e19p_full")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--releases-root", type=Path, default=None)
    parser.add_argument("--release-name", default="v6r2_photo_tight_full")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    print(json.dumps(fit_full_release(
        snapshot_dir=args.snapshot,
        feature_store=args.feature_store,
        store_recipe=args.store_recipe,
        release_recipe=args.release_recipe,
        config=args.config,
        wd=args.wd,
        seed=args.seed,
        n_folds=args.n_folds,
        lam=args.lam,
        run_prefix=args.run_prefix,
        policy_prefix=args.policy_prefix,
        results_root=args.results_root,
        releases_root=args.releases_root,
        release_name=args.release_name,
        build=not args.no_build,
    ), indent=2))


if __name__ == "__main__":
    main()

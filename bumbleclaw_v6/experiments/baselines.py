"""E01/E02: matched-pipeline linear probes on legacy features.

Legacy features are used for development only. The corrected role protocol:
A fit -> H select -> final A∪H fit -> C calibration -> P policy -> T test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import LegacyFeatureBank, load_snapshot
from bumbleclaw_v6.features.bank import FeatureStoreBank
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.data.transforms import BlockSpec, FeatureTransform
from bumbleclaw_v6.eval.artifacts import RunArtifacts
from bumbleclaw_v6.eval.metrics import deck_metrics
from bumbleclaw_v6.eval.splits import make_matched_roles
from bumbleclaw_v6.models.linear import fit_linear_logreg
from bumbleclaw_v6.provenance import stable_hash

CONFIGS: dict[str, tuple[str, ...]] = {
    "arc": ("arc", "fv"),
    "portrait": ("sp", "fv"),
    "arc_portrait": ("arc", "sp", "fv"),
    "base": ("arc", "sf", "sp", "d3", "fv"),
    "no_sf": ("arc", "sp", "d3", "fv"),
    "no_sf_tight": ("arc", "sp_tight", "d3", "fv"),
    "no_sf_wide": ("arc", "sp_wide", "d3", "fv"),
    "no_sf_tight_tta": ("arc", "sp_tight_tta", "d3", "fv"),
}


def specs_for(keys: tuple[str, ...]) -> list[BlockSpec]:
    specs = []
    for key in keys:
        if key == "fv":
            specs.append(BlockSpec(key, "passthrough", False, False))
        else:
            specs.append(BlockSpec(key, "unit", True, True))
    return specs


def run_baselines(
    snapshot_dir: str | Path,
    *,
    configs: tuple[str, ...] = ("arc_portrait",),
    folds: tuple[str, ...] = ("F5",),
    seeds: tuple[int, ...] = (0,),
    wd_grid: tuple[float, ...] = (1.0,),
    run_prefix: str = "e01r",
    results_root: str | Path | None = None,
    feature_store: str | Path | None = None,
    feature_recipe: str = C.LEGACY_FEATURE_RECIPE,
) -> list[Path]:
    """Run one immutable baseline experiment per config."""
    C.setup_env()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)
    snapshot_path = Path(snapshot_dir)
    snapshot = load_snapshot(snapshot_path)
    meta = snapshot["samples"]
    y = meta["y"].to_numpy(dtype=np.int8)
    old_score = meta["old_score"].to_numpy(dtype=np.float64)
    ids = meta["sample_id"].astype(str).to_numpy()
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        raise KeyError(f"unknown configs: {sorted(unknown)}")
    union_keys = tuple(dict.fromkeys(k for name in configs for k in CONFIGS[name]))
    if feature_store is not None:
        bank = FeatureStoreBank(snapshot_path, feature_store, get_recipe(feature_recipe))
        missing = set(union_keys) - set(bank.store.manifest["dims"])
        if missing:
            raise KeyError(f"feature store is missing blocks: {sorted(missing)}")
    else:
        bank = LegacyFeatureBank(snapshot_path, union_keys)
    all_blocks = bank.blocks(union_keys)
    output_runs: list[Path] = []
    root = Path(results_root or C.RESULTS_DIR)

    for config_name in configs:
        keys = CONFIGS[config_name]
        run_id = f"{run_prefix}_{config_name}"
        manifest = {
            "kind": "matched_linear_probe",
            "config": config_name,
            "keys": list(keys),
            "protocol": C.SPLIT_SCHEMA,
            "snapshot": str(snapshot_path),
            "snapshot_hash": snapshot["manifest"]["snapshot_hash"],
            "feature_recipe": feature_recipe,
            "feature_store": None if feature_store is None else str(feature_store),
            "group_policy": "filename_only",
            "identity_clean": False,
            "folds": list(folds),
            "seeds": list(seeds),
            "wd_grid": list(wd_grid),
            "quotas": {"primary": C.QUOTA, "diagnostic_severe_budget": C.DIAGNOSTIC_BUDGET_PER_DECK},
            "feature_coverage": bank.coverage(),
        }
        run = RunArtifacts.create(run_id, root, manifest)
        summary: dict[str, Any] = {"run_id": run_id, "config": config_name, "folds": {}, "seeds": list(seeds),
                                   "wd_grid": list(wd_grid), "protocol": C.SPLIT_SCHEMA}
        for fold_name in folds:
            roles = make_matched_roles(meta, fold_name)
            specs = specs_for(keys)
            transform = FeatureTransform.fit(all_blocks, roles.indices("A"), specs)
            transform_hash = stable_hash(transform.to_dict())
            X, names = transform.transform(all_blocks)
            if X.shape[1] != len(names):
                raise AssertionError("transform column count mismatch")
            A, H, CV, P, T = (roles.indices(r) for r in ("A", "H", "C", "P", "T"))
            fold_summary: dict[str, Any] = {"fingerprint": roles.fingerprint,
                                            "rows": {r: int(len(roles.indices(r))) for r in ("A", "H", "C", "P", "T")},
                                            "seeds": {}}
            for seed in seeds:
                chosen: tuple[int, float, Any] | None = None
                for wd in wd_grid:
                    candidate = fit_linear_logreg(X[A], y[A], weight_decay=wd, seed=seed)
                    h_logits = candidate.predict(X[H])
                    h_metrics = deck_metrics(h_logits, y[H], ids[H], old_score=old_score[H])
                    key = (h_metrics["hits"], wd)
                    if chosen is None or key > (chosen[0], chosen[1]):
                        chosen = (h_metrics["hits"], wd, candidate)
                assert chosen is not None
                _, wd, _ = chosen
                final = fit_linear_logreg(X[np.r_[A, H]], y[np.r_[A, H]], weight_decay=wd, seed=seed)
                path = run.save_model(f"{fold_name}_s{seed}", final)
                role_metrics: dict[str, Any] = {}
                for role_name, idx in (("H", H), ("C", CV), ("P", P), ("T", T)):
                    logits = final.predict(X[idx])
                    run.save_predictions(fold_name, role_name, seed, ids[idx], logits)
                    role_metrics[role_name] = deck_metrics(logits, y[idx], ids[idx], old_score=old_score[idx])
                run.save_json(
                    f"selection_{fold_name}_s{seed}.json",
                    {
                        "config": config_name,
                        "keys": list(keys),
                        "fold": fold_name,
                        "seed": seed,
                        "selected_weight_decay": wd,
                        "selection_hits": int(chosen[0]),
                        "model_file": path.name,
                        "model_hash": final.hash,
                        "transform": transform.to_dict(),
                        "transform_hash": transform_hash,
                        "feature_recipe_hash": stable_hash({
                            "recipe": feature_recipe,
                            "recipe_hash": getattr(bank, "recipe_hash", None),
                            "keys": list(keys),
                        }),
                        "role_fingerprint": roles.fingerprint,
                        "metrics": role_metrics,
                    },
                )
                fold_summary["seeds"][str(seed)] = {"selected_weight_decay": wd,
                                                    "selection_hits": int(chosen[0]),
                                                    "model_hash": final.hash,
                                                    "metrics": role_metrics}
            summary["folds"][fold_name] = fold_summary
        run.finalize(summary)
        output_runs.append(run.root)
    return output_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["arc_portrait"], choices=sorted(CONFIGS))
    parser.add_argument("--folds", nargs="+", default=["F5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--wd-grid", nargs="+", type=float, default=[1.0])
    parser.add_argument("--run-prefix", default="e01r")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--feature-store", type=Path, default=None)
    parser.add_argument("--feature-recipe", default=C.LEGACY_FEATURE_RECIPE)
    args = parser.parse_args()
    paths = run_baselines(args.snapshot, configs=tuple(args.configs), folds=tuple(args.folds),
                          seeds=tuple(args.seeds), wd_grid=tuple(args.wd_grid),
                          run_prefix=args.run_prefix, results_root=args.results_root,
                          feature_store=args.feature_store, feature_recipe=args.feature_recipe)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

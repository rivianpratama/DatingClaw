"""R10: model-bound calibration and expected-cost policy frontier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.decision.calibration import Calibrator, fit_platt
from bumbleclaw_v6.decision.policy import (
    UnknownCostPolicy,
    choose_lambda_on_p,
    select_utility,
    selected_metrics,
)
from bumbleclaw_v6.eval.artifacts import RunArtifacts, load_predictions, load_run
from bumbleclaw_v6.eval.metrics import deck_metrics, k_for
from bumbleclaw_v6.eval.splits import make_matched_roles
from bumbleclaw_v6.provenance import read_json, stable_hash


def run_policy_sweep(
    base_run: str | Path,
    snapshot_dir: str | Path,
    *,
    budgets: tuple[int, ...] = (10,),
    folds: tuple[str, ...] = C.PRIMARY_FOLDS,
    run_prefix: str = "e02r",
    results_root: str | Path | None = None,
) -> list[Path]:
    C.setup_env()
    base = load_run(base_run)
    meta = load_snapshot(snapshot_dir)["samples"]
    y = meta["y"].to_numpy(dtype=np.int8)
    old_score = meta["old_score"].to_numpy(dtype=np.float64)
    ids = meta["sample_id"].astype(str).to_numpy()
    root = Path(results_root or C.RESULTS_DIR)
    output: list[Path] = []
    for budget in budgets:
        run_id = f"{run_prefix}_b{budget}"
        manifest = {
            "kind": "expected_cost_policy_sweep",
            "base_run": str(Path(base_run)),
            "base_run_id": base["manifest"]["run_id"],
            "budget_per_4540": int(budget),
            "unknown_cost_policy": "abstain",
            "folds": list(folds),
            "protocol": C.SPLIT_SCHEMA,
            "group_policy": "filename_only",
            "identity_clean": False,
            "snapshot": str(snapshot_dir),
        }
        run = RunArtifacts.create(run_id, root, manifest)
        summary: dict[str, Any] = {"run_id": run_id, "base_run": str(Path(base_run)), "budget": int(budget),
                                   "folds": {}}
        for fold in folds:
            roles = make_matched_roles(meta, fold)
            cv, p_idx, t_idx = roles.indices("C"), roles.indices("P"), roles.indices("T")
            selection = read_json(Path(base["path"]) / f"selection_{fold}_s0.json")
            model_hash = selection["model_hash"]
            transform_hash = selection["transform_hash"]
            recipe_hash = selection["feature_recipe_hash"]
            c_pred = load_predictions(base["path"], fold, "C", seed=0)
            p_pred = load_predictions(base["path"], fold, "P", seed=0)
            t_pred = load_predictions(base["path"], fold, "T", seed=0)
            if not (np.array_equal(c_pred["sample_id"], ids[cv]) and
                    np.array_equal(p_pred["sample_id"], ids[p_idx]) and
                    np.array_equal(t_pred["sample_id"], ids[t_idx])):
                raise ValueError("base run predictions do not align with snapshot roles")
            calibrator = fit_platt(
                c_pred["logit"], y[cv], model_hash=model_hash, transform_hash=transform_hash,
                recipe_hash=recipe_hash, calibration_ids=ids[cv],
            )
            p_p = calibrator.apply(p_pred["logit"], model_hash=model_hash,
                                   transform_hash=transform_hash, recipe_hash=recipe_hash)
            p_t = calibrator.apply(t_pred["logit"], model_hash=model_hash,
                                   transform_hash=transform_hash, recipe_hash=recipe_hash)
            choice = choose_lambda_on_p(
                p_p, old_score[p_idx], y[p_idx], ids[p_idx],
                budget_per_4540=budget, unknown_policy=UnknownCostPolicy.ABSTAIN,
            )
            fold_result: dict[str, Any] = {
                "fingerprint": roles.fingerprint,
                "calibrator": calibrator.to_dict(),
                "policy": {"lambda": choice.lam, "feasible": choice.feasible, "reason": choice.reason,
                           "sweeps": choice.sweeps, "budget_p": int(max(0, int(np.floor(budget * len(p_idx) / C.DIAGNOSTIC_DECK_SIZE))))},
                "test": None,
            }
            if choice.feasible:
                k_t = k_for(len(t_idx), C.QUOTA)
                selected = select_utility(
                    p_t, old_score[t_idx], ids[t_idx], k_t, float(choice.lam),
                    unknown_policy=UnknownCostPolicy.ABSTAIN,
                )
                if not selected.feasible:
                    fold_result["test"] = {"feasible": False, "reason": selected.reason}
                else:
                    selected_bool = np.zeros(len(t_idx), dtype=bool)
                    selected_bool[selected.selected] = True
                    costs = (np.isfinite(old_score[t_idx]) &
                             (old_score[t_idx] < C.CAT_FP_THRESHOLD)).astype(np.float64)
                    utility_scores = p_t - float(choice.lam) * (1.0 - p_t) * costs
                    # Unknown-cost candidates are ineligible under ABSTAIN. Keep a finite
                    # sentinel so deterministic metric ranking remains well-defined.
                    utility_scores[~np.isfinite(old_score[t_idx])] = -1e30
                    run.save_predictions(fold, "T", 0, ids[t_idx], utility_scores, p=p_t,
                                         selected=selected_bool)
                    metrics = deck_metrics(utility_scores, y[t_idx], ids[t_idx], old_score=old_score[t_idx])
                    fold_result["test"] = {"feasible": True, "selected": int(selected_bool.sum()),
                                           "metrics": metrics,
                                           "selected_metrics": selected_metrics(selected.selected, y[t_idx], old_score[t_idx])}
            run.save_json(f"policy_{fold}.json", fold_result)
            summary["folds"][fold] = fold_result
        run.finalize(summary)
        output.append(run.root)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[10])
    parser.add_argument("--folds", nargs="+", default=list(C.PRIMARY_FOLDS))
    parser.add_argument("--run-prefix", default="e02r")
    parser.add_argument("--results-root", type=Path, default=None)
    args = parser.parse_args()
    for path in run_policy_sweep(args.base_run, args.snapshot, budgets=tuple(args.budgets),
                                 folds=tuple(args.folds), run_prefix=args.run_prefix,
                                 results_root=args.results_root):
        print(path)


if __name__ == "__main__":
    main()

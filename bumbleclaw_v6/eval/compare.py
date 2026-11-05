"""ID-aligned run comparison with recomputed metrics and paired resampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.eval.artifacts import align_prediction_ids, load_predictions, load_run
from bumbleclaw_v6.eval.metrics import deck_metrics, pooled_metrics
from bumbleclaw_v6.eval.splits import make_matched_roles
from bumbleclaw_v6.eval.stats import paired_bootstrap


def _aligned_scores(run_dir: str | Path, fold: str, role: str, reference_ids: np.ndarray) -> np.ndarray:
    pred = load_predictions(run_dir, fold, role, seed=0)
    values = align_prediction_ids(reference_ids, pred["sample_id"], pred["logit"])
    return np.asarray(values, dtype=np.float64)


def compare_runs(
    a_run: str | Path,
    b_run: str | Path,
    snapshot_dir: str | Path,
    *,
    folds: tuple[str, ...] = C.PRIMARY_FOLDS,
    role: str = "T",
    n_boot: int = 1000,
) -> dict[str, Any]:
    a_meta = load_run(a_run)
    b_meta = load_run(b_run)
    meta = load_snapshot(snapshot_dir)["samples"]
    y_all = meta["y"].to_numpy(dtype=np.int8)
    old_all = meta["old_score"].to_numpy(dtype=np.float64)
    ids_all = meta["sample_id"].astype(str).to_numpy()
    per_a: dict[str, dict[str, Any]] = {}
    per_b: dict[str, dict[str, Any]] = {}
    a_scores, b_scores, y_scores = {}, {}, {}
    for fold in folds:
        roles = make_matched_roles(meta, fold)
        idx = roles.indices(role)
        ref_ids = ids_all[idx]
        a_scores[fold] = _aligned_scores(a_run, fold, role, ref_ids)
        b_scores[fold] = _aligned_scores(b_run, fold, role, ref_ids)
        y_scores[fold] = y_all[idx]
        per_a[fold] = deck_metrics(a_scores[fold], y_all[idx], ref_ids, old_score=old_all[idx])
        per_b[fold] = deck_metrics(b_scores[fold], y_all[idx], ref_ids, old_score=old_all[idx])
    paired = paired_bootstrap(a_scores, b_scores, y_scores, folds=folds, n_boot=n_boot)
    out = {
        "a": a_meta["manifest"].get("run_id", str(a_run)),
        "b": b_meta["manifest"].get("run_id", str(b_run)),
        "role": role,
        "per_fold_a": per_a,
        "per_fold_b": per_b,
        "pooled_a": pooled_metrics(per_a, folds),
        "pooled_b": pooled_metrics(per_b, folds),
        "paired": paired,
    }
    out["candidate"] = out["a"]
    out["baseline"] = out["b"]
    out["delta_direction"] = "a_minus_b"
    out["PROMOTE_DIAGNOSTIC"] = _promote_diagnostic(paired, per_a, per_b, folds)
    return out


def _promote_diagnostic(paired: dict[str, Any], per_a: dict[str, dict[str, Any]],
                        per_b: dict[str, dict[str, Any]], folds: tuple[str, ...]) -> bool:
    """Heuristic only: `a` is the candidate and `b` is the baseline."""
    return bool(
        paired["delta_hits"] >= max(4, len(folds))
        and paired["ci95"][0] > 0
        and sum(per_a[f]["hits"] > per_b[f]["hits"] for f in folds) >= int(np.ceil(0.75 * len(folds)))
        and min(per_a[f]["hits"] - per_b[f]["hits"] for f in folds) >= -3
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", default=list(C.PRIMARY_FOLDS))
    parser.add_argument("--role", default="T")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.a, args.b, args.snapshot, folds=tuple(args.folds),
                                  role=args.role, n_boot=args.n_boot), indent=2))


if __name__ == "__main__":
    main()

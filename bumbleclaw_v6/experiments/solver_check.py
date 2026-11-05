"""E02 diagnostic: finite-step AdamW probe vs a converged L2 logistic solver.

This is a development diagnostic, not a promotion run. It uses the same A/H
rows and deterministic hits@K on H; it never touches C/P/T.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import LegacyFeatureBank, load_snapshot
from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.eval.metrics import deck_metrics
from bumbleclaw_v6.eval.splits import make_matched_roles
from bumbleclaw_v6.experiments.baselines import CONFIGS, specs_for
from bumbleclaw_v6.models.convex_probe import fit_sklearn_logreg, predict_logits
from bumbleclaw_v6.models.linear import fit_linear_logreg
from bumbleclaw_v6.provenance import atomic_write_json, ensure_writable_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fold", default="F5")
    parser.add_argument("--config", default="arc_portrait", choices=sorted(CONFIGS))
    parser.add_argument("--max-rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adamw-wd", type=float, default=1.0)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    args = parser.parse_args()
    C.setup_env()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)
    snapshot = load_snapshot(args.snapshot)
    meta = snapshot["samples"]
    y = meta["y"].to_numpy(dtype=np.int8)
    ids = meta["sample_id"].astype(str).to_numpy()
    old = meta["old_score"].to_numpy(dtype=np.float64)
    keys = CONFIGS[args.config]
    bank = LegacyFeatureBank(args.snapshot, keys)
    blocks = bank.blocks(keys)
    roles = make_matched_roles(meta, args.fold)
    transform = FeatureTransform.fit(blocks, roles.indices("A"), specs_for(keys))
    X, _ = transform.transform(blocks)
    A = roles.indices("A")
    H = roles.indices("H")
    if args.max_rows > 0:
        A = A[:args.max_rows]
    result = {"config": args.config, "fold": args.fold, "max_rows": int(len(A)),
              "adamw": {}, "sklearn": {}, "role_fingerprint": roles.fingerprint}
    adamw = fit_linear_logreg(X[A], y[A], weight_decay=args.adamw_wd, seed=args.seed)
    result["adamw"] = {"hits": deck_metrics(adamw.predict(X[H]), y[H], ids[H], old_score=old[H])["hits"],
                       "weight_decay": args.adamw_wd}
    for c_reg in args.c_grid:
        clf = fit_sklearn_logreg(X[A], y[A], C_reg=c_reg, seed=args.seed)
        m = deck_metrics(predict_logits(clf, X[H]), y[H], ids[H], old_score=old[H])
        result["sklearn"][str(c_reg)] = {
            "hits": m["hits"],
            "n_iter": int(getattr(clf, "n_iter_", [-1])[0]),
            "converged": bool(np.max(clf.n_iter_) < clf.max_iter),
        }
    target = ensure_writable_new(C.RESULTS_DIR / f"solver_check_{args.config}_{args.fold}.json", C.RESULTS_DIR)
    atomic_write_json(target, result)
    print(result)
    print(target)


if __name__ == "__main__":
    main()

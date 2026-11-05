"""Development probe: train on all rows before F5, evaluate raw ranking on F5.

This is deliberately separate from the matched A/H/C/P/T protocol. It answers a
narrow question: does using the C/P rows for model fitting improve holdout recall?
It does not produce a calibrated release and it uses F5 as a development deck.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import LegacyFeatureBank, load_snapshot
from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.eval.metrics import deck_metrics
from bumbleclaw_v6.experiments.baselines import CONFIGS, specs_for
from bumbleclaw_v6.models.linear import fit_linear_logreg
from bumbleclaw_v6.provenance import atomic_write_json, ensure_writable_new

EXTRA_CONFIGS = {
    "all": ("arc", "sf", "sp", "d3", "d2", "geom", "fv"),
    "arc_sp_d3": ("arc", "sp", "d3", "fv"),
    "arc_sf_sp": ("arc", "sf", "sp", "fv"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["arc_portrait", "base", "all"])
    parser.add_argument("--wd-grid", nargs="+", type=float, default=[0.1, 0.3, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--train-end", type=int, default=50_070)
    parser.add_argument("--test-start", type=int, default=50_070)
    parser.add_argument("--test-end", type=int, default=54_610)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    C.setup_env()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)
    snapshot = load_snapshot(args.snapshot)
    meta = snapshot["samples"]
    y = meta["y"].to_numpy(dtype=np.int8)
    ids = meta["sample_id"].astype(str).to_numpy()
    old = meta["old_score"].to_numpy(dtype=np.float64)
    configs = {**CONFIGS, **EXTRA_CONFIGS}
    unknown = set(args.configs) - set(configs)
    if unknown:
        raise KeyError(f"unknown configs: {sorted(unknown)}")
    union_keys = tuple(dict.fromkeys(k for name in args.configs for k in configs[name]))
    bank = LegacyFeatureBank(args.snapshot, union_keys)
    blocks = bank.blocks(union_keys)
    train = np.arange(0, args.train_end, dtype=np.int64)
    test = np.arange(args.test_start, args.test_end, dtype=np.int64)
    result = {"train_end": args.train_end, "test": [args.test_start, args.test_end],
              "configs": {}, "note": "F5 is a development deck in this probe"}
    for name in args.configs:
        keys = configs[name]
        transform = FeatureTransform.fit(blocks, train, specs_for(keys))
        X, _ = transform.transform(blocks)
        result["configs"][name] = {"keys": list(keys), "runs": []}
        for wd in args.wd_grid:
            for seed in args.seeds:
                model = fit_linear_logreg(X[train], y[train], weight_decay=wd, seed=seed)
                m = deck_metrics(model.predict(X[test]), y[test], ids[test], old_score=old[test])
                row = {"wd": wd, "seed": seed, "hits": m["hits"], "n_pos": m["n_pos"],
                       "recall": m["recall"], "severe": m["proxy_severe_fp"],
                       "auc": m["auc"], "model_hash": model.hash}
                result["configs"][name]["runs"].append(row)
                print(name, row, flush=True)
    out = args.out or (C.RESULTS_DIR / "full_train_probe.json")
    target = ensure_writable_new(out, C.RESULTS_DIR)
    atomic_write_json(target, result)
    print(target)


if __name__ == "__main__":
    main()

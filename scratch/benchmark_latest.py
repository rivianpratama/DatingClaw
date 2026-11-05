"""Benchmark the latest photo-only method with explicit contamination labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.data.transforms import FeatureTransform
from bumbleclaw_v6.decision.calibration import Calibrator
from bumbleclaw_v6.decision.policy import UnknownCostPolicy, select_utility
from bumbleclaw_v6.eval.metrics import lower_tail_mean
from bumbleclaw_v6.eval.splits import make_matched_roles
from bumbleclaw_v6.features.bank import FeatureStoreBank
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.models.linear import LinearModel
from bumbleclaw_v6.provenance import deterministic_topk

SNAP = Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
STORE = Path(r"D:\bumbleclaw_v6r1\features\crop_v1_full")
RELEASE = Path(r"D:\bumbleclaw_v6r1\releases\v6r2_photo_tight_full")
OUT = Path(r"D:\bumbleclaw_v6r1\results\e19_latest_benchmark.json")


def metrics_from_selection(y, old, selected_idx, p, k):
    selected = np.asarray(selected_idx, dtype=np.int64)
    y_sel = y[selected]
    fp = selected[y_sel == 0]
    old_fp = old[fp]
    known = np.isfinite(old_fp)
    old_known = old_fp[known]
    hits = int(y_sel.sum())
    return {
        "hits": hits,
        "n_pos": int(y.sum()),
        "recall": float(hits / int(y.sum())),
        "selected_count": int(len(selected)),
        "precision": float(hits / max(len(selected), 1)),
        "proxy_severe_fp": int((old_known < C.CAT_FP_THRESHOLD).sum()),
        "cute_fp_share": float((old_known >= C.CUTE_THRESHOLD).mean()) if len(old_known) else None,
        "fp_mean_old_score": float(old_known.mean()) if len(old_known) else None,
        "fp_lower_tail_mean": lower_tail_mean(old_known, 0.05) if len(old_known) else None,
        "unknown_cost_selected": int((~known).sum()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
    }


def main() -> None:
    snap = load_snapshot(SNAP)
    meta = snap["samples"]
    old = meta["old_score"].to_numpy(dtype=np.float64)
    y = meta["y"].to_numpy(dtype=np.int8)
    ids = meta["sample_id"].astype(str).to_numpy()
    t_idx = make_matched_roles(meta, "F5").indices("T")
    y_t, old_t, ids_t = y[t_idx], old[t_idx], ids[t_idx]
    k = int(np.floor(C.QUOTA * len(t_idx)))
    bank = FeatureStoreBank(SNAP, STORE, get_recipe("crop_v1"))
    model = LinearModel.load(RELEASE / "model.npz")
    transform = FeatureTransform.from_dict(json.loads((RELEASE / "transform.json").read_text()))
    cal = Calibrator.from_dict(json.loads((RELEASE / "calibration.json").read_text()))
    blocks = bank.blocks(tuple(json.loads((RELEASE / "manifest.json").read_text())["keys"]))
    X, _ = transform.transform(blocks)
    logits = model.predict(X[t_idx])
    p_final = cal.apply(logits, model_hash=model.hash,
                        transform_hash=__import__("bumbleclaw_v6.provenance", fromlist=["stable_hash"]).stable_hash(transform.to_dict()),
                        recipe_hash=json.loads((RELEASE / "manifest.json").read_text())["recipe_hash"])
    # OOF logits, mapped to F5 T IDs.
    with np.load(Path(r"D:\bumbleclaw_v6r1\results\e19r_full_no_sf_tight\oof_calibration.npz"), allow_pickle=False) as z:
        oof_ids = z["sample_id"].astype(str).tolist()
        oof = z["logit"].astype(np.float64)
    lookup = {sid: i for i, sid in enumerate(oof_ids)}
    p_oof = cal.apply(oof[[lookup[s] for s in ids_t]], model_hash=model.hash,
                      transform_hash=__import__("bumbleclaw_v6.provenance", fromlist=["stable_hash"]).stable_hash(transform.to_dict()),
                      recipe_hash=json.loads((RELEASE / "manifest.json").read_text())["recipe_hash"])
    def raw(p):
        return np.asarray(deterministic_topk(p, ids_t, k), dtype=np.int64)
    def policy(p):
        sel = select_utility(p, old_t, ids_t, k, 1.0, unknown_policy=UnknownCostPolicy.ABSTAIN)
        return np.asarray(sel.selected, dtype=np.int64)
    result = {
        "release": str(RELEASE),
        "fold": "F5",
        "n": int(len(t_idx)), "n_pos": int(y_t.sum()), "K": k,
        "rows": {
            "final_in_sample_raw": {"contamination": "F5 labels were in training", **metrics_from_selection(y_t, old_t, raw(p_final), p_final, k)},
            "final_in_sample_policy": {"contamination": "F5 labels were in training", **metrics_from_selection(y_t, old_t, policy(p_final), p_final, k)},
            "grouped_oof_raw": {"contamination": "out-of-fold for each F5 row; not sealed temporal holdout", **metrics_from_selection(y_t, old_t, raw(p_oof), p_oof, k)},
            "grouped_oof_policy": {"contamination": "out-of-fold for each F5 row; not sealed temporal holdout", **metrics_from_selection(y_t, old_t, policy(p_oof), p_oof, k)},
        },
        "e05_tight_development": {},
    }
    e05 = json.loads(Path(r"D:\bumbleclaw_v6r1\results\e15p_crop_no_sf_tight_b10\summary.json").read_text())["folds"]["F5"]["test"]["metrics"]
    result["e05_tight_development"] = {
        "contamination": "matched development F5; model trained on A/H, not F5 T",
        **{key: e05.get(key) for key in ("hits","n_pos","recall","proxy_severe_fp","cute_fp_share","fp_mean_old_score","fp_lower_tail_mean","auc")},
        "selected_count": e05.get("selected_count"),
        "policy_lambda": 0.75,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

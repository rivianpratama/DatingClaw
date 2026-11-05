"""GPU parity: release scorer on runtime images vs stored crop_v1 features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.features.extract import FeatureBundle
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.features.store import FeatureStore
from bumbleclaw_v6.runtime.scorer import V6Scorer
from bumbleclaw_v6.schemas import FeatureStatus

SNAP = Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
STORE = Path(r"D:\bumbleclaw_v6r1\features\crop_v1_full")
RELEASE = Path(r"D:\bumbleclaw_v6r1\releases\v6r5_photo_tight_wide_blend")
OUT = Path(r"D:\bumbleclaw_v6r1\results\e24_v6r5_parity.json")


def main() -> None:
    meta = load_snapshot(SNAP)["samples"]
    ids = meta["sample_id"].astype(str).to_numpy()
    path = meta["path"].astype(str).to_numpy()
    y = meta["y"].to_numpy()
    old = meta["old_score"].to_numpy(dtype=np.float64)
    store = FeatureStore.load(STORE, get_recipe("crop_v1"))
    scorer = V6Scorer(RELEASE, controller=False)
    keys = tuple(scorer.keys)
    status_arc = np.asarray(store.status["arc"])
    no_face = np.flatnonzero(status_arc == int(FeatureStatus.NO_FACE))[:5]
    unsupported = np.flatnonzero(status_arc == int(FeatureStatus.UNSUPPORTED_LAYOUT))
    decode = np.flatnonzero(status_arc == int(FeatureStatus.DECODE_ERROR))
    rng = np.random.default_rng(0)
    ok = np.flatnonzero(status_arc == int(FeatureStatus.OK))
    sample_ok = rng.choice(ok, size=20, replace=False)
    rows = list(sample_ok) + list(no_face) + list(unsupported) + list(decode)
    report = {"release": str(RELEASE), "rows": [], "keys": list(keys)}
    for row in rows:
        values = {k: np.asarray(store.values[k][row], dtype=np.float32) for k in keys}
        masks = {k: bool(int(np.asarray(store.status[k])[row]) == int(FeatureStatus.OK)) for k in keys}
        status = {k: FeatureStatus(int(np.asarray(store.status[k])[row])) for k in keys}
        if "fv" in masks:
            masks["fv"] = True  # fv is a valid 0/1 flag even when no face
        bundle = FeatureBundle(values, masks, status, {})
        stored = scorer.score_bundle(bundle, float(old[row]) if np.isfinite(old[row]) else None)
        runtime = scorer.decide(path[row], float(old[row]) if np.isfinite(old[row]) else None)
        rec = {
            "row": int(row),
            "sample_id": str(ids[row]),
            "y": int(y[row]),
            "stored_status": {k: status[k].name for k in keys},
            "stored_p": stored.get("p"),
            "runtime_p": runtime.get("p"),
            "stored_logit": stored.get("logit"),
            "runtime_logit": runtime.get("logit"),
            "stored_abstain": bool(stored.get("abstain")),
            "runtime_abstain": bool(runtime.get("abstain")),
            "stored_reason": stored.get("reason"),
            "runtime_reason": runtime.get("reason"),
            "runtime_ms": runtime.get("timings_ms", {}).get("total"),
        }
        if rec["stored_p"] is not None and rec["runtime_p"] is not None:
            rec["abs_p_diff"] = abs(float(rec["stored_p"]) - float(rec["runtime_p"]))
        if rec["stored_logit"] is not None and rec["runtime_logit"] is not None:
            rec["abs_logit_diff"] = abs(float(rec["stored_logit"]) - float(rec["runtime_logit"]))
        report["rows"].append(rec)
    diffs = [r["abs_p_diff"] for r in report["rows"] if "abs_p_diff" in r]
    report["summary"] = {
        "n": len(report["rows"]),
        "max_abs_p_diff": float(max(diffs)) if diffs else None,
        "median_abs_p_diff": float(np.median(diffs)) if diffs else None,
        "status_mismatches": sum(1 for r in report["rows"] if r["stored_abstain"] != r["runtime_abstain"]),
        "pass_at_5e-3": bool(diffs and max(diffs) <= 5e-3),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    for r in report["rows"]:
        if "abs_p_diff" in r and r["abs_p_diff"] > 5e-3:
            print("DIFF", r["row"], r["stored_status"], r["abs_p_diff"])


if __name__ == "__main__":
    main()

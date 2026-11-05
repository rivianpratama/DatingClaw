"""Verify the R13 photo_v1 feature store: completeness, statuses, and legacy parity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bumbleclaw_v6.data.load import LegacyFeatureBank, load_snapshot
from bumbleclaw_v6.features.parity import cosine_rows
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.features.store import FeatureStore
from bumbleclaw_v6.provenance import hash_id_list, read_json
from bumbleclaw_v6.schemas import FeatureStatus

SNAP = Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
STORE = Path(r"D:\bumbleclaw_v6r1\features\photo_v1_full_r2")
OUT = Path(r"D:\bumbleclaw_v6r1\results\r13_store_verification.json")


def main() -> None:
    recipe = get_recipe("photo_v1")
    snapshot = load_snapshot(SNAP)
    samples = snapshot["samples"]
    ids = samples["sample_id"].astype(str).to_numpy()
    store = FeatureStore.load(STORE, recipe)
    manifest = read_json(STORE / "manifest.json")
    progress = read_json(STORE / "progress.json")
    complete = read_json(STORE / "COMPLETE.json")
    report: dict = {
        "store": str(STORE),
        "n": int(manifest["n"]),
        "recipe": manifest["recipe"],
        "recipe_hash": manifest["recipe_hash"],
        "manifest_complete": bool(manifest.get("complete")),
        "progress": progress,
        "complete_marker": complete,
        "sample_ids_match": hash_id_list(ids) == manifest["sample_ids_sha256"],
        "blocks": {},
        "legacy_parity": {},
        "checks": {},
    }
    report["checks"]["row_count"] = int(manifest["n"]) == len(ids) == 54610
    report["checks"]["progress_complete"] = bool(progress.get("complete")) and int(progress.get("next_row", -1)) == len(ids)
    report["checks"]["complete_marker_present"] = (STORE / "COMPLETE.json").exists()
    report["checks"]["sample_ids_match"] = report["sample_ids_match"]

    legacy = LegacyFeatureBank(SNAP, ("arc", "sp", "d3", "fv"))
    for block in ("arc", "sp", "d3", "fv"):
        values = np.asarray(store.values[block])
        status = np.asarray(store.status[block], dtype=np.uint8)
        counts = {FeatureStatus(int(v)).name: int((status == v).sum()) for v in np.unique(status)}
        ok = status == int(FeatureStatus.OK)
        finite_ok = bool(np.isfinite(values[ok]).all()) if ok.any() else True
        zero_non_ok = bool((values[~ok] == 0).all()) if (~ok).any() else True
        report["blocks"][block] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "status_counts": counts,
            "finite_ok": finite_ok,
            "zero_non_ok": zero_non_ok,
        }
        # Legacy parity on rows valid in both stores.
        lb = legacy.block(block)
        lv = np.asarray(lb.values, dtype=np.float32)
        lm = np.asarray(lb.mask, dtype=bool)
        sv = np.asarray(values, dtype=np.float32)
        sm = ok
        both = lm & sm
        if block == "fv":
            report["legacy_parity"][block] = {
                "n_both": int(both.sum()),
                "equal_fraction": float((sv[both, 0] == lv[both, 0]).mean()) if both.any() else None,
            }
        else:
            cos = cosine_rows(sv[both], lv[both])
            cos = cos[np.isfinite(cos)]
            report["legacy_parity"][block] = {
                "n_both": int(both.sum()),
                "median_cos": float(np.median(cos)) if len(cos) else None,
                "p5_cos": float(np.percentile(cos, 5)) if len(cos) else None,
                "min_cos": float(np.min(cos)) if len(cos) else None,
            }
    # Specific legacy zero-row window and no-face checks.
    zero_rows = np.arange(50003, 50070)
    report["checks"]["legacy_zero_window_status"] = {
        block: {FeatureStatus(int(v)).name: int((np.asarray(store.status[block])[zero_rows] == v).sum())
                for v in np.unique(np.asarray(store.status[block])[zero_rows])}
        for block in ("arc", "sp", "d3", "fv")
    }
    report["checks"]["last_row_status"] = {
        block: FeatureStatus(int(np.asarray(store.status[block])[-1])).name for block in ("arc", "sp", "d3", "fv")
    }
    report["checks"]["all_status_values_valid"] = all(
        set(np.unique(np.asarray(store.status[b])).tolist()).issubset({int(s) for s in FeatureStatus})
        for b in ("arc", "sp", "d3", "fv")
    )
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Re-extract a stratified sample and compare against the R13 store."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.features.extract import extract_image, load_models
from bumbleclaw_v6.features.parity import cosine_rows
from bumbleclaw_v6.features.recipes import get_recipe
from bumbleclaw_v6.features.store import FeatureStore
from bumbleclaw_v6.schemas import FeatureStatus

SNAP = Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
STORE = Path(r"D:\bumbleclaw_v6r1\features\photo_v1_full_r2")
OUT = Path(r"D:\bumbleclaw_v6r1\results\r13_runtime_parity_sample.json")


def main() -> None:
    recipe = get_recipe("photo_v1")
    snapshot = load_snapshot(SNAP)
    samples = snapshot["samples"]
    store = FeatureStore.load(STORE, recipe)
    status_arc = np.asarray(store.status["arc"])
    no_face = int(np.flatnonzero(status_arc == int(FeatureStatus.NO_FACE))[0])
    unsupported = int(np.flatnonzero(status_arc == int(FeatureStatus.UNSUPPORTED_LAYOUT))[0])
    rows = [0, 50003, 50010, no_face, unsupported, 54609]
    models = load_models(recipe)
    report = {"rows": rows, "blocks": {}}
    for row in rows:
        bundle = extract_image(samples["path"].iloc[row], recipe, models)
        entry = {"row": row, "sample_id": str(samples["sample_id"].iloc[row]), "status": {}, "cosine": {}}
        for block in recipe.blocks:
            stored = np.asarray(store.values[block][row], dtype=np.float32)
            fresh = np.asarray(bundle.values[block], dtype=np.float32)
            entry["status"][block] = {
                "stored": FeatureStatus(int(np.asarray(store.status[block])[row])).name,
                "fresh": bundle.status[block].name,
            }
            if block == "fv":
                entry["cosine"][block] = float(np.allclose(stored, fresh, atol=1e-6))
            else:
                cos = cosine_rows(stored[None], fresh[None])[0]
                entry["cosine"][block] = float(cos) if np.isfinite(cos) else None
        report["blocks"][str(row)] = entry
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

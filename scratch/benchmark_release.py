"""End-to-end release latency on stored cards."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.runtime.benchmark import benchmark_release
from bumbleclaw_v6.schemas import FeatureStatus

SNAP = Path(r"D:\bumbleclaw_v6r1\datasets\00230fba4faa737e3a52900526bc5fbf160b853debfd8fb0f51e6d88a2138935")
STORE = Path(r"D:\bumbleclaw_v6r1\features\crop_v1_full")
RELEASE = Path(r"D:\bumbleclaw_v6r1\releases\v6r2_photo_tight_full")
OUT = Path(r"D:\bumbleclaw_v6r1\results\e19_release_latency.json")


def main() -> None:
    meta = load_snapshot(SNAP)["samples"]
    paths = meta["path"].astype(str).to_numpy()
    old = meta["old_score"].to_numpy(dtype=np.float64)
    store_status = np.load(STORE / "status" / "arc.npy")
    rng = np.random.default_rng(1)
    ok = np.flatnonzero(store_status == int(FeatureStatus.OK))
    no_face = np.flatnonzero(store_status == int(FeatureStatus.NO_FACE))
    other = np.flatnonzero((store_status != int(FeatureStatus.OK)) & (store_status != int(FeatureStatus.NO_FACE)))
    rows = list(rng.choice(ok, size=180, replace=False)) + list(rng.choice(no_face, size=15, replace=False)) + list(other)
    image_paths = [paths[int(r)] for r in rows]
    old_scores = [float(old[int(r)]) if np.isfinite(old[int(r)]) else None for r in rows]
    result = benchmark_release(RELEASE, image_paths, old_scores)
    result["rows"] = [int(r) for r in rows]
    result["status_counts"] = {"ok": 180, "no_face": 15, "other": int(len(other))}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n", "p50_ms", "p95_ms", "max_ms")}, indent=2))
    print("first5", [round(p["p"] or 0, 4) for p in result["proposals"][:5]])


if __name__ == "__main__":
    main()

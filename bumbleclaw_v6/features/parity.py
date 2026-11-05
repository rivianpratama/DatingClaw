"""Feature parity checks between stored and runtime extraction."""
from __future__ import annotations

from typing import Any

import numpy as np


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("feature arrays must have the same shape")
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (na > 1e-6) & (nb > 1e-6)
    out = np.full(len(a), np.nan, dtype=np.float32)
    out[valid] = (a[valid] * b[valid]).sum(1) / (na[valid] * nb[valid])
    return out


def compare_blocks(stored: dict[str, np.ndarray], runtime: dict[str, np.ndarray],
                   masks: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for block in sorted(set(stored) & set(runtime)):
        a, b = np.asarray(stored[block]), np.asarray(runtime[block])
        if a.shape != b.shape:
            report[block] = {"status": "shape_mismatch", "stored": list(a.shape), "runtime": list(b.shape)}
            continue
        if masks is not None and block in masks:
            valid = np.asarray(masks[block], dtype=bool)
            a, b = a[valid], b[valid]
        cos = cosine_rows(a, b)
        cos = cos[np.isfinite(cos)]
        if len(cos) == 0:
            report[block] = {"status": "no_valid_rows", "n": 0}
            continue
        report[block] = {
            "status": "ok",
            "n": int(len(cos)),
            "median_cos": float(np.median(cos)),
            "p5_cos": float(np.percentile(cos, 5)),
            "min_cos": float(np.min(cos)),
        }
    return report


def parity_gate(report: dict[str, Any], *, min_median: float = 0.995, min_p5: float = 0.98) -> bool:
    if not report:
        return False
    for block, value in report.items():
        if value.get("status") != "ok":
            return False
        if value["median_cos"] < min_median or value["p5_cos"] < min_p5:
            return False
    return True

"""Paired resampling diagnostics for development folds."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.eval.metrics import k_for


def _topk_hits(score: np.ndarray, y: np.ndarray, k: int) -> int:
    if k <= 0:
        return 0
    # Bootstrap indices may repeat, so use position as the stable final tie-break.
    order = np.lexsort((np.arange(len(score)), -np.asarray(score, dtype=np.float64)))
    return int(y[order[:k]].sum())


def paired_bootstrap(
    a_scores: dict[str, np.ndarray],
    b_scores: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    *,
    folds: Iterable[str] = C.PRIMARY_FOLDS,
    quota: float = C.QUOTA,
    n_boot: int = 1000,
    seed: int = 0,
    block_lengths: tuple[int, ...] = (50, 200),
) -> dict:
    names = list(folds)
    rng = np.random.default_rng(seed)
    observed = 0
    for name in names:
        n = len(y[name])
        k = k_for(n, quota)
        observed += _topk_hits(a_scores[name], y[name], k) - _topk_hits(b_scores[name], y[name], k)
    deltas = np.empty(n_boot, dtype=np.float64)
    for r in range(n_boot):
        total = 0
        for name in names:
            n = len(y[name])
            idx = rng.integers(0, n, n)
            k = k_for(n, quota)
            total += _topk_hits(a_scores[name][idx], y[name][idx], k) - _topk_hits(b_scores[name][idx], y[name][idx], k)
        deltas[r] = total
    out = {
        "delta_hits": int(observed),
        "delta_direction": "a_minus_b",
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "fraction_delta_le_0": float((deltas <= 0).mean()),
        "n_boot": n_boot,
    }
    block_results = {}
    for block in block_lengths:
        block_deltas = np.empty(n_boot, dtype=np.float64)
        for r in range(n_boot):
            total = 0
            for name in names:
                n = len(y[name])
                starts = np.arange(0, n, block)
                chosen = rng.choice(starts, size=len(starts), replace=True)
                idx = np.concatenate([(np.arange(s, s + block) % n) for s in chosen])
                k = k_for(n, quota)
                total += _topk_hits(a_scores[name][idx], y[name][idx], k) - _topk_hits(b_scores[name][idx], y[name][idx], k)
            block_deltas[r] = total
        block_results[str(block)] = {
            "ci95": [float(np.percentile(block_deltas, 2.5)), float(np.percentile(block_deltas, 97.5))],
            "fraction_delta_le_0": float((block_deltas <= 0).mean()),
        }
    out["block_sensitivity"] = block_results
    return out

"""Causal replay helpers. Controller state changes only through propose/commit."""
from __future__ import annotations

from typing import Any

import numpy as np

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.eval.metrics import k_for
from bumbleclaw_v6.provenance import deterministic_topk


def _window_rates(actions: np.ndarray, window: int = 500) -> tuple[float | None, float | None]:
    if len(actions) < window:
        return None, None
    values = np.convolve(actions.astype(np.float64), np.ones(window) / window, mode="valid")
    return float(values.min()), float(values.max())


def simulate_stream(
    controller: Any,
    scores: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    *,
    quota: float = C.QUOTA,
    window: int = 500,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int8).reshape(-1)
    ids = np.asarray([str(v) for v in ids], dtype=object)
    if len(scores) != len(y) or len(scores) != len(ids):
        raise ValueError("scores, y and ids must align")
    actions = np.zeros(len(scores), dtype=bool)
    for i, value in enumerate(scores):
        proposal = bool(controller.propose(float(value)))
        controller.commit(f"replay-{i:08d}", proposal, value=float(value))
        actions[i] = proposal
    rights = int(actions.sum())
    k_fixed = k_for(len(y), quota)
    fixed_selected = deterministic_topk(scores, ids, k_fixed)
    matched_selected = deterministic_topk(scores, ids, rights) if rights else np.empty(0, dtype=np.int64)
    rate_min, rate_max = _window_rates(actions, window)
    return {
        "rights": rights,
        "rate": (rights / len(y)) if len(y) else None,
        "hits": int(y[actions].sum()) if rights else 0,
        "fixed_k_offline_hits": int(y[fixed_selected].sum()),
        "matched_count_offline_hits": int(y[matched_selected].sum()) if rights else 0,
        "rate_window_min": rate_min,
        "rate_window_max": rate_max,
        "n": int(len(y)),
        "K": k_fixed,
        "controller": controller.state() if hasattr(controller, "state") else {},
    }

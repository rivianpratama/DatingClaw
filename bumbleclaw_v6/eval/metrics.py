"""Strict binary-label metrics with explicit unknown and undefined cases."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.provenance import deterministic_topk


def k_for(n: int, quota: float = C.QUOTA) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= quota <= 1.0:
        raise ValueError("quota must be in [0,1]")
    return int(np.floor(quota * n))


def validate_binary_labels(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y).reshape(-1)
    if values.size == 0:
        raise ValueError("labels must not be empty")
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"metrics require binary labels in {{0,1}}; got {sorted(unique)}")
    return values.astype(np.int8, copy=False)


def legacy_binary_from_rating(rating: Iterable[float]) -> np.ndarray:
    """Named conversion for legacy 0/100 ratings; never implicit in metrics."""
    return (np.asarray(list(rating), dtype=np.float64) >= 50.0).astype(np.int8)


def lower_tail_mean(values: np.ndarray, alpha: float = 0.05) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0,1]")
    ordered = np.sort(values)
    mass = alpha * len(ordered)
    full = int(np.floor(mass))
    frac = mass - full
    if mass <= 0:
        return float(ordered[0])
    total = float(ordered[:full].sum())
    if frac > 0 and full < len(ordered):
        total += float(frac * ordered[full])
    return total / mass


def _safe_auc(y: np.ndarray, score: np.ndarray, max_fpr: float | None = None) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, score, max_fpr=max_fpr))
    except ValueError:
        return None


def deck_metrics(
    score: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    old_score: np.ndarray | None = None,
    quota: float = C.QUOTA,
    status: np.ndarray | None = None,
) -> dict[str, Any]:
    """Metrics for one deck. `ids` are used for deterministic score ties."""
    y = validate_binary_labels(y)
    scores = np.asarray(score, dtype=np.float64).reshape(-1)
    ids = np.asarray([str(v) for v in ids], dtype=object)
    n = len(y)
    if len(scores) != n or len(ids) != n:
        raise ValueError("score, y and ids must align")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    k = k_for(n, quota)
    selected_idx = deterministic_topk(scores, ids, k)
    selected = np.zeros(n, dtype=bool)
    selected[selected_idx] = True
    positives = y == 1
    hits = int(y[selected].sum())
    n_pos = int(positives.sum())
    fp = selected & ~positives
    fp_count = int(fp.sum())
    out: dict[str, Any] = {
        "N": n,
        "K": k,
        "n_pos": n_pos,
        "hits": hits,
        "selected_count": int(selected.sum()),
        "recall": (hits / n_pos) if n_pos else None,
        "precision": (hits / k) if k else None,
        "hits_q15": int(y[deterministic_topk(scores, ids, k_for(n, 0.15))].sum()),
        "hits_q25": int(y[deterministic_topk(scores, ids, k_for(n, 0.25))].sum()),
        "auc": _safe_auc(y, scores),
        "auc_fpr20": _safe_auc(y, scores, max_fpr=0.20),
        "fp_count": fp_count,
        "proxy_severe_fp": 0,
        "unknown_cost_selected": 0,
        "fp_mean_old_score": None,
        "cute_fp_share": None,
        "fp_lower_tail_mean": None,
        "known_cost_fp_count": 0,
    }
    if old_score is not None:
        old = np.asarray(old_score, dtype=np.float64).reshape(-1)
        if len(old) != n:
            raise ValueError("old_score must align with y")
        known = np.isfinite(old)
        fp_known = fp & known
        fp_unknown = fp & ~known
        out["unknown_cost_selected"] = int(fp_unknown.sum())
        out["known_cost_fp_count"] = int(fp_known.sum())
        out["proxy_severe_fp"] = int((fp_known & (old < C.CAT_FP_THRESHOLD)).sum())
        out["proxy_severe_rate"] = float(out["proxy_severe_fp"] / k) if k else None
        if fp_known.any():
            fp_scores = old[fp_known]
            out["fp_mean_old_score"] = float(fp_scores.mean())
            out["cute_fp_share"] = float((fp_scores >= C.CUTE_THRESHOLD).mean())
            out["fp_lower_tail_mean"] = lower_tail_mean(fp_scores, 0.05)
    if status is not None:
        status_values = np.asarray(status).reshape(-1)
        if len(status_values) != n:
            raise ValueError("status must align with y")
        out["status_counts"] = {str(v): int((status_values == v).sum()) for v in np.unique(status_values)}
    return out


def pooled_metrics(per_fold: dict[str, dict[str, Any]], folds: Iterable[str] | None = None) -> dict[str, Any]:
    names = list(folds or sorted(per_fold))
    hits = sum(int(per_fold[f]["hits"]) for f in names)
    n_pos = sum(int(per_fold[f]["n_pos"]) for f in names)
    fp_count = sum(int(per_fold[f]["fp_count"]) for f in names)
    known_fp = sum(int(per_fold[f].get("known_cost_fp_count", 0)) for f in names)
    severe = sum(int(per_fold[f].get("proxy_severe_fp", 0)) for f in names)
    unknown = sum(int(per_fold[f].get("unknown_cost_selected", 0)) for f in names)
    weighted_fp_mean = None
    if known_fp:
        weighted_fp_mean = sum(
            float(per_fold[f]["fp_mean_old_score"]) * int(per_fold[f]["known_cost_fp_count"])
            for f in names if per_fold[f].get("fp_mean_old_score") is not None
        ) / known_fp
    return {
        "folds": names,
        "hits": hits,
        "n_pos": n_pos,
        "recall": (hits / n_pos) if n_pos else None,
        "macro_recall": float(np.mean([per_fold[f]["recall"] for f in names if per_fold[f]["recall"] is not None])) if any(per_fold[f]["recall"] is not None for f in names) else None,
        "fp_count": fp_count,
        "proxy_severe_fp": severe,
        "unknown_cost_selected": unknown,
        "fp_mean_old_score": weighted_fp_mean,
    }

"""Expected-cost and optional lexicographic selection policies."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.eval.metrics import deck_metrics, k_for
from bumbleclaw_v6.provenance import deterministic_topk


class UnknownCostPolicy(str, Enum):
    ZERO = "zero"
    ONE = "one"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class SelectionResult:
    selected: np.ndarray
    feasible: bool
    reason: str = ""
    utility: np.ndarray | None = None


@dataclass
class PolicyChoice:
    lam: float | None
    selected: np.ndarray
    feasible: bool
    sweeps: list[dict[str, Any]]
    reason: str = ""


def scale_budget(budget_per_4540: int, n: int) -> int:
    if budget_per_4540 < 0 or n < 0:
        raise ValueError("budget and n must be non-negative")
    return int(np.floor(budget_per_4540 * n / C.DIAGNOSTIC_DECK_SIZE))


def cost_from_old_score(
    old_score: np.ndarray,
    *,
    threshold: float = C.CAT_FP_THRESHOLD,
    unknown_policy: UnknownCostPolicy = UnknownCostPolicy.ABSTAIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cost, known mask, eligible mask.

    Unknown old scores are never silently treated as safe under ABSTAIN.
    """
    old = np.asarray(old_score, dtype=np.float64).reshape(-1)
    known = np.isfinite(old)
    c = np.zeros_like(old, dtype=np.float64)
    c[known] = (old[known] < threshold).astype(np.float64)
    eligible = np.ones(len(old), dtype=bool)
    policy = UnknownCostPolicy(unknown_policy)
    if policy == UnknownCostPolicy.ONE:
        c[~known] = 1.0
    elif policy == UnknownCostPolicy.ABSTAIN:
        eligible &= known
    return c, known, eligible


def utility(p: np.ndarray, c: np.ndarray, lam: float) -> np.ndarray:
    probabilities = np.asarray(p, dtype=np.float64).reshape(-1)
    costs = np.asarray(c, dtype=np.float64).reshape(-1)
    if len(probabilities) != len(costs):
        raise ValueError("p and c must align")
    if not (0.0 <= probabilities).all() or not (probabilities <= 1.0).all():
        raise ValueError("p must be in [0,1]")
    if not (0.0 <= costs).all() or not (costs <= 1.0).all():
        raise ValueError("c must be in [0,1]")
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    return (probabilities - lam * (1.0 - probabilities) * costs).astype(np.float64)


def _select_from_eligible(
    values: np.ndarray,
    ids: np.ndarray,
    eligible: np.ndarray,
    k: int,
) -> SelectionResult:
    eligible_idx = np.flatnonzero(eligible)
    if len(eligible_idx) < k:
        return SelectionResult(np.empty(0, dtype=np.int64), False, "fewer eligible candidates than quota")
    order = deterministic_topk(values[eligible_idx], np.asarray(ids, dtype=object)[eligible_idx], k)
    return SelectionResult(eligible_idx[order].astype(np.int64), True)


def select_utility(
    p: np.ndarray,
    old_score: np.ndarray,
    ids: np.ndarray,
    k: int,
    lam: float,
    *,
    unknown_policy: UnknownCostPolicy = UnknownCostPolicy.ABSTAIN,
) -> SelectionResult:
    c, _, eligible = cost_from_old_score(old_score, unknown_policy=unknown_policy)
    u = utility(p, c, lam)
    return _select_from_eligible(u, ids, eligible, k)


def selected_metrics(selected: np.ndarray, y: np.ndarray, old_score: np.ndarray) -> dict[str, int]:
    selected = np.asarray(selected, dtype=np.int64)
    labels = np.asarray(y, dtype=np.int8).reshape(-1)
    old = np.asarray(old_score, dtype=np.float64).reshape(-1)
    fp = selected[labels[selected] == 0]
    known = np.isfinite(old[fp])
    return {
        "hits": int(labels[selected].sum()),
        "selected": int(len(selected)),
        "fp": int(len(fp)),
        "cat_fp": int((old[fp][known] < C.CAT_FP_THRESHOLD).sum()),
        "unknown_cost_selected": int((~known).sum()),
    }


def choose_lambda_on_p(
    p: np.ndarray,
    old_score: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    *,
    budget_per_4540: int,
    unknown_policy: UnknownCostPolicy = UnknownCostPolicy.ABSTAIN,
    grid: Iterable[float] = C.LAMBDA_GRID,
) -> PolicyChoice:
    """Choose a lambda on the P role only; ties prefer lower realized cost, then lower lambda."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    ids = np.asarray([str(v) for v in ids], dtype=object)
    n = len(y)
    k = k_for(n, C.QUOTA)
    budget = scale_budget(budget_per_4540, n)
    sweeps: list[dict[str, Any]] = []
    feasible: list[tuple[float, int, int, np.ndarray]] = []
    for lam in grid:
        c, _, eligible = cost_from_old_score(old_score, unknown_policy=unknown_policy)
        u = utility(p, c, lam)
        selection = _select_from_eligible(u, ids, eligible, k)
        if not selection.feasible:
            sweeps.append({"lambda": float(lam), "feasible": False, "reason": selection.reason,
                           "hits": None, "cat_fp": None, "selected": 0})
            continue
        m = selected_metrics(selection.selected, y, old_score)
        record = {
            "lambda": float(lam),
            "feasible": True,
            "hits": m["hits"],
            "cat_fp": m["cat_fp"],
            "unknown_cost_selected": m["unknown_cost_selected"],
            "selected": m["selected"],
            "budget": budget,
        }
        sweeps.append(record)
        if m["selected"] == k and m["cat_fp"] <= budget:
            feasible.append((float(lam), -m["hits"], m["cat_fp"], selection.selected))
    if not feasible:
        return PolicyChoice(None, np.empty(0, dtype=np.int64), False, sweeps, "INFEASIBLE")
    feasible.sort(key=lambda item: (item[1], item[2], item[0]))
    lam, _, _, selected = feasible[0]
    return PolicyChoice(lam, selected, True, sweeps)


def lexicographic_select(
    p: np.ndarray,
    quality: np.ndarray,
    ids: np.ndarray,
    k: int,
    tau_pos: float,
) -> SelectionResult:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(quality, dtype=np.float64).reshape(-1)
    ids = np.asarray([str(v) for v in ids], dtype=object)
    if len(p) != len(q) or len(p) != len(ids):
        raise ValueError("p, quality and ids must align")
    if len(p) < k:
        return SelectionResult(np.empty(0, dtype=np.int64), False, "fewer candidates than quota")
    preferred = np.flatnonzero(p >= tau_pos)
    remaining = np.flatnonzero(p < tau_pos)
    if len(preferred) >= k:
        order = deterministic_topk(p[preferred], ids[preferred], k)
        return SelectionResult(preferred[order].astype(np.int64), True)
    pref_order = deterministic_topk(p[preferred], ids[preferred], len(preferred))
    selected = list(preferred[pref_order].tolist())
    need = k - len(selected)
    # Tuple order for the filler: quality desc, then p desc, then ID.
    rem_p, rem_q, rem_ids = p[remaining], q[remaining], ids[remaining]
    order = np.lexsort((rem_ids, -rem_p, -rem_q))
    selected.extend(remaining[order[:need]].tolist())
    return SelectionResult(np.asarray(selected, dtype=np.int64), True)


def exposure_cap_select(
    p: np.ndarray,
    exposed: np.ndarray,
    ids: np.ndarray,
    k: int,
    cap: int,
) -> SelectionResult:
    """Exact best-policy for a deterministic cap on selected exposed candidates."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    exposed = np.asarray(exposed, dtype=bool).reshape(-1)
    ids = np.asarray([str(v) for v in ids], dtype=object)
    if len(p) != len(exposed) or len(p) != len(ids):
        raise ValueError("p, exposed and ids must align")
    if cap < 0:
        raise ValueError("cap must be non-negative")
    low_idx = np.flatnonzero(exposed)
    safe_idx = np.flatnonzero(~exposed)
    if len(low_idx) + len(safe_idx) < k:
        return SelectionResult(np.empty(0, dtype=np.int64), False, "fewer candidates than quota")
    low_order = deterministic_topk(p[low_idx], ids[low_idx], len(low_idx))
    safe_order = deterministic_topk(p[safe_idx], ids[safe_idx], len(safe_idx))
    low_idx, safe_idx = low_idx[low_order], safe_idx[safe_order]
    low_prefix = np.r_[0.0, np.cumsum(p[low_idx])]
    safe_prefix = np.r_[0.0, np.cumsum(p[safe_idx])]
    best: tuple[float, int, np.ndarray] | None = None
    min_low = max(0, k - len(safe_idx))
    max_low = min(k, len(low_idx), cap)
    if min_low > max_low:
        return SelectionResult(np.empty(0, dtype=np.int64), False, "exposure cap infeasible")
    for n_low in range(min_low, max_low + 1):
        n_safe = k - n_low
        total = low_prefix[n_low] + safe_prefix[n_safe]
        chosen = np.r_[low_idx[:n_low], safe_idx[:n_safe]]
        if best is None or total > best[0] + 1e-12:
            best = (float(total), n_low, chosen)
    if best is None:
        return SelectionResult(np.empty(0, dtype=np.int64), False, "exposure cap infeasible")
    return SelectionResult(np.sort(best[2]).astype(np.int64), True)

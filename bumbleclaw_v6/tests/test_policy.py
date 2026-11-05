from itertools import combinations

import numpy as np
import pytest

from bumbleclaw_v6.decision.policy import (
    UnknownCostPolicy,
    choose_lambda_on_p,
    cost_from_old_score,
    exposure_cap_select,
    lexicographic_select,
    scale_budget,
    select_utility,
    utility,
)


def test_utility_monotonic_and_exact():
    p = np.array([0.1, 0.5, 0.9])
    c = np.array([1.0, 1.0, 1.0])
    u = utility(p, c, 0.5)
    assert u[0] < u[1] < u[2]
    assert utility(np.array([1.0]), np.array([1.0]), 5.0)[0] == pytest.approx(1.0)


def test_invalid_utility_inputs():
    with pytest.raises(ValueError):
        utility(np.array([1.1]), np.array([0.0]), 1.0)
    with pytest.raises(ValueError):
        utility(np.array([0.5]), np.array([1.1]), 1.0)
    with pytest.raises(ValueError):
        utility(np.array([0.5]), np.array([1.0]), -1.0)


def test_unknown_cost_policies():
    old = np.array([60.0, np.nan])
    c, known, eligible = cost_from_old_score(old, unknown_policy=UnknownCostPolicy.ABSTAIN)
    assert c.tolist() == [0.0, 0.0]
    assert known.tolist() == [True, False]
    assert eligible.tolist() == [True, False]
    c2, _, eligible2 = cost_from_old_score(old, unknown_policy=UnknownCostPolicy.ONE)
    assert c2.tolist() == [0.0, 1.0]
    assert eligible2.tolist() == [True, True]


def test_abstain_can_be_infeasible():
    p = np.array([0.9, 0.8])
    old = np.array([np.nan, 90.0])
    result = select_utility(p, old, np.array(["a", "b"]), 2, 0.0, unknown_policy=UnknownCostPolicy.ABSTAIN)
    assert not result.feasible


def test_budget_scaling():
    assert scale_budget(10, 4540) == 10
    assert scale_budget(10, 2270) == 5
    assert scale_budget(10, 0) == 0


def test_lexicographic_priority():
    p = np.array([0.8, 0.2, 0.9])
    q = np.array([0.1, 0.99, 0.0])
    result = lexicographic_select(p, q, np.array(["a", "b", "c"]), 2, 0.7)
    assert result.feasible
    assert set(result.selected.tolist()) == {0, 2}


def test_exposure_cap_matches_bruteforce():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    exposed = np.array([True, True, True, False, False, False])
    ids = np.array([f"i{i}" for i in range(6)])
    result = exposure_cap_select(p, exposed, ids, 3, 1)
    assert result.feasible
    best = -1.0
    for combo in combinations(range(6), 3):
        if exposed[list(combo)].sum() <= 1:
            best = max(best, float(p[list(combo)].sum()))
    assert float(p[result.selected].sum()) == pytest.approx(best)


def test_exposure_cap_infeasible():
    p = np.array([0.9, 0.8])
    exposed = np.array([True, True])
    result = exposure_cap_select(p, exposed, np.array(["a", "b"]), 1, 0)
    assert not result.feasible


def test_choose_lambda_on_p_can_find_feasible_policy():
    n = 12
    p = np.r_[np.full(8, 0.05), np.full(4, 0.95)]
    y = np.r_[np.zeros(8, dtype=int), np.ones(4, dtype=int)]
    old = np.r_[np.full(8, 80.0), np.full(4, 20.0)]
    ids = np.array([f"i{i}" for i in range(n)])
    choice = choose_lambda_on_p(p, old, y, ids, budget_per_4540=0)
    assert choice.feasible
    assert choice.lam is not None
    # With a zero budget, no low-score card may be selected, even at a recall cost.
    assert choice.sweeps


def test_choose_lambda_reports_infeasible_when_all_are_exposed():
    p = np.r_[np.full(8, 0.05), np.full(2, 0.95)]
    y = np.zeros(10, dtype=int)
    old = np.full(10, 20.0)
    ids = np.array([f"i{i}" for i in range(10)])
    choice = choose_lambda_on_p(p, old, y, ids, budget_per_4540=0)
    assert not choice.feasible
    assert choice.reason == "INFEASIBLE"

import numpy as np
import pytest

from bumbleclaw_v6.eval.metrics import (
    deck_metrics,
    k_for,
    lower_tail_mean,
    legacy_binary_from_rating,
    pooled_metrics,
    validate_binary_labels,
)


def test_k_and_binary_validation():
    assert k_for(4540) == 944
    assert k_for(0) == 0
    with pytest.raises(ValueError):
        validate_binary_labels(np.array([0, 100]))
    assert legacy_binary_from_rating([0, 50, 100]).tolist() == [0, 1, 1]


def test_lower_tail_mean_is_not_percentile():
    assert lower_tail_mean(np.array([0.0, 1.0, 2.0, 100.0]), 0.5) == pytest.approx(0.5)
    assert lower_tail_mean(np.array([]), 0.05) is None


def test_ties_are_permutation_invariant():
    y = np.array([0, 1, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.1])
    ids = np.array(["b", "a", "c", "d"])
    first = deck_metrics(s, y, ids, quota=0.5)
    order = np.array([2, 0, 1, 3])
    second = deck_metrics(s[order], y[order], ids[order], quota=0.5)
    assert first["hits"] == second["hits"]
    assert first["recall"] == second["recall"]


def test_undefined_single_class_and_empty_selection():
    result = deck_metrics(np.array([0.2, 0.1]), np.array([0, 0]), np.array(["a", "b"]), quota=0.0)
    assert result["recall"] is None
    assert result["auc"] is None
    assert result["K"] == 0


def test_unknown_cost_is_counted_not_safe():
    y = np.array([0, 0, 1])
    s = np.array([0.9, 0.8, 0.7])
    ids = np.array(["a", "b", "c"])
    old = np.array([np.nan, 40.0, 90.0])
    result = deck_metrics(s, y, ids, old_score=old, quota=2 / 3)
    assert result["unknown_cost_selected"] == 1
    assert result["known_cost_fp_count"] == 1
    assert result["proxy_severe_fp"] == 1


def test_pooled_fp_mean_is_count_weighted():
    per = {
        "F2": {"hits": 1, "n_pos": 2, "recall": 0.5, "fp_count": 1, "known_cost_fp_count": 1,
               "fp_mean_old_score": 0.0, "proxy_severe_fp": 1, "unknown_cost_selected": 0},
        "F3": {"hits": 1, "n_pos": 1, "recall": 1.0, "fp_count": 9, "known_cost_fp_count": 9,
               "fp_mean_old_score": 100.0, "proxy_severe_fp": 0, "unknown_cost_selected": 0},
    }
    assert pooled_metrics(per, ["F2", "F3"])["fp_mean_old_score"] == pytest.approx(90.0)


def test_nonfinite_score_rejected():
    with pytest.raises(ValueError):
        deck_metrics(np.array([np.inf]), np.array([1]), np.array(["a"]))

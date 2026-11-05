import numpy as np

from bumbleclaw_v6.eval.stats import paired_bootstrap


def test_identical_models_have_zero_delta():
    y = {"F2": np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])}
    scores = {"F2": np.arange(10, dtype=float)}
    result = paired_bootstrap(scores, scores, y, folds=("F2",), n_boot=20, seed=1)
    assert result["delta_hits"] == 0
    assert result["ci95"] == [0.0, 0.0]
    assert result["fraction_delta_le_0"] == 1.0


def test_better_model_has_positive_a_minus_b_delta():
    y = {"F2": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])}
    a = {"F2": np.arange(10, dtype=float)}
    b = {"F2": np.zeros(10, dtype=float)}
    result = paired_bootstrap(a, b, y, folds=("F2",), n_boot=20, seed=1)
    assert result["delta_hits"] > 0
    assert result["delta_direction"] == "a_minus_b"
    assert "block_sensitivity" in result

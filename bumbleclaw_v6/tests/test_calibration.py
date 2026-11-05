import numpy as np
import pytest

from bumbleclaw_v6.decision.calibration import (
    CalibrationUnavailable,
    Calibrator,
    fit_platt,
    prior_shift,
    sigmoid,
)


def _fit(model="m1"):
    logits = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    return fit_platt(logits, y, model_hash=model, transform_hash="t1", recipe_hash="r1",
                     calibration_ids=np.array([f"s{i}" for i in range(len(y))]))


def test_stable_sigmoid_extremes():
    out = sigmoid(np.array([-1e4, 0.0, 1e4]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(1.0)


def test_platt_fit_and_binding():
    calibrator = _fit()
    p = calibrator.apply(np.array([-10.0, 0.0, 10.0]), model_hash="m1", transform_hash="t1", recipe_hash="r1")
    assert np.isfinite(p).all()
    assert p[0] < p[1] < p[2]
    with pytest.raises(ValueError):
        calibrator.apply([0.0], model_hash="other")
    with pytest.raises(ValueError):
        calibrator.apply([0.0], transform_hash="other")
    with pytest.raises(ValueError):
        calibrator.apply([0.0], recipe_hash="other")


def test_platt_requires_both_classes():
    with pytest.raises(CalibrationUnavailable):
        fit_platt(np.array([0.0, 1.0]), np.array([0, 0]), model_hash="m", transform_hash="t",
                  recipe_hash="r", calibration_ids=np.array(["a", "b"]))


def test_calibrator_roundtrip_and_prior_shift():
    calibrator = _fit()
    restored = Calibrator.from_dict(calibrator.to_dict())
    assert restored.a == calibrator.a
    assert restored.b == calibrator.b
    shifted = prior_shift(np.array([0.01, 0.5, 0.99]), 0.1, 0.05)
    expected_half = (0.5 * 0.5) / (0.5 * 0.5 + 0.5 * (0.95 / 0.9))
    assert shifted[1] == pytest.approx(expected_half, abs=1e-6)
    assert shifted[0] < 0.01 and shifted[-1] < 0.99

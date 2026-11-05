import numpy as np

from bumbleclaw_v6.features.parity import compare_blocks, parity_gate


def test_parity_identical_and_mismatch():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    report = compare_blocks({"arc": a}, {"arc": a.copy()})
    assert report["arc"]["median_cos"] == 1.0
    assert parity_gate(report)
    bad = compare_blocks({"arc": a}, {"arc": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)})
    assert not parity_gate(bad)


def test_parity_shape_mismatch_fails_gate():
    report = compare_blocks({"arc": np.zeros((2, 2))}, {"arc": np.zeros((2, 3))})
    assert report["arc"]["status"] == "shape_mismatch"
    assert not parity_gate(report)

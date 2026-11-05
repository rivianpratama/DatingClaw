import json

import numpy as np
import pytest

from bumbleclaw_v6.data.load import FeatureBlock
from bumbleclaw_v6.data.transforms import BlockSpec, FeatureTransform


def block(name, values, mask=None, unit=False):
    values = np.asarray(values, dtype=np.float32)
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    return FeatureBlock(name, values, np.asarray(mask, bool), np.zeros(len(values), np.uint8), unit)


def test_fit_median_does_not_use_test_rows():
    values = np.array([[2.0], [4.0], [1000.0]], dtype=np.float32)
    b = block("geom", values, np.ones(3, bool))
    transform = FeatureTransform.fit({"geom": b}, np.array([0, 1]), [BlockSpec("geom", "standardize")])
    before = transform.to_dict()
    changed = block("geom", np.array([[2.0], [4.0], [-1000.0]], dtype=np.float32), np.ones(3, bool))
    transform.transform({"geom": changed})
    assert transform.to_dict() == before
    assert transform.blocks[0].median.tolist() == [3.0]


def test_missing_standardized_value_uses_saved_median_and_indicator():
    values = np.array([[2.0], [4.0], [0.0]], dtype=np.float32)
    b = block("geom", values, np.array([True, True, False]))
    transform = FeatureTransform.fit({"geom": b}, np.array([0, 1]), [BlockSpec("geom", "standardize")])
    out, names = transform.transform({"geom": b})
    assert names[-1] == "geom__missing"
    assert out[-1, -1] == 1.0
    assert out[-1, 0] == pytest.approx(0.0)


def test_unit_missing_indicator_and_serialization():
    values = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    b = block("arc", values, np.array([True, False]), unit=True)
    transform = FeatureTransform.fit({"arc": b}, np.array([0, 1]), [BlockSpec("arc", "unit", scale_unit=True)])
    out, names = transform.transform({"arc": b})
    assert names[-1] == "arc__missing"
    assert out[0, 0] == pytest.approx(0.6 * np.sqrt(2))
    assert out[1, 0] == 0.0
    roundtrip = FeatureTransform.from_dict(json.loads(json.dumps(transform.to_dict())))
    out2, names2 = roundtrip.transform({"arc": b})
    assert names == names2
    assert np.allclose(out, out2)


def test_nonfinite_transform_output_rejected():
    finite = block("arc", np.array([[3.0, 4.0]], dtype=np.float32))
    transform = FeatureTransform.fit({"arc": finite}, np.array([0]), [BlockSpec("arc", "unit")])
    bad = block("arc", np.array([[np.inf, 4.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        transform.transform({"arc": bad})

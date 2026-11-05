from pathlib import Path

import numpy as np
import pytest

from bumbleclaw_v6.features.recipes import PHOTO_V1_RECIPE
from bumbleclaw_v6.features.store import FeatureStore


def test_store_roundtrip_and_completion(tmp_path: Path):
    ids = np.array(["a", "b"])
    store = FeatureStore.create(tmp_path / "s", ids, PHOTO_V1_RECIPE, {"arc": 2, "fv": 1})
    store.write_chunk(0, {"arc": np.ones((1, 2), np.float16), "fv": np.ones((1, 1), np.float16)},
                      {"arc": np.array([0], np.uint8), "fv": np.array([0], np.uint8)})
    store.write_chunk(1, {"arc": np.zeros((1, 2), np.float16), "fv": np.zeros((1, 1), np.float16)},
                      {"arc": np.array([1], np.uint8), "fv": np.array([0], np.uint8)})
    store.finalize()
    loaded = FeatureStore.load(tmp_path / "s", PHOTO_V1_RECIPE)
    assert loaded.manifest["complete"] is True
    assert loaded.values["arc"].shape == (2, 2)
    assert loaded.status["arc"].tolist() == [0, 1]


def test_store_rejects_wrong_recipe_and_incomplete(tmp_path: Path):
    ids = np.array(["a"])
    store = FeatureStore.create(tmp_path / "s", ids, PHOTO_V1_RECIPE, {"arc": 2})
    with pytest.raises(ValueError):
        FeatureStore.load(tmp_path / "s", PHOTO_V1_RECIPE)
    with pytest.raises(ValueError):
        FeatureStore.open(tmp_path / "s", type(PHOTO_V1_RECIPE)("other", ("arc",), ()))
    store.write_chunk(0, {"arc": np.ones((1, 2), np.float16)}, {"arc": np.array([0], np.uint8)})
    store.finalize()
    with pytest.raises(FileExistsError):
        FeatureStore.create(tmp_path / "s", ids, PHOTO_V1_RECIPE, {"arc": 2})


def test_store_rejects_wrong_chunk_start(tmp_path: Path):
    store = FeatureStore.create(tmp_path / "s", np.array(["a"]), PHOTO_V1_RECIPE, {"arc": 2})
    with pytest.raises(ValueError):
        store.write_chunk(1, {"arc": np.ones((1, 2), np.float16)}, {"arc": np.array([0], np.uint8)})

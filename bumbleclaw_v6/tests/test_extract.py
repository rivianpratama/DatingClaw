from pathlib import Path

import numpy as np

from bumbleclaw_v6.features.extract import FeatureBundle, ModelBundle, extract_image
from bumbleclaw_v6.features.recipes import FeatureRecipe
from bumbleclaw_v6.schemas import FeatureStatus


def test_decode_error_is_explicit(tmp_path: Path):
    recipe = FeatureRecipe("fv_only", ("fv",), ())
    bundle = extract_image(tmp_path / "missing.webp", recipe, ModelBundle())
    assert bundle.status["fv"] == FeatureStatus.DECODE_ERROR
    assert bundle.values["fv"].shape == (1,)


def test_unsupported_layout_is_explicit(tmp_path: Path):
    import cv2
    path = tmp_path / "landscape.png"
    cv2.imwrite(str(path), np.zeros((100, 200, 3), np.uint8))
    recipe = FeatureRecipe("fv_only", ("fv",), ())
    bundle = extract_image(path, recipe, ModelBundle())
    assert bundle.status["fv"] == FeatureStatus.UNSUPPORTED_LAYOUT


def test_in_memory_image_matches_path(tmp_path: Path):
    import cv2
    img = np.zeros((200, 100, 3), np.uint8)
    path = tmp_path / "portrait.png"
    cv2.imwrite(str(path), img)
    recipe = FeatureRecipe("fv_only", ("fv",), ())
    from_path = extract_image(path, recipe, ModelBundle())
    from_array = extract_image(None, recipe, ModelBundle(), image=img)
    assert from_path.status["fv"] == from_array.status["fv"] == FeatureStatus.OK
    assert np.array_equal(from_path.values["fv"], from_array.values["fv"])

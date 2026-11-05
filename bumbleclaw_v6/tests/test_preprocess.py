import numpy as np

from bumbleclaw_v6.features.preprocess import (
    PHOTO_V1,
    crop_portrait,
    layout_for_image,
    letterbox_full,
    siglip_inputs,
    top_view,
)


def test_layout_accepts_phone_portrait_and_rejects_landscape():
    assert layout_for_image(np.zeros((1480, 720, 3), np.uint8)) == "phone_portrait"
    assert layout_for_image(np.zeros((720, 1480, 3), np.uint8)) == "unsupported"


def test_crop_and_letterbox_shapes():
    img = np.zeros((1480, 720, 3), np.uint8)
    crop, valid = crop_portrait(img, np.array([100, 200, 300, 400], np.float32), 384)
    assert crop.shape == (384, 384, 3)
    assert valid == 1.0
    full = letterbox_full(img, 384)
    assert full.shape == (3, 384, 384)
    top = top_view(img, 384)
    assert top.shape == (3, 384, 384)


def test_siglip_inputs_flip_changes_pixels():
    img = np.arange(1480 * 720 * 3, dtype=np.uint8).reshape(1480, 720, 3)
    bbox = np.array([100, 200, 300, 400], np.float32)
    normal = siglip_inputs(img, bbox, PHOTO_V1)
    flipped = siglip_inputs(img, bbox, type(PHOTO_V1)("flip", "portrait", 1.8, True))
    assert normal.shape == flipped.shape
    assert not np.array_equal(normal, flipped)

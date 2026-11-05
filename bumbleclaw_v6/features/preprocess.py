"""Pure image preprocessing shared by offline extraction and runtime.

The legacy full-screenshot view is retained only for historical reproduction. The
default photo view is a portrait crop; it is not a proof that no UI text is visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PhotoView:
    name: str
    kind: str = "portrait"       # portrait | full | top
    scale: float = 1.8
    hflip: bool = False
    top_start: float = 0.078
    top_end: float = 0.60


PHOTO_V1 = PhotoView("sp_v6", "portrait", 1.8)
PHOTO_FLIP = PhotoView("sp_flip_v6", "portrait", 1.8, hflip=True)
PHOTO_TIGHT = PhotoView("sp_tight_v6", "portrait", 1.25)
PHOTO_TIGHT_FLIP = PhotoView("sp_tight_flip_v6", "portrait", 1.25, hflip=True)
PHOTO_WIDE = PhotoView("sp_wide_v6", "portrait", 3.0)
LEGACY_FULL = PhotoView("sf_v6", "full")
TEXT_LIGHT = PhotoView("stop_v6", "top", top_start=0.078, top_end=0.60)


def load_bgr(path: str | Path) -> np.ndarray | None:
    return cv2.imread(str(path))


def layout_for_image(img: np.ndarray) -> str:
    if img is None or img.ndim != 3:
        return "unsupported"
    h, w = img.shape[:2]
    if h <= w:
        return "unsupported"
    ratio = h / max(w, 1)
    if 1.8 <= ratio <= 2.3:
        return "phone_portrait"
    return "unsupported"


def letterbox_full(img_bgr: np.ndarray, target_size: int = 384) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    px = (target_size - nw) // 2
    py = (target_size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - 0.5) / 0.5).transpose(2, 0, 1)


def crop_portrait(img_bgr: np.ndarray, bbox: np.ndarray | None, target_size: int = 384,
                  scale: float = 1.8) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    if bbox is not None and float(bbox[0]) >= 0:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bw, bh = x2 - x1, y2 - y1
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2) - 0.05 * bh
        side = int(round(scale * max(bw, bh)))
        face_valid = 1.0
    else:
        cx = 0.5 * w
        cy = 0.35 * h
        side = int(round(0.65 * min(w, h)))
        face_valid = 0.0
    side = max(16, side)
    half = side // 2
    if side <= w:
        cx = np.clip(cx, half, w - (side - half))
    if side <= h:
        cy = np.clip(cy, half, h - (side - half))
    x_start, y_start = int(round(cx)) - half, int(round(cy)) - half
    x_end, y_end = x_start + side, y_start + side
    pad_left, pad_top = max(0, -x_start), max(0, -y_start)
    pad_right, pad_bottom = max(0, x_end - w), max(0, y_end - h)
    crop = img_bgr[max(0, y_start):min(h, y_end), max(0, x_start):min(w, x_end)]
    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
    if crop.size == 0:
        crop = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    else:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return crop, face_valid


def top_view(img_bgr: np.ndarray, target_size: int = 384, start: float = 0.078, end: float = 0.60) -> np.ndarray:
    h = img_bgr.shape[0]
    return letterbox_full(img_bgr[int(start * h):int(end * h)], target_size)


def siglip_inputs(img_bgr: np.ndarray, bbox: np.ndarray | None, view: PhotoView) -> np.ndarray:
    if view.kind == "full":
        x = letterbox_full(img_bgr, 384)
    elif view.kind == "top":
        x = top_view(img_bgr, 384, view.top_start, view.top_end)
    elif view.kind == "portrait":
        crop, _ = crop_portrait(img_bgr, bbox, 384, view.scale)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)
    else:
        raise ValueError(f"unknown view kind {view.kind}")
    if view.hflip:
        x = x[:, :, ::-1]
    return np.ascontiguousarray(x, dtype=np.float32)

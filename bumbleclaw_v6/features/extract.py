"""Lazy model loading and image-to-feature extraction.

Heavy libraries are imported only when a model is requested. The extractor returns
explicit masks/statuses; a decode failure is never converted into a valid black image.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bumbleclaw_v6.features.geometry import extract_16d_geometry_and_veto
from bumbleclaw_v6.features.preprocess import (
    PhotoView,
    crop_portrait,
    layout_for_image,
    load_bgr,
    siglip_inputs,
)
from bumbleclaw_v6.features.recipes import FeatureRecipe
from bumbleclaw_v6.schemas import FeatureStatus

_IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


@dataclass
class ModelBundle:
    device: str = "cuda"
    app: Any = None
    siglip: Any = None
    dinov3: Any = None
    siglip_model_name: str | None = None
    dinov3_path: str | None = None


@dataclass
class FeatureBundle:
    values: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    status: dict[str, FeatureStatus]
    meta: dict[str, Any] = field(default_factory=dict)


def _torch():
    import torch
    return torch


def _device() -> str:
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def insightface_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def siglip_model(name: str = "google/siglip-so400m-patch14-384"):
    from transformers import SiglipVisionModel
    torch = _torch()
    return SiglipVisionModel.from_pretrained(
        name, cache_dir=r"D:\BumbleClawHFCache", local_files_only=True,
        torch_dtype=torch.float16,
    ).to(_device()).eval()


def dinov3_model(path: str = r"C:\Users\Rivian\dinov3-vitb16-pretrain-lvd1689m"):
    from transformers import AutoModel
    return AutoModel.from_pretrained(path).to(_device()).eval()


def load_models(recipe: FeatureRecipe, *, need_arc: bool = True, need_siglip: bool = True,
                need_dinov3: bool = True) -> ModelBundle:
    bundle = ModelBundle(device=_device())
    if need_arc:
        bundle.app = insightface_app()
    if need_siglip:
        bundle.siglip = siglip_model(recipe.siglip_model)
        bundle.siglip_model_name = recipe.siglip_model
    if need_dinov3:
        bundle.dinov3 = dinov3_model(recipe.dinov3_path)
        bundle.dinov3_path = recipe.dinov3_path
    return bundle


def detect_face(app, img_bgr: np.ndarray) -> dict[str, Any]:
    faces = app.get(img_bgr) if img_bgr is not None else []
    if not faces:
        return {
            "arc": np.zeros(512, np.float32), "geom": np.zeros(16, np.float32),
            "bbox": np.full(4, -1.0, np.float32), "face_valid": 0,
            "det_score": 0.0, "arc_norm_raw": float("nan"), "status": FeatureStatus.NO_FACE,
        }
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    geom, info = extract_16d_geometry_and_veto(face, img_bgr)
    raw = face.embedding.astype(np.float32)
    norm = float(np.linalg.norm(raw))
    return {
        "arc": raw / (norm + 1e-8), "geom": geom, "bbox": np.asarray(info["bbox"], np.float32),
        "face_valid": 1, "det_score": float(face.det_score), "arc_norm_raw": norm,
        "status": FeatureStatus.OK,
    }


def siglip_embed(model, batch_chw: np.ndarray) -> np.ndarray:
    torch = _torch()
    import torch.nn.functional as F
    x = torch.tensor(batch_chw, dtype=torch.float16, device=_device())
    with torch.no_grad():
        out = model(pixel_values=x).pooler_output.float()
    return F.normalize(out, dim=-1).cpu().numpy()


def dinov3_embed(model, crops_bgr: list[np.ndarray]) -> np.ndarray:
    torch = _torch()
    import torch.nn.functional as F
    rgb = np.stack(crops_bgr)[..., ::-1].copy()
    x = torch.from_numpy(rgb).to(_device()).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(_IMNET_MEAN, device=_device())
    std = torch.tensor(_IMNET_STD, device=_device())
    x = (x - mean) / std
    with torch.no_grad():
        out = model(x).last_hidden_state
        cls = F.normalize(out[:, 0, :], dim=-1)
        patches = F.normalize(out[:, 5:, :], dim=-1)
        weights = F.softmax(torch.bmm(patches, cls.unsqueeze(-1)).squeeze(-1) / 0.15, dim=-1).unsqueeze(1)
        fg = F.normalize(torch.bmm(weights, patches).squeeze(1), dim=-1)
        return F.normalize(torch.cat([cls, fg], -1), dim=-1).cpu().numpy()


def _view_for_block(recipe: FeatureRecipe, block: str) -> PhotoView | None:
    if block == "sp":
        return recipe.view("sp_v6")
    if block == "sp_flip":
        return recipe.view("sp_flip_v6")
    if block == "sp_tight":
        return recipe.view("sp_tight_v6")
    if block == "sp_tight_tta":
        return recipe.view("sp_tight_v6")
    if block == "sp_wide":
        return recipe.view("sp_wide_v6")
    if block == "sf":
        return recipe.view("sf_v6")
    if block == "stop":
        return recipe.view("stop_v6")
    if block == "d3":
        return recipe.view("sp_v6")
    return None


def extract_image(path: str | Path | None, recipe: FeatureRecipe, models: ModelBundle,
                  *, image: np.ndarray | None = None) -> FeatureBundle:
    """Extract from a path, or from an already-decoded BGR array (avoids temp-file I/O)."""
    t0 = time.perf_counter()
    img = image if image is not None else load_bgr(path)
    if img is None:
        status = {block: FeatureStatus.DECODE_ERROR for block in recipe.blocks}
        return FeatureBundle(
            {block: np.zeros(_dim(block), np.float32) for block in recipe.blocks},
            {block: np.zeros(1, bool) for block in recipe.blocks},
            status,
            {"layout": "decode_error", "timings_ms": {"total": (time.perf_counter() - t0) * 1e3}},
        )
    layout = layout_for_image(img)
    if layout == "unsupported":
        status = {block: FeatureStatus.UNSUPPORTED_LAYOUT for block in recipe.blocks}
        return FeatureBundle(
            {block: np.zeros(_dim(block), np.float32) for block in recipe.blocks},
            {block: np.zeros(1, bool) for block in recipe.blocks},
            status,
            {"layout": layout, "timings_ms": {"total": (time.perf_counter() - t0) * 1e3}},
        )
    face = detect_face(models.app, img) if models.app is not None else {
        "arc": np.zeros(512, np.float32), "geom": np.zeros(16, np.float32),
        "bbox": np.full(4, -1.0, np.float32), "face_valid": 0, "det_score": 0.0,
        "arc_norm_raw": float("nan"), "status": FeatureStatus.NO_FACE,
    }
    bbox = face["bbox"]
    values: dict[str, np.ndarray] = {}
    masks: dict[str, bool] = {}
    status: dict[str, FeatureStatus] = {}
    timings: dict[str, float] = {}
    for block in recipe.blocks:
        if block == "arc":
            values[block] = face["arc"]
            masks[block] = bool(face["face_valid"])
            status[block] = face["status"]
        elif block == "geom":
            values[block] = face["geom"]
            masks[block] = bool(face["face_valid"])
            status[block] = face["status"]
        elif block == "fv":
            values[block] = np.array([float(face["face_valid"])], np.float32)
            masks[block] = True
            status[block] = FeatureStatus.OK
        elif block in {"sp", "sp_flip", "sp_tight", "sp_wide"}:
            if not face["face_valid"]:
                values[block] = np.zeros(_dim(block), np.float32)
                masks[block] = False
                status[block] = FeatureStatus.NO_FACE
                continue
            view = _view_for_block(recipe, block)
            if view is None:
                raise KeyError(f"no view for block {block}")
            t = time.perf_counter()
            x = siglip_inputs(img, bbox, view)[None]
            values[block] = siglip_embed(models.siglip, x)[0]
            masks[block] = True
            status[block] = FeatureStatus.OK
            timings[block] = (time.perf_counter() - t) * 1e3
        elif block == "sp_tight_tta":
            if not face["face_valid"]:
                values[block] = np.zeros(_dim(block), np.float32)
                masks[block] = False
                status[block] = FeatureStatus.NO_FACE
                continue
            t = time.perf_counter()
            base_view = recipe.view("sp_tight_v6")
            flip_view = recipe.view("sp_tight_flip_v6")
            x = siglip_inputs(img, bbox, base_view)[None]
            xf = siglip_inputs(img, bbox, flip_view)[None]
            a = siglip_embed(models.siglip, x)[0]
            b = siglip_embed(models.siglip, xf)[0]
            avg = a + b
            norm = float(np.linalg.norm(avg))
            values[block] = avg / max(norm, 1e-8)
            masks[block] = True
            status[block] = FeatureStatus.OK
            timings[block] = (time.perf_counter() - t) * 1e3
        elif block in {"sf", "stop"}:
            # Full-screenshot/text-light views do not require a detected face.
            view = _view_for_block(recipe, block)
            if view is None:
                raise KeyError(f"no view for block {block}")
            t = time.perf_counter()
            x = siglip_inputs(img, bbox, view)[None]
            values[block] = siglip_embed(models.siglip, x)[0]
            masks[block] = True
            status[block] = FeatureStatus.OK
            timings[block] = (time.perf_counter() - t) * 1e3
        elif block == "d3":
            if not face["face_valid"]:
                values[block] = np.zeros(_dim(block), np.float32)
                masks[block] = False
                status[block] = FeatureStatus.NO_FACE
                continue
            t = time.perf_counter()
            crop, _ = crop_portrait(img, bbox, 384, 1.8)
            values[block] = dinov3_embed(models.dinov3, [crop])[0]
            masks[block] = True
            status[block] = FeatureStatus.OK
            timings[block] = (time.perf_counter() - t) * 1e3
        else:
            raise KeyError(f"unsupported block {block}")
    timings["total"] = (time.perf_counter() - t0) * 1e3
    return FeatureBundle(values, masks, status, {
        "layout": layout, "bbox": bbox, "face_valid": int(face["face_valid"]),
        "det_score": float(face["det_score"]), "arc_norm_raw": float(face["arc_norm_raw"]),
        "timings_ms": timings,
    })


def _dim(block: str) -> int:
    return {"arc": 512, "geom": 16, "fv": 1, "sf": 1152, "sp": 1152, "sp_flip": 1152,
            "sp_tight": 1152, "sp_tight_flip": 1152, "sp_tight_tta": 1152,
            "sp_wide": 1152, "stop": 1152, "d3": 1536}[block]

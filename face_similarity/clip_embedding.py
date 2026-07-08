from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from face_similarity.clip_runtime import configure_clip_cache
from face_similarity.clip_store import DEFAULT_CLIP_MODEL


def get_clip_embedding(
    image_path: str | Path,
    *,
    model_name: str = DEFAULT_CLIP_MODEL,
    provider: str = "auto",
) -> np.ndarray:
    torch, processor, model, device = get_clip_model(model_name, provider)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.detach().cpu().numpy().astype(np.float32)[0]


@lru_cache(maxsize=4)
def get_clip_model(model_name: str, provider: str) -> tuple[Any, Any, Any, str]:
    torch, CLIPModel, CLIPProcessor = import_clip_dependencies()
    device = select_device(torch, provider)
    local_files_only = os.environ.get("BUMBLECLAW_CLIP_ALLOW_DOWNLOAD") != "1"
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    model.eval()
    return torch, processor, model, device


def import_clip_dependencies() -> tuple[Any, Any, Any]:
    configure_clip_cache()
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("CLIP prediction requires torch and transformers. Run: pip install -r requirements-clip.txt") from exc
    return torch, CLIPModel, CLIPProcessor


def select_device(torch: Any, provider: str) -> str:
    if provider == "cpu":
        return "cpu"
    if provider == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"

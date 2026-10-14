from __future__ import annotations

import os
from pathlib import Path
import sys

CLIP_VENV_PYTHON = Path(r"D:\DatingClawClipVenv\Scripts\python.exe")
HF_CACHE_DIR = Path(r"D:\DatingClawHFCache")
CLIP_METHODS = {"face_biased", "multimodal", "multimodalx", "multimodalx2", "multimodalx_original"}


def configure_clip_cache() -> None:
    cache = str(HF_CACHE_DIR)
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(HF_CACHE_DIR / "torch"))


def ensure_clip_runtime(method: str) -> None:
    configure_clip_cache()
    if method not in CLIP_METHODS:
        return
    if _same_path(Path(sys.executable), CLIP_VENV_PYTHON):
        return
    if os.environ.get("BUMBLECLAW_NO_CLIP_REEXEC") == "1":
        return
    if not CLIP_VENV_PYTHON.exists():
        raise RuntimeError(f"CLIP venv Python does not exist: {CLIP_VENV_PYTHON}")

    os.execv(str(CLIP_VENV_PYTHON), [str(CLIP_VENV_PYTHON), *sys.argv])


def _same_path(left: Path, right: Path) -> bool:
    return str(left.resolve()).casefold() == str(right.resolve()).casefold()

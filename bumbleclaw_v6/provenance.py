"""Deterministic hashing, identifiers, and safe artifact writes."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from bumbleclaw_v6 import config as C


def canonical_json(value: Any) -> str:
    """Stable JSON for hashing. Reject values that cannot be represented unambiguously."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sample_id_for_filename(filename: str) -> str:
    """Opaque local sample identifier; does not expose the filename."""
    return sha256_text("bumbleclaw-sample:" + filename)[:24]


def event_id_for_index(index: int) -> str:
    return f"e{int(index):08d}"


def hash_id_list(ids: Sequence[str] | np.ndarray) -> str:
    values = [str(v) for v in ids]
    return sha256_text("\n".join(values))


def id_codes(ids: Sequence[str] | np.ndarray) -> np.ndarray:
    """Stable uint64 codes for fast deterministic tie-breaking."""
    values = [str(v) for v in ids]
    if len(set(values)) != len(values):
        raise ValueError("sample IDs must be unique")
    return np.array(
        [int.from_bytes(hashlib.sha256(v.encode("utf-8")).digest()[:8], "big") for v in values],
        dtype=np.uint64,
    )


def deterministic_topk(score: Sequence[float] | np.ndarray, ids: Sequence[str] | np.ndarray, k: int) -> np.ndarray:
    """Top-k by descending score, then ascending opaque ID. Returns positions."""
    scores = np.asarray(score, dtype=np.float64)
    id_values = [str(v) for v in ids]
    if scores.ndim != 1 or len(scores) != len(id_values):
        raise ValueError("score and ids must be one-dimensional and equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k > len(scores):
        raise ValueError("k cannot exceed the number of candidates")
    codes = id_codes(id_values)
    order = np.lexsort((codes, -scores))
    return order[:k].astype(np.int64, copy=False)


def ensure_under(path: str | Path, root: str | Path) -> Path:
    """Resolve a target and require it to be inside the new artifact tree."""
    target = Path(path).expanduser().resolve()
    root_resolved = Path(root).expanduser().resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"artifact target {target} is outside {root_resolved}") from exc
    return target


def ensure_writable_new(path: str | Path, root: str | Path) -> Path:
    target = ensure_under(path, root)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    return target


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        # Windows can transiently lock the destination (AV/indexer/reader).
        # Retry the atomic replace instead of failing the whole extraction.
        last_error: PermissionError | None = None
        for attempt in range(12):
            try:
                os.replace(tmp_name, target)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(min(0.05 * (2 ** attempt), 2.0))
        assert last_error is not None
        raise last_error
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def file_manifest(paths: Iterable[str | Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        p = Path(path)
        if p.exists() and p.is_file():
            result[str(p)] = sha256_file(p)
    return result


def source_hashes() -> dict[str, str]:
    """Hash the files whose behavior this revision depends on, when present."""
    paths = [
        C.LABELS_CSV,
        C.SCORES_CSV,
        C.TRAIN_D2_NPZ,
        C.TRAIN_D3_NPZ,
        C.HOLDOUT_NPZ,
        Path(__file__).with_name("config.py"),
        Path(__file__).with_name("provenance.py"),
    ]
    return file_manifest(paths)


def split_fingerprint(role_ids: dict[str, Sequence[str]], protocol: str, group_policy: str) -> str:
    payload = {
        "protocol": protocol,
        "group_policy": group_policy,
        "roles": {name: [str(v) for v in role_ids[name]] for name in sorted(role_ids)},
    }
    return stable_hash(payload)

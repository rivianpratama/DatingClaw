"""Immutable run directories and ID-aligned prediction artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.provenance import atomic_write_json, ensure_writable_new, read_json, stable_hash


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunArtifacts:
    run_id: str
    root: Path
    manifest: dict[str, Any]

    @classmethod
    def create(cls, run_id: str, root: Path, manifest: dict[str, Any]) -> "RunArtifacts":
        if not run_id or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be a simple non-empty name")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = ensure_writable_new(root / run_id, root)
        run_dir.mkdir(parents=True)
        payload = {
            "schema": C.RUN_SCHEMA,
            "run_id": run_id,
            "created_utc": _now(),
            "status": "running",
            **manifest,
        }
        atomic_write_json(run_dir / "run_manifest.json", payload)
        (run_dir / "predictions").mkdir()
        (run_dir / "models").mkdir()
        return cls(run_id, run_dir, payload)

    @property
    def path(self) -> Path:
        return self.root

    def save_json(self, name: str, value: Any) -> Path:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("artifact name must be a simple filename")
        target = self.root / name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        atomic_write_json(target, value)
        return target

    def save_predictions(
        self,
        fold: str,
        role: str,
        seed: int,
        ids: np.ndarray,
        logits: np.ndarray,
        *,
        p: np.ndarray | None = None,
        selected: np.ndarray | None = None,
        status: np.ndarray | None = None,
    ) -> Path:
        ids = np.asarray([str(v) for v in ids], dtype=object)
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        if len(ids) != len(logits) or len(set(ids)) != len(ids):
            raise ValueError("prediction IDs must be unique and align with logits")
        payload: dict[str, np.ndarray] = {
            "sample_id": ids.astype("U"),
            "logit": logits,
            "fold": np.array(fold, dtype="U"),
            "role": np.array(role, dtype="U"),
            "seed": np.array(int(seed), dtype=np.int16),
        }
        if p is not None:
            payload["p"] = np.asarray(p, dtype=np.float32).reshape(-1)
        if selected is not None:
            payload["selected"] = np.asarray(selected, dtype=bool).reshape(-1)
        if status is not None:
            payload["status"] = np.asarray(status, dtype=np.int16).reshape(-1)
        target = self.root / "predictions" / f"{fold}_{role}_s{int(seed)}.npz"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite prediction {target}")
        np.savez_compressed(target, **payload)
        return target

    def finalize(self, summary: dict[str, Any]) -> None:
        self.save_json("summary.json", summary)
        complete = {
            "schema": C.RUN_SCHEMA,
            "run_id": self.run_id,
            "completed_utc": _now(),
            "artifact_hash": stable_hash({"manifest": self.manifest, "summary": summary}),
        }
        atomic_write_json(self.root / "COMPLETE.json", complete)
        manifest = dict(self.manifest)
        manifest["status"] = "complete"
        atomic_write_json(self.root / "run_manifest.json", manifest)

    def save_model(self, name: str, model: Any) -> Path:
        target = self.root / "models" / f"{name}.npz"
        model.save(target)
        return target

    def model_hash(self) -> str:
        files = sorted((self.root / "predictions").glob("*.npz"))
        return stable_hash([{"name": p.name, "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest()} for p in files])


def load_run(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)
    complete = path / "COMPLETE.json"
    if not complete.exists():
        raise ValueError(f"run is not complete: {path}")
    manifest = read_json(path / "run_manifest.json")
    summary = read_json(path / "summary.json")
    return {"path": path, "manifest": manifest, "summary": summary, "complete": read_json(complete)}


def load_predictions(run_dir: str | Path, fold: str, role: str, seed: int | None = None) -> dict[str, np.ndarray]:
    path = Path(run_dir) / "predictions"
    if seed is not None:
        files = [path / f"{fold}_{role}_s{int(seed)}.npz"]
    else:
        files = sorted(path.glob(f"{fold}_{role}_s*.npz"))
    if not files or not files[0].exists():
        raise FileNotFoundError(f"no predictions for {fold}/{role}")
    arrays = []
    for file in files:
        with np.load(file, allow_pickle=False) as z:
            arrays.append({key: z[key] for key in z.files})
        ids = arrays[-1]["sample_id"]
        if arrays[0]["sample_id"].tolist() != ids.tolist():
            raise ValueError("seed prediction IDs differ")
    merged = {key: np.stack([a[key] for a in arrays], axis=0) for key in arrays[0]}
    if len(arrays) == 1:
        merged = {key: value[0] for key, value in merged.items()}
    return merged


def align_prediction_ids(
    reference_ids: np.ndarray,
    ids: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    ref = [str(v) for v in reference_ids]
    got = [str(v) for v in ids]
    if ref != got:
        if set(ref) != set(got):
            raise ValueError("prediction IDs do not match reference ID set")
        lookup = {sample: i for i, sample in enumerate(got)}
        values = np.asarray(values)[[lookup[sample] for sample in ref]]
    return values

"""Crash-safe, recipe-bound feature stores."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bumbleclaw_v6.features.recipes import FeatureRecipe
from bumbleclaw_v6.provenance import atomic_write_json, hash_id_list, read_json, stable_hash


@dataclass
class FeatureStore:
    root: Path
    manifest: dict[str, Any]
    values: dict[str, np.ndarray]
    status: dict[str, np.ndarray]
    next_row: int

    @classmethod
    def create(cls, root: str | Path, sample_ids: np.ndarray, recipe: FeatureRecipe,
               dims: dict[str, int], dtype: str = "float16") -> "FeatureStore":
        path = Path(root)
        if path.exists():
            raise FileExistsError(f"feature store already exists: {path}")
        path.mkdir(parents=True)
        (path / "values").mkdir()
        (path / "status").mkdir()
        ids = np.asarray([str(v) for v in sample_ids], dtype=object)
        if len(set(ids)) != len(ids):
            raise ValueError("sample IDs must be unique")
        manifest = {
            "schema": "feature_store_v1",
            "recipe": recipe.name,
            "recipe_hash": recipe.hash,
            "sample_ids_sha256": hash_id_list(ids),
            "n": int(len(ids)),
            "dims": {k: int(v) for k, v in dims.items()},
            "dtype": dtype,
            "complete": False,
        }
        atomic_write_json(path / "manifest.json", manifest)
        values = {}
        status = {}
        for block, dim in dims.items():
            values[block] = np.lib.format.open_memmap(
                path / "values" / f"{block}.npy", mode="w+", dtype=np.dtype(dtype), shape=(len(ids), dim)
            )
            status[block] = np.lib.format.open_memmap(
                path / "status" / f"{block}.npy", mode="w+", dtype=np.uint8, shape=(len(ids),)
            )
        atomic_write_json(path / "progress.json", {"next_row": 0, "complete": False})
        return cls(path, manifest, values, status, 0)

    @classmethod
    def open(cls, root: str | Path, recipe: FeatureRecipe) -> "FeatureStore":
        path = Path(root)
        manifest = read_json(path / "manifest.json")
        if manifest.get("recipe_hash") != recipe.hash:
            raise ValueError("feature store recipe hash mismatch")
        progress = read_json(path / "progress.json")
        values = {block: np.load(path / "values" / f"{block}.npy", mmap_mode="r+")
                  for block in manifest["dims"]}
        status = {block: np.load(path / "status" / f"{block}.npy", mmap_mode="r+")
                  for block in manifest["dims"]}
        return cls(path, manifest, values, status, int(progress["next_row"]))

    def write_chunk(self, start: int, values: dict[str, np.ndarray], status: dict[str, np.ndarray]) -> None:
        if start != self.next_row:
            raise ValueError(f"expected chunk start {self.next_row}, got {start}")
        n = len(next(iter(values.values())))
        if n == 0:
            return
        for block, array in values.items():
            if block not in self.values:
                raise KeyError(f"unknown block {block}")
            arr = np.asarray(array)
            if arr.shape != (n, self.manifest["dims"][block]):
                raise ValueError(f"block {block} shape mismatch")
            self.values[block][start:start + n] = arr
            self.values[block].flush()
        for block, array in status.items():
            if block not in self.status:
                raise KeyError(f"unknown status block {block}")
            arr = np.asarray(array, dtype=np.uint8)
            if arr.shape != (n,):
                raise ValueError(f"status {block} shape mismatch")
            self.status[block][start:start + n] = arr
            self.status[block].flush()
        self.next_row += n
        atomic_write_json(self.root / "progress.json", {"next_row": self.next_row, "complete": False})

    def finalize(self) -> None:
        if self.next_row != self.manifest["n"]:
            raise ValueError(f"store incomplete: {self.next_row}/{self.manifest['n']}")
        manifest = dict(self.manifest)
        manifest["complete"] = True
        atomic_write_json(self.root / "manifest.json", manifest)
        atomic_write_json(self.root / "progress.json", {"next_row": self.next_row, "complete": True})
        atomic_write_json(self.root / "COMPLETE.json", {
            "schema": "feature_store_v1",
            "recipe_hash": self.manifest["recipe_hash"],
            "sample_ids_sha256": self.manifest["sample_ids_sha256"],
            "n": self.manifest["n"],
        })
        self.manifest = manifest

    @classmethod
    def load(cls, root: str | Path, recipe: FeatureRecipe) -> "FeatureStore":
        store = cls.open(root, recipe)
        if not (store.root / "COMPLETE.json").exists():
            raise ValueError("feature store is not complete")
        return store

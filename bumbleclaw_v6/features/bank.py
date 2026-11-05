"""Read-only feature access for a completed unified FeatureStore."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from bumbleclaw_v6.data.load import FeatureBlock, load_snapshot
from bumbleclaw_v6.features.recipes import FeatureRecipe
from bumbleclaw_v6.features.store import FeatureStore
from bumbleclaw_v6.provenance import hash_id_list
from bumbleclaw_v6.schemas import FeatureStatus


class FeatureStoreBank:
    """Expose a completed FeatureStore with the same block API as LegacyFeatureBank."""

    def __init__(self, snapshot_dir: str | Path, store_root: str | Path, recipe: FeatureRecipe):
        self.snapshot = load_snapshot(snapshot_dir)
        self.samples = self.snapshot["samples"]
        self.store = FeatureStore.load(store_root, recipe)
        self.ids = self.samples["sample_id"].astype(str).to_numpy()
        if hash_id_list(self.ids) != self.store.manifest["sample_ids_sha256"]:
            raise ValueError("feature store sample IDs do not match snapshot")
        self._blocks: dict[str, FeatureBlock] = {}
        for block in self.store.manifest["dims"]:
            values = np.asarray(self.store.values[block], dtype=np.float32)
            status = np.asarray(self.store.status[block], dtype=np.uint8)
            mask = status == int(FeatureStatus.OK)
            self._blocks[block] = FeatureBlock(
                block, values, mask, status,
                unit_norm=block in {"arc", "sp", "d3", "sf", "sp_flip", "sp_tight", "sp_tight_flip", "sp_tight_tta", "sp_wide", "stop"},
            )

    @property
    def recipe_hash(self) -> str:
        return str(self.store.manifest["recipe_hash"])

    def block(self, name: str) -> FeatureBlock:
        if name not in self._blocks:
            raise KeyError(f"block {name} is not in the feature store")
        return self._blocks[name]

    def blocks(self, names: Iterable[str]) -> dict[str, FeatureBlock]:
        return {name: self.block(name) for name in names}

    def coverage(self) -> dict[str, dict[str, int]]:
        out = {}
        for name, block in self._blocks.items():
            values = np.asarray(block.status)
            out[name] = {FeatureStatus(int(v)).name: int((values == v).sum()) for v in np.unique(values)}
        return out

"""Chronological split roles, with conservative group purging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from bumbleclaw_v6 import config as C
from bumbleclaw_v6.provenance import split_fingerprint


@dataclass(frozen=True)
class FoldRoles:
    name: str
    protocol: str
    role_indices: dict[str, np.ndarray]
    role_ids: dict[str, np.ndarray]
    fingerprint: str

    def indices(self, role: str) -> np.ndarray:
        return self.role_indices[role]

    def ids(self, role: str) -> np.ndarray:
        return self.role_ids[role]


def _fold_start(name: str) -> int:
    try:
        index = int(name[1:]) - 1
    except (ValueError, IndexError) as exc:
        raise KeyError(f"unknown fold {name!r}") from exc
    if index < 0 or index >= len(C.FOLD_STARTS):
        raise KeyError(f"unknown fold {name!r}")
    return C.FOLD_STARTS[index]


def _effective_groups(meta: pd.DataFrame) -> np.ndarray:
    ids = meta["sample_id"].astype(str).to_numpy()
    if "group_id" not in meta.columns:
        return np.array(["g:" + v for v in ids], dtype=object)
    raw = meta["group_id"].astype(str).to_numpy()
    return np.array([v if v and v not in {"-1", "nan", "None"} else "g:" + sample
                     for v, sample in zip(raw, ids)], dtype=object)


def _purge_against(role: np.ndarray, later: Iterable[np.ndarray], groups: np.ndarray) -> np.ndarray:
    later_groups = set()
    for arr in later:
        later_groups.update(groups[np.asarray(arr, dtype=np.int64)].tolist())
    if not later_groups:
        return role
    keep = np.array([g not in later_groups for g in groups[role]], dtype=bool)
    return role[keep]


def make_matched_roles(
    meta: pd.DataFrame,
    fold: str,
    *,
    purge: bool = True,
) -> FoldRoles:
    """Build A/H/C/P/T for one development deck.

    A = fit, H = hyperparameter/epoch selection, C = calibration,
    P = policy selection, T = test. Each earlier role is conservatively purged
    against later role groups; T is never modified.
    """
    s = _fold_start(fold)
    if s - 9080 < 0:
        raise ValueError("fold is too early for the matched five-role protocol")
    n = len(meta)
    if s + C.FOLD_LEN > n:
        raise ValueError("fold test window exceeds snapshot")
    base = {
        "A": np.arange(0, s - 9080, dtype=np.int64),
        "H": np.arange(s - 9080, s - C.FOLD_LEN, dtype=np.int64),
        "C": np.arange(s - C.FOLD_LEN, s - C.FOLD_LEN // 2, dtype=np.int64),
        "P": np.arange(s - C.FOLD_LEN // 2, s, dtype=np.int64),
        "T": np.arange(s, s + C.FOLD_LEN, dtype=np.int64),
    }
    groups = _effective_groups(meta)
    if purge:
        names = ["A", "H", "C", "P", "T"]
        original = {name: base[name].copy() for name in names}
        for i, name in enumerate(names[:-1]):
            base[name] = _purge_against(original[name], [original[n] for n in names[i + 1:]], groups)
    role_ids = {name: meta["sample_id"].astype(str).to_numpy()[idx] for name, idx in base.items()}
    fp = split_fingerprint(role_ids, C.SPLIT_SCHEMA, C.MATCHED_PURGE_POLICY if purge else "none")
    return FoldRoles(fold, C.SPLIT_SCHEMA, base, role_ids, fp)


def make_historical_roles(meta: pd.DataFrame, fold: str, *, purge: bool = True) -> FoldRoles:
    """Legacy rolling-origin roles for a named reproduction run."""
    s = _fold_start(fold)
    n = len(meta)
    if s + C.FOLD_LEN > n:
        raise ValueError("fold test window exceeds snapshot")
    base = {
        "train": np.arange(0, s, dtype=np.int64),
        "inner_val": np.arange(s - C.FOLD_LEN, s, dtype=np.int64),
        "test": np.arange(s, s + C.FOLD_LEN, dtype=np.int64),
    }
    groups = _effective_groups(meta)
    if purge:
        base["train"] = _purge_against(base["train"], [base["test"], base["inner_val"]], groups)
        base["inner_val"] = _purge_against(base["inner_val"], [base["test"]], groups)
    role_ids = {name: meta["sample_id"].astype(str).to_numpy()[idx] for name, idx in base.items()}
    fp = split_fingerprint(role_ids, C.HISTORICAL_SCHEMA, C.MATCHED_PURGE_POLICY if purge else "none")
    return FoldRoles(fold, C.HISTORICAL_SCHEMA, base, role_ids, fp)


def assert_disjoint(roles: FoldRoles, names: Iterable[str] | None = None) -> None:
    selected = list(names or roles.role_indices)
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            overlap = np.intersect1d(roles.role_indices[a], roles.role_indices[b])
            if len(overlap):
                raise AssertionError(f"roles {a} and {b} overlap by {len(overlap)} rows")

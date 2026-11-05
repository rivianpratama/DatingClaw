"""Unique-trial labeling state.

Filename keying is deliberately forbidden here: the same image may be presented
multiple times and each presentation must have its own trial/response identity.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bumbleclaw_v6.provenance import atomic_write_text


@dataclass(frozen=True)
class Trial:
    trial_id: str
    sample_id: str
    filename: str
    stratum: str = "default"
    repeat_group: str | None = None
    presentation_index: int = 0
    inclusion_probability: float = 1.0
    study_version: str = "v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrialStore:
    def __init__(self, trials: Iterable[Trial] = ()):
        self._trials: dict[str, Trial] = {}
        self._order: list[str] = []
        self._responses: list[dict[str, Any]] = []
        self._answered: set[str] = set()
        self._by_response_id: dict[str, dict[str, Any]] = {}
        self.add_trials(trials)

    @property
    def trials(self) -> list[Trial]:
        return [self._trials[tid] for tid in self._order]

    @property
    def responses(self) -> list[dict[str, Any]]:
        return list(self._responses)

    def add_trials(self, trials: Iterable[Trial]) -> None:
        for trial in trials:
            if not trial.trial_id:
                raise ValueError("trial_id must be non-empty")
            if trial.trial_id in self._trials:
                raise ValueError(f"duplicate trial_id {trial.trial_id}")
            if not (0 < trial.inclusion_probability <= 1):
                raise ValueError("inclusion_probability must be in (0,1]")
            self._trials[trial.trial_id] = trial
            self._order.append(trial.trial_id)

    def next_trial(self) -> Trial | None:
        for trial_id in self._order:
            if trial_id not in self._answered:
                return self._trials[trial_id]
        return None

    def record_response(self, trial_id: str, response_id: str, action: str,
                        timestamp: str | None = None) -> dict[str, Any]:
        if action not in ("left", "right"):
            raise ValueError("action must be 'left' or 'right'")
        if not response_id:
            raise ValueError("response_id must be non-empty")
        if trial_id not in self._trials:
            raise KeyError(f"unknown trial {trial_id}")
        if response_id in self._by_response_id:
            existing = self._by_response_id[response_id]
            if existing["action"] != action:
                raise ValueError("conflicting retry for the same response_id")
            return existing
        if trial_id in self._answered:
            raise ValueError("trial already answered")
        row = {
            "response_id": response_id,
            "trial_id": trial_id,
            "sample_id": self._trials[trial_id].sample_id,
            "action": action,
            "timestamp": timestamp or _now(),
            "repeat_group": self._trials[trial_id].repeat_group,
            "presentation_index": self._trials[trial_id].presentation_index,
        }
        self._responses.append(row)
        self._answered.add(trial_id)
        self._by_response_id[response_id] = row
        return row

    def undo_last(self) -> dict[str, Any] | None:
        if not self._responses:
            return None
        row = self._responses.pop()
        self._answered.discard(row["trial_id"])
        self._by_response_id.pop(row["response_id"], None)
        return row

    def pending_count(self) -> int:
        return len(self._trials) - len(self._answered)

    def answered_count(self) -> int:
        return len(self._answered)

    def to_rows(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self.trials]

    def save(self, path: str | Path) -> None:
        payload = {"trials": self.to_rows(), "responses": self._responses}
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TrialStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(Trial(**row) for row in payload["trials"])
        for row in payload.get("responses", []):
            store.record_response(row["trial_id"], row["response_id"], row["action"], row.get("timestamp"))
        return store


def build_repeat_schedule(
    items: list[dict[str, str]],
    *,
    repeat_count: int = 0,
    min_spacing: int = 100,
    study_version: str = "v1",
    seed: int = 0,
) -> list[Trial]:
    """Build a schedule with unique trial IDs and deterministic repeat placement.

    Repeats are inserted after `min_spacing` later trials; the same filename may
    appear more than once, but never with the same trial_id.
    """
    import numpy as np

    if repeat_count < 0 or min_spacing < 1:
        raise ValueError("invalid repeat schedule settings")
    if repeat_count > len(items):
        raise ValueError("repeat_count cannot exceed the number of unique items")
    rng = np.random.default_rng(seed)
    trials: list[Trial] = []
    for index, item in enumerate(items):
        trials.append(Trial(
            trial_id=f"t{index:06d}",
            sample_id=str(item["sample_id"]),
            filename=str(item["filename"]),
            stratum=str(item.get("stratum", "default")),
            repeat_group=str(item.get("repeat_group", item["sample_id"])),
            presentation_index=0,
            inclusion_probability=float(item.get("inclusion_probability", 1.0)),
            study_version=study_version,
        ))
    chosen = rng.choice(len(trials), size=repeat_count, replace=False) if repeat_count else []
    for repeat_index, source_index in enumerate(sorted(chosen.tolist())):
        source = trials[source_index]
        insert_at = min(len(trials), source_index + min_spacing + repeat_index)
        repeated = Trial(
            trial_id=f"r{repeat_index:06d}",
            sample_id=source.sample_id,
            filename=source.filename,
            stratum=source.stratum,
            repeat_group=source.repeat_group,
            presentation_index=source.presentation_index + 1,
            inclusion_probability=source.inclusion_probability,
            study_version=study_version,
        )
        trials.insert(insert_at, repeated)
    return trials

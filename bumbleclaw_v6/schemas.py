"""Small, explicit schemas shared by data, evaluation and decision code."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping


class FeatureStatus(IntEnum):
    OK = 0
    NO_FACE = 1
    AMBIGUOUS_PRIMARY_FACE = 2
    DECODE_ERROR = 3
    UNSUPPORTED_LAYOUT = 4
    EXTRACTION_ERROR = 5
    NOT_EXTRACTED = 6

    @classmethod
    def from_value(cls, value: Any) -> "FeatureStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls[value]
        return cls(int(value))


@dataclass(frozen=True)
class RoleNames:
    """The only roles permitted by the matched-pipeline protocol."""

    fit_a: str = "A"
    select_h: str = "H"
    calibrate_c: str = "C"
    policy_p: str = "P"
    test_t: str = "T"

    @property
    def all(self) -> tuple[str, ...]:
        return (self.fit_a, self.select_h, self.calibrate_c, self.policy_p, self.test_t)


ROLES = RoleNames()

REQUIRED_SAMPLE_COLUMNS = (
    "sample_id",
    "filename",
    "path",
    "source_store",
    "source_row",
    "label_order",
    "capture_order",
    "label_event_id",
    "y",
    "old_score",
    "old_score_method",
    "face_valid",
    "hard_veto",
    "group_id",
    "group_method",
)

REQUIRED_EVENT_COLUMNS = (
    "event_id",
    "sample_id",
    "timestamp",
    "action",
    "rating",
    "label_day",
)


def require_mapping_keys(mapping: Mapping[str, Any], required: tuple[str, ...], name: str) -> None:
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"{name} is missing required keys: {missing}")

import json
from pathlib import Path

import pandas as pd
import pytest

from bumbleclaw_v6.data.load import load_snapshot
from bumbleclaw_v6.schemas import REQUIRED_EVENT_COLUMNS, REQUIRED_SAMPLE_COLUMNS


def _write_snapshot(tmp_path: Path, samples: pd.DataFrame, events: pd.DataFrame | None = None,
                    conflicts: pd.DataFrame | None = None, count: int | None = None):
    events = events if events is not None else pd.DataFrame(columns=REQUIRED_EVENT_COLUMNS)
    conflicts = conflicts if conflicts is not None else pd.DataFrame(columns=["sample_id", "filename"])
    samples.to_csv(tmp_path / "samples.csv", index=False)
    events.to_csv(tmp_path / "label_events.csv", index=False)
    conflicts.to_csv(tmp_path / "label_conflicts.csv", index=False)
    manifest = {"counts": {"samples": len(samples) if count is None else count}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _samples(n=2):
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "filename": [f"f{i}" for i in range(n)],
        "path": [f"/tmp/f{i}" for i in range(n)],
        "source_store": ["train"] * n,
        "source_row": list(range(n)),
        "label_order": list(range(n)),
        "capture_order": list(range(n)),
        "label_event_id": ["e0"] * n,
        "y": [0, 1],
        "old_score": [60.0, 40.0],
        "old_score_method": ["m"] * n,
        "face_valid": [1, 1],
        "hard_veto": [0, 0],
        "group_id": [f"s{i}" for i in range(n)],
        "group_method": ["filename_only"] * n,
    })


def test_load_snapshot_schema(tmp_path: Path):
    _write_snapshot(tmp_path, _samples())
    loaded = load_snapshot(tmp_path)
    assert len(loaded["samples"]) == 2
    assert loaded["manifest"]["counts"]["samples"] == 2


def test_duplicate_sample_ids_rejected(tmp_path: Path):
    samples = _samples()
    samples.loc[1, "sample_id"] = samples.loc[0, "sample_id"]
    _write_snapshot(tmp_path, samples)
    with pytest.raises(ValueError):
        load_snapshot(tmp_path)


def test_manifest_count_mismatch_rejected(tmp_path: Path):
    _write_snapshot(tmp_path, _samples(), count=3)
    with pytest.raises(ValueError):
        load_snapshot(tmp_path)

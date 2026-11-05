from pathlib import Path

import pytest

from bumbleclaw_v6.labeling.trials import Trial, TrialStore, build_repeat_schedule


def test_repeated_image_is_not_removed_after_first_response(tmp_path: Path):
    store = TrialStore([
        Trial("a1", "sample_a", "a.jpg"),
        Trial("b1", "sample_b", "b.jpg"),
        Trial("a2", "sample_a", "a.jpg", repeat_group="sample_a", presentation_index=1),
    ])
    assert store.next_trial().trial_id == "a1"
    store.record_response("a1", "resp1", "right")
    assert store.next_trial().trial_id == "b1"
    store.record_response("b1", "resp2", "left")
    assert store.next_trial().trial_id == "a2"
    assert len(store.responses) == 2
    assert store.responses[0]["sample_id"] == "sample_a"
    assert store.responses[1]["trial_id"] == "b1"


def test_response_retry_is_idempotent_and_conflicts_raise():
    store = TrialStore([Trial("a1", "a", "a.jpg")])
    first = store.record_response("a1", "same", "right")
    second = store.record_response("a1", "same", "right")
    assert first == second
    with pytest.raises(ValueError):
        store.record_response("a1", "same", "left")
    with pytest.raises(ValueError):
        store.record_response("a1", "different", "right")


def test_undo_restores_trial_and_persistence_roundtrip(tmp_path: Path):
    store = TrialStore([Trial("a1", "a", "a.jpg"), Trial("a2", "a", "a.jpg", presentation_index=1)])
    store.record_response("a1", "r1", "left")
    assert store.undo_last()["response_id"] == "r1"
    assert store.next_trial().trial_id == "a1"
    store.record_response("a1", "r2", "right")
    path = tmp_path / "state.json"
    store.save(path)
    loaded = TrialStore.load(path)
    assert loaded.answered_count() == 1
    assert loaded.responses[0]["trial_id"] == "a1"


def test_schedule_has_unique_trial_ids_and_repeat_after_spacing():
    items = [
        {"sample_id": "a", "filename": "a.jpg"},
        {"sample_id": "b", "filename": "b.jpg"},
        {"sample_id": "c", "filename": "c.jpg"},
    ]
    schedule = build_repeat_schedule(items, repeat_count=1, min_spacing=2, seed=0)
    assert len(schedule) == 4
    assert len({t.trial_id for t in schedule}) == 4
    repeated = [t for t in schedule if t.presentation_index == 1]
    assert len(repeated) == 1
    assert repeated[0].sample_id in {"a", "b", "c"}

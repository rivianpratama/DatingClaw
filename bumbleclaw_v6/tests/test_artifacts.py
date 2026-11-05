from pathlib import Path

import numpy as np
import pytest

from bumbleclaw_v6.eval.artifacts import RunArtifacts, align_prediction_ids, load_predictions, load_run
from bumbleclaw_v6.models.linear import LinearModel


def test_immutable_run_and_predictions(tmp_path: Path):
    run = RunArtifacts.create("r1", tmp_path, {"purpose": "test"})
    ids = np.array(["a", "b", "c"])
    run.save_predictions("F5", "T", 0, ids, np.array([0.1, 0.2, 0.3]))
    run.finalize({"hits": 1})
    loaded = load_run(run.root)
    assert loaded["manifest"]["run_id"] == "r1"
    pred = load_predictions(run.root, "F5", "T", seed=0)
    assert pred["sample_id"].tolist() == ids.tolist()
    with pytest.raises(FileExistsError):
        RunArtifacts.create("r1", tmp_path, {})


def test_incomplete_run_is_rejected(tmp_path: Path):
    run = RunArtifacts.create("r2", tmp_path, {})
    with pytest.raises(ValueError):
        load_run(run.root)


def test_prediction_alignment_by_ids():
    ref = np.array(["a", "b", "c"])
    perm = np.array(["c", "a", "b"])
    values = np.array([3.0, 1.0, 2.0])
    aligned = align_prediction_ids(ref, perm, values)
    assert aligned.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        align_prediction_ids(ref, np.array(["a", "b"]), np.array([1.0, 2.0]))


def test_duplicate_prediction_ids_rejected(tmp_path: Path):
    run = RunArtifacts.create("r3", tmp_path, {})
    with pytest.raises(ValueError):
        run.save_predictions("F5", "T", 0, np.array(["a", "a"]), np.array([0.1, 0.2]))


def test_model_save_load(tmp_path: Path):
    model = LinearModel(np.array([1.0, 2.0], dtype=np.float32), 0.5)
    path = tmp_path / "m.npz"
    model.save(path)
    restored = LinearModel.load(path)
    assert np.allclose(restored.predict(np.array([[1.0, 2.0]], dtype=np.float32)), [5.5])
    with pytest.raises(FileExistsError):
        model.save(path)

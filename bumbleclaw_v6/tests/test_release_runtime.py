import json
from pathlib import Path

import numpy as np

from bumbleclaw_v6.data.load import FeatureBlock
from bumbleclaw_v6.data.transforms import BlockSpec, FeatureTransform
from bumbleclaw_v6.decision.calibration import Calibrator
from bumbleclaw_v6.eval.artifacts import RunArtifacts
from bumbleclaw_v6.features.extract import FeatureBundle
from bumbleclaw_v6.models.linear import LinearModel
from bumbleclaw_v6.provenance import stable_hash
from bumbleclaw_v6.release.build import build_release
from bumbleclaw_v6.release.validate import validate_release
from bumbleclaw_v6.runtime.scorer import V6Scorer
from bumbleclaw_v6.schemas import FeatureStatus


def _transform():
    blocks = {
        "arc": FeatureBlock("arc", np.array([[3.0, 4.0]], np.float32), np.array([True]), np.array([0], np.uint8)),
        "fv": FeatureBlock("fv", np.array([[1.0]], np.float32), np.array([True]), np.array([0], np.uint8)),
    }
    return FeatureTransform.fit(
        blocks, np.array([0]),
        [BlockSpec("arc", "unit", True, False), BlockSpec("fv", "passthrough", False, False)],
    )


def _release(tmp_path: Path, *, bad_calibrator: bool = False) -> Path:
    runs = tmp_path / "runs"
    transform = _transform()
    th = stable_hash(transform.to_dict())
    recipe = stable_hash({"recipe": "test"})
    model = LinearModel(np.array([0.5, 0.2, 0.1], np.float32), 0.0)
    base = RunArtifacts.create("base", runs, {"snapshot": "synthetic"})
    base.save_model("F5_s0", model)
    base.save_json("selection_F5_s0.json", {
        "config": "test", "keys": ["arc", "fv"], "fold": "F5", "seed": 0,
        "selected_weight_decay": 0.1, "selection_hits": 1,
        "model_file": "F5_s0.npz", "model_hash": model.hash,
        "transform": transform.to_dict(), "transform_hash": th,
        "feature_recipe_hash": recipe, "role_fingerprint": "synthetic",
    })
    base.finalize({"ok": True})
    policy = RunArtifacts.create("policy", runs, {"budget_per_4540": 10})
    cal = Calibrator(
        a=1.0, b=0.0, model_hash="bad" if bad_calibrator else model.hash,
        transform_hash=th, recipe_hash=recipe, calibration_ids_hash="cal",
        n_pos=1, n_neg=1, objective=0.1, converged=True, optimizer="test",
    )
    policy.save_json("policy_F5.json", {
        "calibrator": cal.to_dict(),
        "policy": {"lambda": 1.0, "feasible": True, "reason": "", "sweeps": [], "budget_p": 5},
        "test": {},
    })
    policy.finalize({"ok": True})
    return build_release(
        base_run=base.root, policy_run=policy.root, name="test_release",
        releases_root=tmp_path / "releases",
        warm_start=[0.0] * 10,
    )


def _bundle(status_arc=FeatureStatus.OK):
    return FeatureBundle(
        values={"arc": np.array([3.0, 4.0], np.float32), "fv": np.array([1.0], np.float32)},
        masks={"arc": True, "fv": True},
        status={"arc": status_arc, "fv": FeatureStatus.OK},
        meta={"timings_ms": {}},
    )


def test_release_build_and_validate(tmp_path: Path):
    release = _release(tmp_path)
    report = validate_release(release)
    assert report["PASS"], report
    assert (release / "model.npz").is_file()


def test_scorer_proposal_and_binding(tmp_path: Path):
    release = _release(tmp_path)
    scorer = V6Scorer(release, controller=True, feature_fn=lambda path: _bundle())
    result = scorer.decide("fake.jpg", old_score=60.0, event_id="e1")
    assert result["p"] is not None
    assert result["right"] is True
    assert result["u"] == result["p"]
    scorer.commit("e1", True)
    scorer.commit("e1", True)
    assert scorer.ctrl.rights == 1


def test_unknown_cost_abstains(tmp_path: Path):
    release = _release(tmp_path)
    scorer = V6Scorer(release, controller=True, feature_fn=lambda path: _bundle())
    result = scorer.decide("fake.jpg", old_score=None)
    assert result["right"] is None
    assert result["reason"] == "unknown_cost"


def test_feature_failure_abstains(tmp_path: Path):
    release = _release(tmp_path)
    scorer = V6Scorer(release, controller=True, feature_fn=lambda path: _bundle(FeatureStatus.DECODE_ERROR))
    result = scorer.decide("fake.jpg", old_score=60.0)
    assert result["right"] is None
    assert result["reason"] == "feature_failure"


def test_validate_rejects_wrong_calibrator_binding(tmp_path: Path):
    release = _release(tmp_path)
    decision_path = release / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["calibrator"]["model_hash"] = "wrong"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    report = validate_release(release)
    assert not report["PASS"]

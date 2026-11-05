import pytest

from bumbleclaw_v6.eval.prospective import build_manifest, validate_manifest


def test_build_manifest_hides_labels_and_freezes_release():
    release = {"release": "r", "release_dir": "D:/r", "release_manifest_sha256": "a" * 64,
               "model_hash": "m", "transform_hash": "t", "recipe_hash": "p", "keys": ["arc"]}
    payload = build_manifest(release, plan_name="p", target_cards=10)
    assert payload["labels_hidden"] is True
    assert payload["labels_revealed"] is False
    assert payload["labels"] == []
    assert payload["release"]["release_manifest_sha256"] == "a" * 64


def test_build_manifest_rejects_nonpositive_target():
    release = {"release": "r", "release_dir": "D:/r", "release_manifest_sha256": "a" * 64}
    with pytest.raises(ValueError):
        build_manifest(release, plan_name="p", target_cards=0)


def test_validate_manifest_rejects_revealed_labels(tmp_path):
    release = {"release": "r", "release_dir": str(tmp_path), "release_manifest_sha256": "a" * 64}
    payload = build_manifest(release, plan_name="p", target_cards=10)
    payload["labels_hidden"] = False
    path = tmp_path / "m.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_manifest(path)

from pathlib import Path

import numpy as np
import pytest

from bumbleclaw_v6.provenance import (
    atomic_write_text,
    canonical_json,
    deterministic_topk,
    ensure_writable_new,
    id_codes,
    sample_id_for_filename,
    stable_hash,
)


def test_canonical_json_and_stable_hash():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert stable_hash({"a": 2, "b": 1}) == stable_hash({"b": 1, "a": 2})
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_sample_ids_opaque_and_stable():
    one = sample_id_for_filename("profile_20260101_000000.webp")
    two = sample_id_for_filename("profile_20260101_000000.webp")
    assert one == two
    assert one != "profile_20260101_000000.webp"
    assert len(one) == 24


def test_deterministic_topk_permutation_invariant():
    scores = np.array([0.5, 0.5, 0.5, 0.1])
    ids = np.array(["b", "a", "c", "d"])
    first = deterministic_topk(scores, ids, 2)
    order = np.array([2, 0, 1, 3])
    second = deterministic_topk(scores[order], ids[order], 2)
    assert set(ids[first]) == set(ids[order][second])
    # Tie-breaking is by opaque ID code, not by the displayed string order.
    assert len(set(ids[first])) == 2


def test_deterministic_topk_validates():
    with pytest.raises(ValueError):
        deterministic_topk([1.0, np.nan], ["a", "b"], 1)
    with pytest.raises(ValueError):
        deterministic_topk([1.0], ["a"], 2)
    with pytest.raises(ValueError):
        id_codes(["a", "a"])


def test_artifact_target_guard(tmp_path: Path):
    allowed = tmp_path / "artifacts"
    assert ensure_writable_new(allowed / "new", allowed) == (allowed / "new").resolve()
    allowed.mkdir()
    (allowed / "new").mkdir()
    with pytest.raises(FileExistsError):
        ensure_writable_new(allowed / "new", allowed)
    with pytest.raises(ValueError):
        ensure_writable_new(tmp_path / "outside", allowed)


def test_atomic_write_retries_transient_permission_error(tmp_path, monkeypatch):
    import bumbleclaw_v6.provenance as prov

    real_replace = prov.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("transient lock")
        return real_replace(src, dst)

    monkeypatch.setattr(prov.os, "replace", flaky_replace)
    target = tmp_path / "progress.json"
    atomic_write_text(target, "ok\n")
    assert target.read_text(encoding="utf-8") == "ok\n"
    assert calls["n"] == 2

import numpy as np
import pandas as pd
import pytest

from bumbleclaw_v6.eval.splits import (
    assert_disjoint,
    make_historical_roles,
    make_matched_roles,
)


def synthetic_meta(n=41_000):
    ids = np.array([f"s{i:06d}" for i in range(n)], dtype=object)
    return pd.DataFrame({
        "sample_id": ids,
        "group_id": ids.copy(),
        "y": np.zeros(n, dtype=np.int8),
    })


def test_matched_roles_chronology():
    meta = synthetic_meta()
    roles = make_matched_roles(meta, "F2", purge=False)
    assert roles.indices("A")[0] == 0 and roles.indices("A")[-1] == 27_369
    assert roles.indices("H")[0] == 27_370 and roles.indices("H")[-1] == 31_909
    assert roles.indices("C")[0] == 31_910 and roles.indices("C")[-1] == 34_179
    assert roles.indices("P")[0] == 34_180 and roles.indices("P")[-1] == 36_449
    assert roles.indices("T")[0] == 36_450 and roles.indices("T")[-1] == 40_989
    assert_disjoint(roles)


def test_group_purge_removes_earlier_role_only():
    meta = synthetic_meta()
    meta.loc[0, "group_id"] = "shared"
    meta.loc[36_500, "group_id"] = "shared"
    roles = make_matched_roles(meta, "F2", purge=True)
    assert 0 not in roles.indices("A")
    assert 36_500 in roles.indices("T")
    assert_disjoint(roles)


def test_unknown_group_ids_are_not_one_giant_group():
    meta = synthetic_meta()
    meta.loc[0, "group_id"] = "-1"
    meta.loc[36_500, "group_id"] = "-1"
    roles = make_matched_roles(meta, "F2", purge=True)
    assert 0 in roles.indices("A")


def test_historical_roles():
    meta = synthetic_meta()
    roles = make_historical_roles(meta, "F2", purge=False)
    assert roles.indices("train")[-1] == 36_449
    assert roles.indices("inner_val")[0] == 31_910
    assert roles.indices("test")[0] == 36_450
    assert len(roles.indices("test")) == 4_540
    assert roles.fingerprint


def test_bad_fold_and_early_fold():
    meta = synthetic_meta()
    with pytest.raises(KeyError):
        make_matched_roles(meta, "F9")
    with pytest.raises(ValueError):
        make_matched_roles(synthetic_meta(10_000), "F1")

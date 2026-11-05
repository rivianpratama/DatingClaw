import numpy as np

from bumbleclaw_v6.experiments.full_release import grouped_folds


def test_grouped_folds_never_split_a_group():
    groups = np.array(["a", "a", "b", "c", "c", "d", "e", "e"])
    folds = grouped_folds(groups, n_folds=3, seed=0)
    for group in np.unique(groups):
        assert len(set(folds[groups == group].tolist())) == 1


def test_grouped_folds_cover_every_row():
    groups = np.array(["a", "b", "c", "d", "e", "f"])
    folds = grouped_folds(groups, n_folds=2, seed=1)
    assert len(folds) == len(groups)
    assert set(folds.tolist()) == {0, 1}

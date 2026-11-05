import numpy as np

from bumbleclaw_v6.models.convex_probe import fit_sklearn_logreg, predict_logits


def test_converged_probe_ranks_separable_data():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(-1, 0.2, size=(20, 2)), rng.normal(1, 0.2, size=(20, 2))].astype(np.float32)
    y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    clf = fit_sklearn_logreg(X, y, C_reg=1.0, max_iter=200)
    logits = predict_logits(clf, X)
    assert np.isfinite(logits).all()
    assert logits[y == 1].mean() > logits[y == 0].mean()

import numpy as np

from bumbleclaw_v6.models.linear import fit_linear_logreg


def test_linear_probe_learns_separable_direction():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(-1, 0.2, size=(30, 3)), rng.normal(1, 0.2, size=(30, 3))].astype(np.float32)
    y = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
    model = fit_linear_logreg(X, y, weight_decay=0.01, seed=0, epochs=30, lr=1e-2, batch_size=16)
    logits = model.predict(X)
    assert np.isfinite(logits).all()
    assert logits[y == 1].mean() > logits[y == 0].mean()

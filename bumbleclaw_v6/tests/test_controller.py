import numpy as np
import pytest

from bumbleclaw_v6.decision.controller import KnownHorizonController, RollingQuotaController
from bumbleclaw_v6.eval.replay import simulate_stream


def test_commit_is_idempotent_and_conflict_checked():
    controller = RollingQuotaController(target=0.2, window=5, warm_start=np.zeros(5))
    controller.commit("a", True, value=1.0)
    controller.commit("a", True, value=1.0)
    assert controller.rights == 1
    with pytest.raises(ValueError):
        controller.commit("a", False, value=1.0)


def test_propose_does_not_change_state():
    controller = RollingQuotaController(target=0.2, window=5, warm_start=np.zeros(5))
    before = controller.state()
    controller.propose(1.0)
    assert controller.state() == before


def test_future_spike_does_not_alter_past_decisions():
    prefix = np.array([0.1, 0.2, 0.3, 0.4])
    a = np.r_[prefix, 100.0]
    b = np.r_[prefix, -100.0]
    first = RollingQuotaController(target=0.2, window=3, warm_start=np.array([0.0, 0.1, 0.2]))
    second = RollingQuotaController(target=0.2, window=3, warm_start=np.array([0.0, 0.1, 0.2]))
    for i in range(4):
        pa, pb = first.propose(a[i]), second.propose(b[i])
        assert pa == pb
        first.commit(f"e{i}", pa, value=a[i])
        second.commit(f"e{i}", pb, value=b[i])
    assert first.state()["history"] == second.state()["history"]


def test_known_horizon_reaches_exact_count():
    controller = KnownHorizonController(K=3, N=10, threshold=0.5)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    replay = simulate_stream(controller, scores, np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), np.array([f"i{i}" for i in range(10)]))
    assert replay["rights"] == 3
    assert replay["rights"] == 3  # controller K, not the offline floor(.208*N)


def test_known_horizon_early_termination_reports_shortfall():
    controller = KnownHorizonController(K=5, N=10, threshold=0.0)
    for i in range(3):
        action = controller.propose(0.1)
        controller.commit(f"e{i}", action, value=0.1)
    assert controller.rights < 5
    assert controller.remaining > 0


def test_replay_reports_matched_count_reference():
    controller = KnownHorizonController(K=2, N=10, threshold=0.0)
    scores = np.arange(10, dtype=float)
    y = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    replay = simulate_stream(controller, scores, y, np.array([f"i{i}" for i in range(10)]))
    assert replay["rights"] == 2
    assert "matched_count_offline_hits" in replay
    assert "fixed_k_offline_hits" in replay


def test_scorer_commit_records_utility_not_action():
    from types import SimpleNamespace
    from bumbleclaw_v6.runtime.scorer import V6Scorer
    scorer = V6Scorer.__new__(V6Scorer)
    scorer.ctrl = RollingQuotaController(target=0.2, window=5, warm_start=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    scorer._pending_utility = {"e1": 0.37}
    scorer.commit("e1", True)
    assert list(scorer.ctrl._history)[-1] == 0.37
    with pytest.raises(KeyError):
        scorer.commit("never_decided", True)

"""Streaming quota controllers with separate propose/commit semantics."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RollingQuotaController:
    target: float = 0.208
    window: int = 300
    half_life: float = 50.0
    kp: float = 2.0
    q_min: float = 0.05
    q_max: float = 0.5
    warm_start: np.ndarray | None = None
    _history: deque[float] = field(init=False)
    _seen: dict[str, bool] = field(default_factory=dict, init=False)
    rights: int = 0
    decisions: int = 0
    ema_rate: float = field(init=False)

    def __post_init__(self) -> None:
        if not 0 < self.target < 1 or self.window < 1 or self.half_life <= 0:
            raise ValueError("invalid controller settings")
        self._history = deque(maxlen=int(self.window))
        if self.warm_start is not None:
            self._history.extend(float(v) for v in np.asarray(self.warm_start).reshape(-1)[-self.window:])
        self.ema_rate = float(self.target)
        self.alpha = 1.0 - 0.5 ** (1.0 / self.half_life)

    def threshold(self) -> float:
        if not self._history:
            return -np.inf
        q = float(np.clip(self.target + self.kp * (self.target - self.ema_rate), self.q_min, self.q_max))
        return float(np.quantile(np.fromiter(self._history, dtype=float), 1.0 - q))

    def propose(self, value: float) -> bool:
        return bool(value >= self.threshold())

    def commit(self, event_id: str, action: bool, value: float | None = None) -> None:
        if event_id in self._seen:
            if self._seen[event_id] != bool(action):
                raise ValueError("conflicting duplicate commit for event")
            return
        self._seen[event_id] = bool(action)
        self._history.append(float(action if value is None else value))
        self.decisions += 1
        self.rights += int(bool(action))
        self.ema_rate = (1.0 - self.alpha) * self.ema_rate + self.alpha * float(bool(action))

    def state(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "window": self.window,
            "half_life": self.half_life,
            "kp": self.kp,
            "rights": self.rights,
            "decisions": self.decisions,
            "ema_rate": self.ema_rate,
            "history": list(self._history),
        }


@dataclass
class KnownHorizonController:
    """Exact final-count controller for a declared, uninterrupted N-card session."""

    K: int
    N: int
    threshold: float
    _seen: dict[str, bool] = field(default_factory=dict, init=False)
    rights: int = 0
    decisions: int = 0

    def __post_init__(self) -> None:
        if self.N < 0 or self.K < 0 or self.K > self.N:
            raise ValueError("require 0 <= K <= N")
        self.rights = 0
        self.decisions = 0

    @property
    def remaining(self) -> int:
        return self.N - self.decisions

    def propose(self, score: float) -> bool:
        if self.remaining <= 0:
            return False
        needs = self.K - self.rights
        if needs <= 0:
            return False
        if needs >= self.remaining:
            return True
        return bool(score >= self.threshold)

    def commit(self, event_id: str, action: bool, value: float | None = None) -> None:
        if event_id in self._seen:
            if self._seen[event_id] != bool(action):
                raise ValueError("conflicting duplicate commit for event")
            return
        if self.decisions >= self.N:
            raise ValueError("session has no remaining actions")
        self._seen[event_id] = bool(action)
        self.decisions += 1
        self.rights += int(bool(action))

    def state(self) -> dict[str, Any]:
        return {"K": self.K, "N": self.N, "rights": self.rights, "decisions": self.decisions,
                "remaining": self.remaining, "threshold": self.threshold}


# Compatibility alias used by the release/runtime contract and older reference code.
OnlineQuotaController = RollingQuotaController

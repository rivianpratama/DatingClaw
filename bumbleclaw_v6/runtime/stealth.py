"""Human-like pacing: swipe ASAP; every 2-3 min, take 1-3 short pauses, forever."""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class StealthSchedule:
    """Swipe ASAP; after each 2-3 min active interval, take 1-3 pauses of 1-3 s.

    The cycle repeats for the whole session. There is no session-wide pause cap.
    """
    every_min: float = 120.0
    every_max: float = 180.0
    delay_min: float = 1.0
    delay_max: float = 3.0
    burst_min: int = 1
    burst_max: int = 3
    _next_pause_at: float | None = field(default=None, init=False, repr=False)
    _remaining_pauses: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        for low, high, name in (
            (self.every_min, self.every_max, "every"),
            (self.delay_min, self.delay_max, "delay"),
        ):
            if low < 0 or high < low:
                raise ValueError(f"invalid stealth {name} range: {low}..{high}")
        if self.burst_min < 0 or self.burst_max < self.burst_min:
            raise ValueError("invalid stealth burst range")

    def delay(self, now: float, rng: random.Random | None = None) -> float:
        """Return 0.0 (swipe now) or a 1-3 s pause. Repeats every 2-3 min."""
        r = rng or random
        if self._next_pause_at is None:
            self._next_pause_at = now + r.uniform(self.every_min, self.every_max)
        if now < self._next_pause_at:
            return 0.0
        if self._remaining_pauses <= 0:
            self._remaining_pauses = r.randint(self.burst_min, self.burst_max)
        pause = r.uniform(self.delay_min, self.delay_max)
        self._remaining_pauses -= 1
        if self._remaining_pauses > 0:
            # Next call should pause again immediately after this sleep.
            self._next_pause_at = now + pause
        else:
            # Burst done; next pause only after another 2-3 min of active swiping.
            self._next_pause_at = now + pause + r.uniform(self.every_min, self.every_max)
        return pause

    @property
    def next_pause_at(self) -> float | None:
        return self._next_pause_at

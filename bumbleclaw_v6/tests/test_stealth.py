import random

import pytest

from bumbleclaw_v6.runtime.stealth import StealthSchedule


def test_no_delay_before_first_interval():
    s = StealthSchedule()
    r = random.Random(0)
    assert s.delay(now=0.0, rng=r) == 0.0
    assert 120.0 <= s.next_pause_at <= 180.0


def test_burst_is_one_to_three_pauses_of_one_to_three_seconds():
    s = StealthSchedule()
    r = random.Random(1)
    s.delay(now=0.0, rng=r)
    pauses = []
    t = 10_000.0
    for _ in range(10):
        d = s.delay(now=t, rng=r)
        if d:
            pauses.append(d)
            t += d + 0.1
        else:
            break
    assert 1 <= len(pauses) <= 3
    assert all(1.0 <= d <= 3.0 for d in pauses)


def test_cycle_repeats_after_each_burst():
    s = StealthSchedule()
    r = random.Random(7)
    s.delay(now=0.0, rng=r)
    t = 10_000.0
    bursts = 0
    pauses = 0
    for _ in range(200):
        d = s.delay(now=t, rng=r)
        if d:
            pauses += 1
            t += d + 0.1
        else:
            t += 200.0  # jump into the next active interval
            if pauses:
                bursts += 1
                pauses = 0
    assert bursts >= 3  # not capped at 1-3 for the session
    assert s._remaining_pauses >= 0


def test_invalid_ranges_rejected():
    with pytest.raises(ValueError):
        StealthSchedule(every_min=3.0, every_max=1.0)
    with pytest.raises(ValueError):
        StealthSchedule(delay_min=-1.0)
    with pytest.raises(ValueError):
        StealthSchedule(burst_min=3, burst_max=1)

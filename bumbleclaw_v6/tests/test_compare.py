from bumbleclaw_v6.eval.compare import _promote_diagnostic


def test_promotion_diagnostic_treats_a_as_candidate():
    per_a = {"F5": {"hits": 172}}
    per_b = {"F5": {"hits": 163}}
    paired = {"delta_hits": 9, "ci95": [1.0, 17.0]}
    assert _promote_diagnostic(paired, per_a, per_b, ("F5",)) is True

    inverted = {"delta_hits": -9, "ci95": [-17.0, 1.0]}
    assert _promote_diagnostic(inverted, per_a, per_b, ("F5",)) is False

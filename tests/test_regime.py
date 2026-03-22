"""Tests for RegimeState multipliers after bear-market fix."""
from bot.execution.regime import RegimeState, REGIME_MULTIPLIERS


def test_bear_multiplier_is_not_zero():
    """BEAR_TREND must never block trades (was 0.0, now 0.35)."""
    assert RegimeState.BEAR_TREND.size_multiplier == 0.35


def test_bull_multiplier_unchanged():
    assert RegimeState.BULL_TREND.size_multiplier == 1.0


def test_sideways_multiplier_unchanged():
    assert RegimeState.SIDEWAYS.size_multiplier == 0.5


def test_regime_multipliers_dict_has_all_states():
    for state in RegimeState:
        assert state in REGIME_MULTIPLIERS


def test_no_multiplier_is_zero():
    """No regime should ever produce a zero multiplier."""
    for state in RegimeState:
        assert state.size_multiplier > 0

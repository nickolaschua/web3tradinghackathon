"""Test that regime multiplier gate in risk.py allows bear-market trades."""
from bot.execution.risk import RiskManager, RiskDecision


def test_bear_regime_passes_gate():
    """0.35 regime multiplier must pass Gate 2 (was blocked at == 0.0)."""
    config = {
        "max_positions": 5,
        "hard_stop_pct": 0.05,
        "atr_stop_multiplier": 10.0,
        "trailing_stop_multiplier": 10.0,
        "risk_per_trade_pct": 0.02,
        "expected_win_loss_ratio": 1.5,
        "circuit_breaker": {
            "halt_threshold": 0.30,
            "reduce_heavy_threshold": 0.20,
            "reduce_light_threshold": 0.10,
        },
    }
    rm = RiskManager(config)
    rm._portfolio_hwm = 1_000_000

    result = rm.size_new_position(
        pair="BTC/USD",
        current_price=70000.0,
        current_atr=700.0,
        free_balance_usd=950_000.0,
        open_positions={},
        regime_multiplier=0.35,
        confidence=0.65,
        portfolio_weight=0.5,
    )
    assert result.decision == RiskDecision.APPROVED, (
        f"Expected APPROVED, got {result.decision}: {result.reason}"
    )


def test_near_zero_regime_blocked():
    """Multipliers below 0.10 should still be blocked."""
    config = {
        "max_positions": 5,
        "hard_stop_pct": 0.05,
        "atr_stop_multiplier": 10.0,
        "trailing_stop_multiplier": 10.0,
        "risk_per_trade_pct": 0.02,
        "expected_win_loss_ratio": 1.5,
        "circuit_breaker": {
            "halt_threshold": 0.30,
            "reduce_heavy_threshold": 0.20,
            "reduce_light_threshold": 0.10,
        },
    }
    rm = RiskManager(config)
    rm._portfolio_hwm = 1_000_000

    result = rm.size_new_position(
        pair="BTC/USD",
        current_price=70000.0,
        current_atr=700.0,
        free_balance_usd=500_000.0,
        open_positions={},
        regime_multiplier=0.05,
        confidence=0.65,
        portfolio_weight=0.5,
    )
    assert result.decision == RiskDecision.BLOCKED_ZERO_REGIME_MULTIPLIER

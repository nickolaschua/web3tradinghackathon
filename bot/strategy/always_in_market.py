"""
Always-In-Market Strategy

Maintains constant market exposure, switching between long and neutral based on trend.
Designed to maximize daily coverage (100% theoretically).

Logic:
- Enter LONG when: EMA_20 > EMA_50 (uptrend) and not already in
- Exit when: EMA_20 < EMA_50 (downtrend)
- Re-enter quickly when trend resumes

This ensures we're trading almost every day (whenever trend changes).
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class AlwaysInMarketStrategy(BaseStrategy):
    """
    Always-in-market trend-following strategy.

    Stays in the market whenever in uptrend, maximizing daily coverage.
    """

    def __init__(
        self,
        fast_ema: int = 20,
        slow_ema: int = 50,
        min_bars_between_trades: int = 3,
    ):
        """
        Initialize strategy.

        Args:
            fast_ema: Fast EMA period (default 20)
            slow_ema: Slow EMA period (default 50)
            min_bars_between_trades: Min bars to wait before re-entry (default 3 = 45min)
        """
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.min_bars_between_trades = min_bars_between_trades
        self._bars_since_last_trade = 999

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """Generate always-in-market signal."""
        if len(features) < self.slow_ema + 5:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]
        ema_fast = latest.get("ema_20", 0.0)
        ema_slow = latest.get("ema_50", 0.0)
        close = latest.get("close", 0.0)

        if pd.isna(ema_fast) or pd.isna(ema_slow) or ema_fast == 0 or ema_slow == 0:
            return TradingSignal(pair=pair)

        self._bars_since_last_trade += 1

        # Uptrend: Fast EMA above slow EMA
        in_uptrend = ema_fast > ema_slow

        # Trend strength
        trend_strength = abs(ema_fast - ema_slow) / ema_slow
        confidence = min(0.9, 0.5 + trend_strength * 10)

        # --- ENTRY: Uptrend and waited enough ---
        if in_uptrend and self._bars_since_last_trade >= self.min_bars_between_trades:
            self._bars_since_last_trade = 0
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.4,
                confidence=confidence,
            )

        # --- EXIT: Downtrend ---
        if not in_uptrend:
            self._bars_since_last_trade = 0
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.7,
            )

        return TradingSignal(pair=pair)

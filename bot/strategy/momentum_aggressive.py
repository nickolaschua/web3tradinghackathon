"""
Aggressive Momentum Strategy

Trades on simple momentum thresholds with minimal confirmation.
Designed for very high frequency to maximize daily coverage.

Entry Logic:
- Price momentum > +0.3% over last 4 bars (1 hour on 15m)
- RSI > 45 (not oversold)

Exit Logic:
- Momentum turns negative
- Or RSI < 40 (momentum fading)

Very aggressive - trades frequently.
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class MomentumAggressiveStrategy(BaseStrategy):
    """
    Aggressive momentum strategy for high-frequency trading.

    Simple momentum threshold with minimal filters.
    """

    def __init__(
        self,
        momentum_threshold: float = 0.003,
        momentum_bars: int = 4,
        rsi_min: float = 45.0,
        rsi_exit: float = 40.0,
    ):
        """
        Initialize with aggressive parameters.

        Args:
            momentum_threshold: Min momentum for entry (default 0.003 = 0.3%)
            momentum_bars: Lookback for momentum (default 4 bars = 1h on 15m)
            rsi_min: Minimum RSI for entry (default 45)
            rsi_exit: RSI threshold for exit (default 40)
        """
        self.momentum_threshold = momentum_threshold
        self.momentum_bars = momentum_bars
        self.rsi_min = rsi_min
        self.rsi_exit = rsi_exit

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """Generate momentum signal."""
        if len(features) < self.momentum_bars + 5:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]
        close = latest.get("close", 0.0)
        rsi = latest.get("rsi", 50.0)

        if pd.isna(close) or pd.isna(rsi) or close == 0:
            return TradingSignal(pair=pair)

        # Calculate momentum
        close_prev = features["close"].iloc[-self.momentum_bars - 1]
        momentum = (close - close_prev) / close_prev if close_prev > 0 else 0

        # --- ENTRY: Positive momentum + not oversold ---
        if momentum > self.momentum_threshold and rsi > self.rsi_min:
            confidence = min(0.9, 0.5 + momentum * 20)
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.35,
                confidence=confidence,
            )

        # --- EXIT: Momentum turns negative or RSI drops ---
        if momentum < 0 or rsi < self.rsi_exit:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.6,
            )

        return TradingSignal(pair=pair)

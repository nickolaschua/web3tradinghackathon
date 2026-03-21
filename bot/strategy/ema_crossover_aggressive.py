"""
Aggressive EMA Crossover Strategy

High-frequency trend-following strategy using fast EMA crossovers.
Designed for maximum trade frequency to achieve high daily coverage.

Entry Logic:
- Fast EMA (5) crosses above slow EMA (10) = BUY
- Volume > 80% of average (light confirmation)

Exit Logic:
- Fast EMA crosses below slow EMA = SELL
- Or RSI > 70 (overbought exit)

Aggressive parameters for high frequency.
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class EMACrossoverAggressiveStrategy(BaseStrategy):
    """
    Aggressive EMA crossover for high-frequency trading.

    Uses 5/10 EMA crossover with minimal confirmation for maximum signals.
    """

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 10,
        volume_threshold: float = 0.8,
        rsi_overbought: float = 70.0,
    ):
        """
        Initialize with aggressive parameters.

        Args:
            fast_period: Fast EMA period (default 5)
            slow_period: Slow EMA period (default 10)
            volume_threshold: Volume multiplier threshold (default 0.8 = light)
            rsi_overbought: RSI exit threshold (default 70)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_threshold = volume_threshold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """Generate EMA crossover signal."""
        if len(features) < self.slow_period + 5:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]
        prev = features.iloc[-2]

        # Get EMAs (compute if not in features)
        close = features["close"]
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        fast_now = ema_fast.iloc[-1]
        fast_prev = ema_fast.iloc[-2]
        slow_now = ema_slow.iloc[-1]
        slow_prev = ema_slow.iloc[-2]

        volume = latest.get("volume", 0.0)
        volume_ma = latest.get("volume_ma_20", 1.0)
        rsi = latest.get("rsi", 50.0)

        if pd.isna(fast_now) or pd.isna(slow_now):
            return TradingSignal(pair=pair)

        # Volume confirmation (very light)
        volume_ok = volume > (volume_ma * self.volume_threshold) if volume_ma > 0 else True

        # Crossover detection
        bullish_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
        bearish_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

        # --- ENTRY: Bullish crossover ---
        if bullish_cross and volume_ok:
            confidence = min(0.8, 0.5 + (fast_now - slow_now) / slow_now * 10)
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.3,
                confidence=confidence,
            )

        # --- EXIT: Bearish crossover or overbought ---
        if bearish_cross or rsi > self.rsi_overbought:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.6,
            )

        return TradingSignal(pair=pair)

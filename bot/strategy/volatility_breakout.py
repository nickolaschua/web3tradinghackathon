"""
Volatility Breakout Strategy

Trades volatility expansion after periods of compression. Based on Bollinger Band
width (volatility proxy) and volume confirmation.

Entry Logic:
- BB width is expanding after being compressed (bb_width increasing)
- Price breaks above EMA_20 (momentum confirmation)
- Volume spike confirms breakout (volume > 1.5x average)

Exit Logic:
- BB width contracts significantly (volatility returning to normal)
- Price falls below EMA_20 (momentum lost)

No hardcoded stop-loss or sizing - uses RiskManager + PortfolioAllocator framework.
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility breakout strategy.

    Captures momentum moves that occur after volatility compression,
    confirmed by volume and price action.
    """

    def __init__(
        self,
        bb_width_threshold: float = 0.02,
        volume_multiplier: float = 1.5,
        min_compression_bars: int = 10,
    ):
        """
        Initialize strategy with tunable parameters.

        Args:
            bb_width_threshold: BB width below which is considered "compressed" (default 0.02)
            volume_multiplier: Volume must be this multiple of MA to confirm (default 1.5)
            min_compression_bars: Min bars of compression before expansion signal (default 10)
        """
        self.bb_width_threshold = bb_width_threshold
        self.volume_multiplier = volume_multiplier
        self.min_compression_bars = min_compression_bars

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate volatility breakout signal.

        Required features:
        - close: Current close price
        - ema_20: 20-period EMA
        - bb_width: Bollinger Band width (normalized)
        - volume: Current volume
        - volume_ma_20: 20-period volume MA

        Args:
            pair: Trading pair (e.g., "BTC/USD")
            features: DataFrame with required volatility features

        Returns:
            TradingSignal with BUY/SELL/HOLD direction
        """
        if len(features) < self.min_compression_bars + 5:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]
        close = latest.get("close", 0.0)
        ema_20 = latest.get("ema_20", close)
        bb_width = latest.get("bb_width", 0.1)
        volume = latest.get("volume", 0.0)
        volume_ma = latest.get("volume_ma_20", 1.0)

        # Handle missing data
        if pd.isna(close) or pd.isna(ema_20) or pd.isna(bb_width):
            return TradingSignal(pair=pair)

        # Check recent BB width history
        recent_bb_width = features["bb_width"].iloc[-self.min_compression_bars:].dropna()
        if len(recent_bb_width) < self.min_compression_bars:
            return TradingSignal(pair=pair)

        # Was volatility compressed recently?
        was_compressed = (recent_bb_width < self.bb_width_threshold).any()

        # Is volatility expanding now?
        bb_width_prev = features["bb_width"].iloc[-2] if len(features) > 1 else bb_width
        is_expanding = bb_width > bb_width_prev * 1.1 and bb_width > self.bb_width_threshold

        # Volume confirmation
        volume_spike = volume > (volume_ma * self.volume_multiplier) if volume_ma > 0 else False

        # Price above EMA (momentum)
        price_above_ema = close > ema_20

        # --- ENTRY CONDITIONS ---
        # BUY when: compression → expansion + price breaks above EMA + volume confirms
        if was_compressed and is_expanding and price_above_ema and volume_spike:
            # Confidence based on strength of breakout
            vol_ratio = volume / volume_ma if volume_ma > 0 else 1.0
            confidence = min(0.9, 0.5 + (vol_ratio - self.volume_multiplier) * 0.1)

            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.35,
                confidence=confidence,
            )

        # --- EXIT CONDITIONS ---
        # SELL when volatility contracts or price loses momentum
        is_contracting = bb_width < bb_width_prev * 0.9
        price_below_ema = close < ema_20

        if is_contracting or price_below_ema:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.6,
            )

        # Default: HOLD
        return TradingSignal(pair=pair)

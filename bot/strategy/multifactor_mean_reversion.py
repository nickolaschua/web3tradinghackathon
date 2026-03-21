"""
Multi-Factor Mean Reversion Strategy

Combines three complementary mean-reversion indicators into a composite score:
1. Bollinger Band position - price relative to volatility bands
2. RSI extremes with volume confirmation - momentum exhaustion
3. Price-to-MA z-score - statistical deviation from trend

Entry when multiple factors align (composite score >= 0.7), suggesting
strong oversold conditions likely to revert to mean.

Exit when mean reversion completes (composite < 0.3 or RSI crosses 50).

No hardcoded stop-loss or sizing - uses RiskManager + PortfolioAllocator framework.
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class MultifactorMeanReversionStrategy(BaseStrategy):
    """
    Multi-factor mean reversion strategy.

    Combines Bollinger Bands, RSI with volume, and price-to-MA z-score
    into a weighted composite score for robust mean-reversion detection.
    """

    def __init__(
        self,
        entry_threshold: float = 0.7,
        exit_threshold: float = 0.3,
        rsi_exit_threshold: float = 50.0,
        bb_weight: float = 0.35,
        rsi_weight: float = 0.35,
        zscore_weight: float = 0.30,
    ):
        """
        Initialize strategy with tunable parameters.

        Args:
            entry_threshold: Composite score threshold for entry (default 0.7)
            exit_threshold: Composite score threshold for exit (default 0.3)
            rsi_exit_threshold: RSI level for mean reversion exit (default 50)
            bb_weight: Weight for Bollinger Band factor (default 0.35)
            rsi_weight: Weight for RSI factor (default 0.35)
            zscore_weight: Weight for z-score factor (default 0.30)
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.rsi_exit_threshold = rsi_exit_threshold

        # Ensure weights sum to 1.0
        total = bb_weight + rsi_weight + zscore_weight
        self.bb_weight = bb_weight / total
        self.rsi_weight = rsi_weight / total
        self.zscore_weight = zscore_weight / total

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate multi-factor mean reversion signal.

        Required features:
        - close: Current close price
        - bb_upper: Upper Bollinger Band (20-period, 2 std)
        - bb_lower: Lower Bollinger Band
        - rsi: RSI indicator (14-period)
        - volume: Current bar volume
        - volume_ma_20: 20-period volume moving average
        - ema_20: 20-period exponential moving average
        - returns_std_20: 20-period rolling standard deviation of returns

        Args:
            pair: Trading pair (e.g., "BTC/USD")
            features: DataFrame with required mean reversion features

        Returns:
            TradingSignal with BUY/SELL/HOLD direction, size request, and confidence
        """
        if len(features) == 0:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]

        # Extract required features
        close = latest.get("close", 0.0)
        bb_upper = latest.get("bb_upper", close)
        bb_lower = latest.get("bb_lower", close)
        rsi = latest.get("rsi", 50.0)
        volume = latest.get("volume", 0.0)
        volume_ma = latest.get("volume_ma_20", 1.0)
        ema_20 = latest.get("ema_20", close)
        returns_std = latest.get("returns_std_20", 0.01)

        # Handle missing or invalid data
        if pd.isna(close) or close <= 0:
            return TradingSignal(pair=pair)

        # --- FACTOR 1: Bollinger Band Position ---
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_position = (close - bb_lower) / bb_range
            # Score higher when price is near/below lower band
            if bb_position < 0.1:
                bb_score = 1.0
            elif bb_position < 0.3:
                bb_score = 0.5
            else:
                bb_score = 0.0
        else:
            bb_score = 0.0

        # --- FACTOR 2: RSI with Volume Confirmation ---
        volume_spike = volume > (volume_ma * 1.5) if volume_ma > 0 else False

        if rsi < 25 and volume_spike:
            rsi_score = 1.0  # Extreme oversold with volume = strong signal
        elif rsi < 30:
            rsi_score = 0.5  # Oversold but no volume confirmation
        else:
            rsi_score = 0.0

        # --- FACTOR 3: Price-to-MA Z-Score ---
        if ema_20 > 0 and returns_std > 0:
            price_deviation = (close - ema_20) / ema_20
            z_score = price_deviation / returns_std

            if z_score < -2.0:
                ma_score = 1.0  # Strong statistical deviation
            elif z_score < -1.5:
                ma_score = 0.5  # Moderate deviation
            else:
                ma_score = 0.0
        else:
            ma_score = 0.0

        # --- COMPOSITE MEAN-REVERSION SCORE ---
        mr_composite = (
            self.bb_weight * bb_score +
            self.rsi_weight * rsi_score +
            self.zscore_weight * ma_score
        )

        # --- ENTRY CONDITIONS ---
        # BUY when composite score indicates strong mean-reversion opportunity
        if mr_composite >= self.entry_threshold:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.4,  # Request 40% of portfolio
                confidence=mr_composite,  # Use composite score as confidence
            )

        # --- EXIT CONDITIONS ---
        # SELL when mean reversion completes (back to neutral) or RSI crosses above 50
        if mr_composite < self.exit_threshold or rsi > self.rsi_exit_threshold:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,  # Exit full position
                confidence=0.5,
            )

        # Default: HOLD current position
        return TradingSignal(pair=pair)

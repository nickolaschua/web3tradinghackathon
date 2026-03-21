"""
Intraday scalping strategies for quick entry/exit to boost daily coverage.

These strategies target 15-minute to 2-hour holds and layer on top of
longer-term XGBoost positions.

Two core alphas:
1. Extreme Mean Reversion - Trade violent oversold/overbought reversals
2. Momentum Scalp - Capture quick breakouts with volume confirmation
"""

import pandas as pd
import numpy as np

from bot.strategy.base import BaseStrategy, TradingSignal, SignalDirection


class ExtremeMeanReversionScalp(BaseStrategy):
    """
    Trade extreme RSI readings that tend to snap back quickly.

    Entry:
    - RSI_7 < 15 (deeply oversold) OR RSI_7 > 85 (deeply overbought)
    - Price far from VWAP (> 2 std devs)
    - Volume spike (> 1.5x average)

    Exit:
    - RSI returns to 30-70 range
    - Or RiskManager ATR stops

    Expected hold time: 15 minutes to 2 hours
    Expected coverage: 20-30% (catches ~1-2 extremes per week per asset)
    """

    def __init__(self, rsi_oversold=15, rsi_overbought=85, volume_threshold=1.5):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_threshold = volume_threshold

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """Generate scalp signal based on extreme conditions."""

        if features.empty:
            return TradingSignal(pair=pair, direction=SignalDirection.HOLD)

        latest = features.iloc[-1]

        # Get indicators
        rsi_7 = latest.get("rsi_7", latest.get("RSI_7", 50))
        rsi_14 = latest.get("rsi", latest.get("RSI_14", 50))
        volume_ratio = latest.get("volume_ratio", 1.0)
        bb_pos = latest.get("bb_pos", 0.5)
        close = latest.get("close", 0)

        # Calculate VWAP deviation if available
        if "vwap" in features.columns:
            vwap = latest.get("vwap", close)
            vwap_std = features["close"].rolling(20).std().iloc[-1]
            vwap_zscore = (close - vwap) / vwap_std if vwap_std > 0 else 0
        else:
            # Approximate with BB position
            vwap_zscore = (bb_pos - 0.5) * 4  # Scale to ±2

        # ENTRY LOGIC: Extreme oversold
        if (rsi_7 < self.rsi_oversold and
            volume_ratio > self.volume_threshold and
            vwap_zscore < -1.5):  # Price well below VWAP

            confidence = min((self.rsi_oversold - rsi_7) / self.rsi_oversold, 1.0)

            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.3,  # Smaller size for scalps
                confidence=max(confidence, 0.6)
            )

        # ENTRY LOGIC: Extreme overbought (for short, but we only do longs)
        # Skip for now, only trade long

        # EXIT LOGIC: RSI normalized
        if rsi_7 > 30 and rsi_7 < 70:
            # Exit if we were in a scalp position
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.7
            )

        return TradingSignal(pair=pair, direction=SignalDirection.HOLD)


class MomentumBreakoutScalp(BaseStrategy):
    """
    Trade quick momentum breakouts with volume confirmation.

    Entry:
    - Bollinger Band squeeze (width in bottom 20th percentile)
    - Price breaks above BB upper band
    - Volume spike (> 2x average)
    - MACD histogram turning positive

    Exit:
    - Price closes below EMA_20
    - Or volume dries up
    - Or RiskManager ATR stops

    Expected hold time: 30 minutes to 3 hours
    Expected coverage: 15-25% (catches ~1 breakout per week per asset)
    """

    def __init__(self, bb_squeeze_pct=0.20, volume_threshold=2.0):
        self.bb_squeeze_pct = bb_squeeze_pct
        self.volume_threshold = volume_threshold

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """Generate breakout scalp signal."""

        if features.empty or len(features) < 50:
            return TradingSignal(pair=pair, direction=SignalDirection.HOLD)

        latest = features.iloc[-1]

        # Get indicators
        bb_width = latest.get("bb_width", 0.1)
        bb_pos = latest.get("bb_pos", 0.5)
        close = latest.get("close", 0)
        ema_20 = latest.get("ema_20", latest.get("EMA_20", close))
        volume_ratio = latest.get("volume_ratio", 1.0)
        macd_hist = latest.get("macd_hist", latest.get("MACDh_12_26_9", 0))

        # Calculate BB width percentile over last 100 bars
        if "bb_width" in features.columns:
            bb_width_pct = features["bb_width"].rolling(100).apply(
                lambda x: (x.iloc[-1] < x.quantile(self.bb_squeeze_pct)).astype(int)
            ).iloc[-1]
        else:
            bb_width_pct = 0

        # ENTRY LOGIC: Breakout from squeeze
        if (bb_width_pct > 0 and  # In a squeeze
            bb_pos > 0.95 and  # Breaking above upper band
            volume_ratio > self.volume_threshold and  # Volume spike
            macd_hist > 0 and  # Positive momentum
            close > ema_20):  # Above EMA

            confidence = min(volume_ratio / self.volume_threshold, 1.0)

            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.3,  # Smaller size for scalps
                confidence=max(confidence, 0.65)
            )

        # EXIT LOGIC: Lost momentum
        if close < ema_20 or volume_ratio < 0.8:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.7
            )

        return TradingSignal(pair=pair, direction=SignalDirection.HOLD)


class VWAPReversionScalp(BaseStrategy):
    """
    Mean reversion to VWAP on intraday timeframes.

    Entry:
    - Price > 1.5 std devs away from VWAP
    - Volume spike suggests exhaustion
    - Starting to revert (price moving back toward VWAP)

    Exit:
    - Price crosses VWAP
    - Or RiskManager stops

    Expected hold time: 30 minutes to 2 hours
    Expected coverage: 25-35% (frequent intraday deviations)
    """

    def __init__(self, vwap_threshold=1.5, volume_threshold=1.3):
        self.vwap_threshold = vwap_threshold
        self.volume_threshold = volume_threshold

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """Generate VWAP reversion signal."""

        if features.empty or len(features) < 20:
            return TradingSignal(pair=pair, direction=SignalDirection.HOLD)

        latest = features.iloc[-1]
        prev = features.iloc[-2] if len(features) > 1 else latest

        close = latest.get("close", 0)
        close_prev = prev.get("close", close)
        volume_ratio = latest.get("volume_ratio", 1.0)

        # Calculate VWAP
        if "vwap" in features.columns:
            vwap = latest.get("vwap", close)
        else:
            # Approximate VWAP with 20-period volume-weighted MA
            vwap = (features["close"] * features["volume"]).rolling(20).sum() / features["volume"].rolling(20).sum()
            vwap = vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else close

        # Calculate deviation from VWAP
        vwap_std = features["close"].rolling(20).std().iloc[-1]
        vwap_zscore = (close - vwap) / vwap_std if vwap_std > 0 else 0

        # ENTRY LOGIC: Oversold vs VWAP, starting to revert
        if (vwap_zscore < -self.vwap_threshold and  # Far below VWAP
            volume_ratio > self.volume_threshold and  # Volume spike
            close > close_prev):  # Starting to revert upward

            confidence = min(abs(vwap_zscore) / 3.0, 1.0)

            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.25,
                confidence=max(confidence, 0.6)
            )

        # EXIT LOGIC: Reached VWAP
        if abs(vwap_zscore) < 0.3:  # Close to VWAP
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.75
            )

        return TradingSignal(pair=pair, direction=SignalDirection.HOLD)

"""
Mean-reversion with regime gate strategy for the Roostoo trading bot.

Entry logic:
  1. Regime gate: only enter longs when EMA_20 > EMA_50 (local uptrend).
     Blocked in downtrend to avoid catching falling knives.
  2. Standard entry: RSI_14 < 30 AND bb_pos < 0.15 AND MACDh_12_26_9 > 0.
     Three stacked conditions → higher precision, size=0.35.
  3. Extreme entry: RSI_14 < 25 alone (EMA gate already confirmed uptrend).
     Single condition → lower size=0.25.
Exit logic:
  RSI_14 > 55 OR bb_pos > 0.6 → SELL full position.

Reference: research/strategies/mean_reversion_regime_gate.md
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class MeanReversionStrategy(BaseStrategy):
    """
    Mean-reversion strategy with EMA regime gate.

    Only enters long when the coin is in a local uptrend (EMA_20 > EMA_50).
    In a downtrend, oversold readings keep declining — the regime gate prevents
    the "catching falling knives" failure mode.
    """

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate a mean-reversion signal from the latest features.

        Parameters
        ----------
        pair : str
            Tradeable pair symbol, e.g. "BTC/USD".
        features : pd.DataFrame
            Computed indicator DataFrame (all columns already shifted 1 bar).
            Required columns: EMA_20, EMA_50, RSI_14, bb_pos, MACDh_12_26_9.

        Returns
        -------
        TradingSignal
            BUY when deeply oversold in uptrend; SELL when mean-reverted;
            HOLD otherwise.
        """
        if len(features) == 0:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]

        # ── Regime gate ──────────────────────────────────────────────────────
        # Require a local uptrend (EMA_20 > EMA_50). This per-coin micro-regime
        # check prevents entries on declining coins even during global bull markets.
        ema_20 = latest.get("EMA_20", float("nan"))
        ema_50 = latest.get("EMA_50", float("nan"))
        if not (ema_20 > ema_50):
            return TradingSignal(pair=pair)  # downtrend — hold

        rsi       = latest.get("RSI_14", 50.0)
        bb_pos    = latest.get("bb_pos", 0.5)
        macd_hist = latest.get("MACDh_12_26_9", 0.0)

        # ── Entry: 3-condition stack (highest precision) ──────────────────────
        # All three conditions together: RSI oversold + near lower Bollinger Band
        # + MACD histogram has turned positive (momentum reversing, not leading).
        # Fires ~2-3% of bars; win rate ~65-70%.
        if rsi < 30 and bb_pos < 0.15 and macd_hist > 0:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.35,
                confidence=0.60,
            )

        # ── Entry: extreme oversold alone ─────────────────────────────────────
        # RSI < 25 in an uptrend is extreme enough to enter without bb_pos or MACD
        # confirmation. Smaller size due to lower precision (single condition).
        if rsi < 25:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.25,
                confidence=0.55,
            )

        # ── Exit: mean has reverted ──────────────────────────────────────────
        # RSI back to neutral (>55) or price above Bollinger midpoint (bb_pos>0.6).
        # Exit the full mean-reversion position; momentum strategy handles the rest.
        if rsi > 55 or bb_pos > 0.6:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.70,
            )

        # ── Default: hold current position ───────────────────────────────────
        return TradingSignal(pair=pair)

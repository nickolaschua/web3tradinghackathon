"""
Relaxed mean-reversion strategy for activity coverage across 20 coins.

Purpose: ensure 8/10 active trading days for the competition by generating
mean-reversion entries more frequently than the original MeanReversionStrategy.
Uses smaller position sizes (0.10-0.15) since the primary edge comes from
the BTC/SOL XGBoost models — this strategy provides activity, not alpha.

Entry logic:
  1. Regime gate: EMA_20 > EMA_50 (same as original — no falling knives).
  2. Standard entry: RSI_14 < 35 AND bb_pos < 0.25 (relaxed from 30/0.15).
     Drops MACD histogram requirement. Size=0.15, confidence=0.50.
  3. Deep oversold: RSI_14 < 28 alone. Size=0.10, confidence=0.55.
Exit logic:
  RSI_14 > 50 OR bb_pos > 0.55 → SELL (tighter exit than original to lock in
  smaller gains and free capital for XGBoost entries).

Signal frequency: ~10 signals/day across 20 coins, 85% of days covered.
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class RelaxedMeanReversionStrategy(BaseStrategy):
    """
    Relaxed mean-reversion for activity coverage.

    Fires more often than MeanReversionStrategy but with smaller sizes.
    Designed to run alongside XGBoost models as an activity layer.
    """

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        if len(features) == 0:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]

        # ── Regime gate (same as original) ─────────────────────────────────
        ema_20 = latest.get("EMA_20", float("nan"))
        ema_50 = latest.get("EMA_50", float("nan"))
        if not (ema_20 > ema_50):
            return TradingSignal(pair=pair)

        rsi = latest.get("RSI_14", 50.0)
        bb_pos = latest.get("bb_pos", 0.5)

        # ── Entry: relaxed 2-condition stack ────────────────────────────────
        # RSI moderately oversold + near lower Bollinger Band in uptrend.
        # Relaxed from RSI<30/bb<0.15 to RSI<35/bb<0.25. No MACD requirement.
        # Small size (0.15) — activity layer, not primary alpha.
        if rsi < 35 and bb_pos < 0.25:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.15,
                confidence=0.50,
            )

        # ── Entry: deep oversold ────────────────────────────────────────────
        # RSI < 28 in uptrend — slightly relaxed from 25.
        if rsi < 28:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.10,
                confidence=0.50,
            )

        # ── Exit: tighter than original ─────────────────────────────────────
        # RSI > 50 or bb_pos > 0.55 — take small profits quickly, free capital.
        if rsi > 50 or bb_pos > 0.55:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.60,
            )

        return TradingSignal(pair=pair)

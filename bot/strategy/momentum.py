"""
Momentum strategy for the Roostoo trading bot.

Regime-filtered trend-following:
- Uptrend regime: EMA_20 > EMA_50
- Entry: RSI_14 < 50 AND MACD histogram positive (momentum rising in uptrend)
- Exit: RSI_14 > 65 OR MACD histogram turns negative (momentum exhausted)

Feature columns used (all shifted 1 bar — no look-ahead):
  RSI_14, MACDh_12_26_9, EMA_20, EMA_50
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class MomentumStrategy(BaseStrategy):
    """
    Regime-filtered momentum strategy.

    Only enters longs when EMA_20 > EMA_50 (uptrend). Buys on RSI dip within
    positive MACD momentum; exits when overbought or momentum reverses.
    """

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate a momentum signal from the latest feature row.

        Parameters
        ----------
        pair : str
            Tradeable pair, e.g. "BTC/USD".
        features : pd.DataFrame
            Full feature history up to and including the current bar.
            Columns are shifted 1 bar — no look-ahead bias.
            Uses only features.iloc[-1].

        Returns
        -------
        TradingSignal with direction BUY, SELL, or HOLD.
        """
        if len(features) == 0:
            return TradingSignal(pair=pair)

        latest = features.iloc[-1]

        # Guard: require all needed columns
        required = ["RSI_14", "MACDh_12_26_9", "EMA_20", "EMA_50"]
        for col in required:
            if col not in latest.index or pd.isna(latest[col]):
                return TradingSignal(pair=pair)

        rsi = latest["RSI_14"]
        macd_hist = latest["MACDh_12_26_9"]
        ema_fast = latest["EMA_20"]
        ema_slow = latest["EMA_50"]

        # Regime filter: only trade in an established uptrend
        in_uptrend = ema_fast > ema_slow

        # --- Entry: RSI dip + positive MACD momentum + uptrend ---
        # RSI < 50: not yet overbought — catching the early move
        # MACD hist > 0: momentum is positive / recently crossed up
        if in_uptrend and rsi < 50 and macd_hist > 0:
            # Scale confidence by how far RSI is from overbought (lower RSI = higher confidence)
            confidence = min(0.9, max(0.5, (50 - rsi) / 50 + 0.5))
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=1.0,
                confidence=round(confidence, 2),
            )

        # --- Exit: overbought OR momentum reversal ---
        # RSI > 65: extended / overbought
        # MACD hist < 0: momentum has turned negative
        if rsi > 65 or macd_hist < 0:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                size=1.0,
                confidence=0.7,
            )

        # Default: hold current position
        return TradingSignal(pair=pair)

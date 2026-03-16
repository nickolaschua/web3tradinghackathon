"""
Mean-reversion strategy stub for the Roostoo trading bot.

This module contains MeanReversionStrategy — a placeholder that returns a safe
HOLD signal until you fill in your alpha logic.  Open this file, read the
docstring on generate_signal(), and write your conditions in the marked zones.
You do not need to read any other file to understand what data is available.
"""
from __future__ import annotations

import pandas as pd
from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class MeanReversionStrategy(BaseStrategy):
    """
    Mean-reversion strategy stub.

    Returns a HOLD signal by default.  Safe to deploy before alpha logic is
    filled in — no real positions will be opened until the entry zones below
    contain actual conditions.
    """

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate a mean-reversion trading signal from the latest features.

        Parameters
        ----------
        pair : str
            The tradeable pair symbol, e.g. ``"BTC/USD"``.  The returned
            TradingSignal will carry this exact value in its ``pair`` field.
        features : pd.DataFrame
            Recent candle/indicator data.  Every indicator column is already
            shifted by 1 bar (``shift(1)``) — there is no look-ahead bias.
            The most recent complete bar is the last row (``features.iloc[-1]``).

        Available feature columns
        -------------------------
        close         : float  — latest close price
        atr_proxy     : float  — volatility proxy: log_returns.rolling(14).std()
                                 * close * 1.25
        rsi           : float  — RSI(14), value 0–100
        macd          : float  — MACD line (12, 26)
        macd_signal   : float  — MACD signal line (9)
        macd_hist     : float  — MACD histogram (macd − macd_signal)
        ema_slope     : float  — rate of change of EMA(20) over last N bars
        eth_btc_corr  : float  — rolling 20-bar correlation of BTC/USD and
                                 ETH/USD returns
        sol_btc_corr  : float  — rolling 20-bar correlation of BTC/USD and
                                 SOL/USD returns
        eth_return    : float  — latest ETH/USD log return
        sol_return    : float  — latest SOL/USD log return

        Returns
        -------
        TradingSignal
            ``pair`` always matches the input ``pair`` argument.
            ``direction`` is BUY, SELL, or HOLD.
            ``size`` is a fraction of the portfolio (0.0–1.0).
            ``confidence`` is 0.0–1.0 (reserved for future sizing/filtering).

        Example
        -------
        Simple oversold bounce (mean-reversion buy):

            if latest["rsi"] < 25 and latest["macd_hist"] > 0:
                return TradingSignal(
                    pair=pair,
                    direction=SignalDirection.BUY,
                    size=0.4,
                    confidence=0.65,
                )
        """
        # Get the most recent row of features
        latest = features.iloc[-1] if len(features) > 0 else None
        if latest is None:
            return TradingSignal(pair=pair)

        # --- ADD YOUR ENTRY CONDITIONS HERE ---
        # Mean-reversion example: buy when RSI is deeply oversold and MACD hist is turning up
        # if latest["rsi"] < 25 and latest["macd_hist"] > 0:
        #     return TradingSignal(pair=pair, direction=SignalDirection.BUY, size=0.4, confidence=0.65)

        # --- ADD YOUR EXIT CONDITIONS HERE ---
        # Example: exit when RSI returns to mean (>50)
        # if latest["rsi"] > 50:
        #     return TradingSignal(pair=pair, direction=SignalDirection.SELL, size=1.0, confidence=0.5)

        # Default: no signal — hold current position
        return TradingSignal(pair=pair)

"""
Intraday trigger strategy with 4H directional bias scaling.

Design:
- Fast trigger (5m/15m) creates frequent opportunities.
- Slow 4H bias (momentum + funding + macro) scales size only.
- Bias never blocks base trades.
"""
from __future__ import annotations

import pandas as pd

from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal


class IntradayMomentumStrategy(BaseStrategy):
    """
    Configurable fast trigger + slow bias size scaling.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.interval = cfg.get("interval", "15m")
        self.trigger_mode = cfg.get("trigger_mode", "momentum_breakout")
        self.return_threshold = float(cfg.get("return_threshold", 0.0015))
        self.volume_ratio_threshold = float(cfg.get("volume_ratio_threshold", 1.2))
        self.zscore_threshold = float(cfg.get("zscore_threshold", 1.5))
        self.base_size = float(cfg.get("base_size", 1.0))
        self.bias_weight = float(cfg.get("bias_weight", 0.3))
        self.require_trend_confirmation = bool(cfg.get("require_trend_confirmation", False))
        self.require_volume_confirmation = bool(cfg.get("require_volume_confirmation", False))

    def _required_return_col(self) -> str:
        return "return_5m" if self.interval == "5m" else "return_15m"

    def _trigger(self, row: pd.Series) -> bool:
        ret_col = self._required_return_col()
        if self.trigger_mode == "momentum_breakout":
            trig = row[ret_col] > self.return_threshold
            if self.require_volume_confirmation:
                trig = trig and (row["volume_ratio"] > self.volume_ratio_threshold)
            return trig
        if self.trigger_mode == "mean_reversion":
            z_col = "zscore_5m" if self.interval == "5m" else "zscore_15m"
            # Long-only mean reversion: buy pullbacks against an uptrend.
            trig = row[z_col] < -self.zscore_threshold
            if self.require_trend_confirmation:
                trig = trig and (row["EMA_5"] > row["EMA_20"])
            if self.require_volume_confirmation:
                trig = trig and (row["volume_ratio"] > self.volume_ratio_threshold)
            return trig
        raise ValueError(f"Unsupported trigger_mode: {self.trigger_mode}")

    @staticmethod
    def _bias(row: pd.Series) -> int:
        b = 0
        if row["EMA_20_4h"] > row["EMA_50_4h"]:
            b += 1
        if row["btc_funding_zscore_4h"] < -1:
            b += 1
        if (row["oil_return_1d_4h"] > 0) and (row["dxy_return_1d_4h"] < 0):
            b += 1
        return b

    def _has_required(self, row: pd.Series) -> bool:
        cols = [
            self._required_return_col(),
            "volume_ratio",
            "EMA_5",
            "EMA_20",
            "EMA_20_4h",
            "EMA_50_4h",
            "btc_funding_zscore_4h",
            "oil_return_1d_4h",
            "dxy_return_1d_4h",
        ]
        if self.trigger_mode == "mean_reversion":
            cols.append("zscore_5m" if self.interval == "5m" else "zscore_15m")
        return all((c in row.index) and (not pd.isna(row[c])) for c in cols)

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        if len(features) == 0:
            return TradingSignal(pair=pair)
        row = features.iloc[-1]
        if not self._has_required(row):
            return TradingSignal(pair=pair)

        trig = self._trigger(row)
        if trig:
            bias = self._bias(row)
            size = min(1.0, max(0.0, self.base_size * (1.0 + self.bias_weight * bias)))
            confidence = min(1.0, 0.55 + 0.15 * bias)
            return TradingSignal(pair=pair, direction=SignalDirection.BUY, size=size, confidence=confidence)

        # Lightweight momentum fade exit for intraday re-entry cadence
        ret_col = self._required_return_col()
        if row[ret_col] < -self.return_threshold:
            return TradingSignal(pair=pair, direction=SignalDirection.SELL, size=1.0, confidence=0.6)

        return TradingSignal(pair=pair)

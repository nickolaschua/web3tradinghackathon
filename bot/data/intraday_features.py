"""
Intraday feature helpers for fast trigger strategies.

These functions are pure and backtest/live safe:
- no I/O
- no side effects
- all derived columns are shifted 1 bar to avoid look-ahead
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_intraday_features(df: pd.DataFrame, interval: str = "15m") -> pd.DataFrame:
    """
    Build short-horizon trigger features on intraday OHLCV.

    Required columns: open, high, low, close, volume

    Adds:
    - return_5m or return_15m
    - volume_ratio
    - vol_5m or vol_15m (20-bar rolling std of returns)
    - zscore_5m or zscore_15m (20-bar rolling z-score of close)
    - EMA_5, EMA_10, EMA_20
    - range_expansion (high-low normalized by rolling ATR proxy)

    All added columns are shifted by 1 bar.
    """
    if interval not in {"5m", "15m"}:
        raise ValueError(f"Unsupported interval: {interval}. Use '5m' or '15m'.")

    out = df.copy()
    out.columns = out.columns.str.lower()

    req = {"open", "high", "low", "close", "volume"}
    missing = req - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns for intraday features: {sorted(missing)}")

    ret_col = "return_5m" if interval == "5m" else "return_15m"
    vol_col = "vol_5m" if interval == "5m" else "vol_15m"
    z_col = "zscore_5m" if interval == "5m" else "zscore_15m"

    out[ret_col] = out["close"].pct_change()
    # Keep ATR proxy consistent with the core feature pipeline.
    log_ret = np.log(out["close"] / out["close"].shift(1))
    out["atr_proxy"] = log_ret.rolling(14).std() * out["close"] * 1.25
    out["volume_ratio"] = out["volume"] / (out["volume"].rolling(20).mean() + 1e-10)
    out[vol_col] = out[ret_col].rolling(20).std()

    mean20 = out["close"].rolling(20).mean()
    std20 = out["close"].rolling(20).std()
    out[z_col] = (out["close"] - mean20) / (std20 + 1e-10)

    out["EMA_5"] = out["close"].ewm(span=5, adjust=False).mean()
    out["EMA_10"] = out["close"].ewm(span=10, adjust=False).mean()
    out["EMA_20"] = out["close"].ewm(span=20, adjust=False).mean()

    true_range = np.maximum.reduce(
        [
            (out["high"] - out["low"]).values,
            (out["high"] - out["close"].shift(1)).abs().values,
            (out["low"] - out["close"].shift(1)).abs().values,
        ]
    )
    tr = pd.Series(true_range, index=out.index)
    atr20 = tr.rolling(20).mean()
    out["range_expansion"] = (out["high"] - out["low"]) / (atr20 + 1e-10)

    added = ["atr_proxy", ret_col, "volume_ratio", vol_col, z_col, "EMA_5", "EMA_10", "EMA_20", "range_expansion"]
    out[added] = out[added].shift(1)
    return out


def merge_slow_bias_to_intraday(intraday_df: pd.DataFrame, slow_bias_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 4H bias features onto intraday bars via forward-fill in UTC time.
    """
    out = intraday_df.copy()
    slow = slow_bias_df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    slow.index = pd.to_datetime(slow.index, utc=True)
    merged = out.join(slow, how="left")
    merged[slow.columns] = merged[slow.columns].ffill()
    return merged


__all__ = ["compute_intraday_features", "merge_slow_bias_to_intraday"]

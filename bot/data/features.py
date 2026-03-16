"""
Feature computation for the Roostoo trading bot.

All functions are pure (no side effects, no I/O) and safe to call
in both backtest and live environments. The output is identical in
both contexts — this is the calibration consistency guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # NOT pandas_ta — that package is broken on Python 3.11


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on an OHLCV DataFrame and return a new
    DataFrame with indicator columns shifted 1 bar to prevent look-ahead bias.

    Indicators included:
    - atr_proxy: Close-to-close volatility proxy (log_ret.rolling(14).std() * close * 1.25)
    - RSI_14:    Relative Strength Index (14-period)
    - MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9: MACD line, signal, histogram
    - EMA_20, EMA_50: Exponential Moving Averages
    - ema_slope: (EMA_20[t] - EMA_20[t-1]) / EMA_20[t-1] — rate of change of EMA_20

    Indicators DISABLED (≈0 on Roostoo synthetic candles where H=L=O=C, volume=0):
    - ta.atr()  — H-L range required; H=L → ATR≈0 → stop-loss system breaks
    - ta.adx()  — directional movement required; H=L → ADX≈0
    - ta.obv()  — volume required; volume=0 → OBV is static

    Shift behaviour:
    The indicator columns are shifted 1 bar AFTER all indicators are computed.
    This means bar N's feature row contains bar N-1's indicator values.
    The raw OHLCV columns (open, high, low, close, volume) are NOT shifted.

    Args:
        df: OHLCV DataFrame. Must have columns: open, high, low, close, volume.
            DatetimeIndex recommended but not required.
            Minimum required rows: 50 (MACD needs 35, atr_proxy needs 15).

    Returns:
        New DataFrame with original OHLCV columns plus indicator columns shifted 1 bar.
        Rows with all-NaN indicators (warmup period) are NOT dropped here —
        caller is responsible for dropna() after cross-asset injection.
    """
    ohlcv_cols = {"open", "high", "low", "close", "volume"}

    out = df.copy()

    # --- ATR proxy ---
    # Standard ta.atr() returns ≈0 when High == Low (synthetic candles from Roostoo).
    # Close-to-close proxy: SD of log returns × current price × 1.25.
    # Factor derivation: ATR ≈ SD / 0.875 (empirical); 1.25 > 1/0.875 = 1.143 for
    # conservative crypto fat-tail buffer. Same formula must be used in backtest
    # and live — do not switch to ta.atr() in backtest.
    log_ret = np.log(out["close"] / out["close"].shift(1))
    out["atr_proxy"] = log_ret.rolling(14).std() * out["close"] * 1.25

    # --- pandas-ta-classic indicators ---
    # All use .ta accessor with append=True — mutates `out` in-place, returns None.
    # Column names are in UPPER_UNDERSCORE format as documented by pandas-ta-classic.
    out.ta.rsi(length=14, append=True)                         # → RSI_14
    out.ta.macd(fast=12, slow=26, signal=9, append=True)       # → MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    out.ta.ema(length=20, append=True)                         # → EMA_20
    out.ta.ema(length=50, append=True)                         # → EMA_50

    # --- EMA slope (manual — df.ta.slope(close="EMA_20") kwarg form is unverified) ---
    # Rate of change of EMA_20: (EMA_20[t] - EMA_20[t-1]) / EMA_20[t-1]
    out["ema_slope"] = (out["EMA_20"] - out["EMA_20"].shift(1)) / out["EMA_20"].shift(1)

    # --- Shift all indicator columns 1 bar to prevent look-ahead bias ---
    # CRITICAL: shift AFTER computing indicators, never before.
    # Shifting close prices first would make indicators compute on bar N-1 close
    # but label it as bar N — a subtle off-by-one that causes garbage signal values.
    ind_cols = [c for c in out.columns if c not in ohlcv_cols]
    out[ind_cols] = out[ind_cols].shift(1)

    return out

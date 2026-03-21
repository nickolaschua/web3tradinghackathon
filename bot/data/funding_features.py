"""
Funding rate sentiment feature computation.

Computes features from Binance perpetual futures funding rates and aligns
them to the 4H OHLCV bar index for use in the XGBoost classifier.

Features:
  btc_funding_latest       — latest settled funding rate (fwd-filled, shifted 1)
  btc_funding_ma_24h       — rolling 3-settlement (24h) mean
  btc_funding_change_24h   — latest minus 3-settlements-ago (sentiment momentum)
  btc_funding_self_zscore  — z-score vs own rolling 270-settlement (90d) history
  btc_funding_extreme      — binary: |self_zscore| > 2

All features are 1-bar shifted to prevent look-ahead bias (same convention as
compute_features() in features.py).

Usage (backtest / IC test):
    from bot.data.funding_features import compute_btc_funding_features
    feat_df = compute_btc_funding_features(funding_df, btc_ohlcv.index)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_btc_funding_features(
    funding_df: pd.DataFrame,
    ohlcv_index: pd.DatetimeIndex,
    zscore_window: int = 270,  # 90 days × 3 settlements/day
) -> pd.DataFrame:
    """
    Compute BTC funding rate sentiment features aligned to the 4H OHLCV index.

    Args:
        funding_df:    DataFrame with columns fundingTime (datetime UTC) and
                       fundingRate (float). From funding_fetcher.load_or_fetch_funding().
        ohlcv_index:   DatetimeIndex of the 4H OHLCV DataFrame (UTC).
        zscore_window: Rolling window (in settlements) for the self-z-score.
                       270 = 90 days × 3 settlements/day.

    Returns:
        DataFrame indexed by ohlcv_index with columns:
          btc_funding_latest, btc_funding_ma_24h, btc_funding_change_24h,
          btc_funding_self_zscore, btc_funding_extreme
        All columns are already 1-bar shifted.
    """
    _COLS = [
        "btc_funding_latest",
        "btc_funding_ma_24h",
        "btc_funding_change_24h",
        "btc_funding_self_zscore",
        "btc_funding_extreme",
    ]

    if funding_df.empty:
        # Neutral imputation — missing funding data is not the same as neutral,
        # but 0.0 is the least harmful value for a model that handles NaN via dropna.
        out = pd.DataFrame(0.0, index=ohlcv_index, columns=_COLS)
        out["btc_funding_extreme"] = out["btc_funding_extreme"].astype(int)
        return out

    # Build a Series indexed by settlement timestamp (8H cadence)
    funding = (
        funding_df.set_index("fundingTime")["fundingRate"]
        .sort_index()
        .astype(float)
    )

    # --- Compute features on the native 8H funding cadence ---

    # 3-settlement rolling mean ≈ 24h average
    ma_24h = funding.rolling(3, min_periods=1).mean()

    # Change vs 24h ago (3 settlements = 24h at 8H cadence)
    change_24h = funding - funding.shift(3)

    # Self z-score: how extreme is the current rate vs its own history?
    roll_mean = funding.rolling(zscore_window, min_periods=30).mean()
    roll_std  = funding.rolling(zscore_window, min_periods=30).std()
    self_zscore = (funding - roll_mean) / (roll_std + 1e-10)

    # Extreme flag: |z-score| > 2
    extreme = (self_zscore.abs() > 2).astype(float)

    # --- Align to 4H OHLCV index (forward-fill) then shift 1 bar ---
    def _ffill_and_shift(series: pd.Series) -> pd.Series:
        return series.reindex(ohlcv_index, method="ffill").shift(1)

    out = pd.DataFrame(index=ohlcv_index)
    out["btc_funding_latest"]      = _ffill_and_shift(funding)
    out["btc_funding_ma_24h"]      = _ffill_and_shift(ma_24h)
    out["btc_funding_change_24h"]  = _ffill_and_shift(change_24h)
    out["btc_funding_self_zscore"] = _ffill_and_shift(self_zscore)
    out["btc_funding_extreme"]     = _ffill_and_shift(extreme)

    return out


__all__ = ["compute_btc_funding_features"]

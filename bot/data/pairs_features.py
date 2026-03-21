"""
Shared feature computation for the BTC/ETH pairs ML strategy.
Called identically from scripts/train_pairs_model.py (training) and
bot/strategy/pairs_ml_strategy.py (live) — any change here affects both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ar1_halflife(series: pd.Series) -> float:
    """
    Compute AR(1) half-life: -log(2) / log(abs(rho))
    where rho is the lag-1 autocorrelation coefficient.

    Helper for rolling halflife computation.
    Returns: half-life clipped to [1, 200].
    """
    if len(series) < 2:
        return np.nan

    # Remove NaN
    s = series.dropna()
    if len(s) < 2:
        return np.nan

    # Lag-1 autocorrelation
    rho = s.autocorr(lag=1)

    if np.isnan(rho) or rho == 0 or abs(rho) >= 1.0:
        return np.nan

    halflife = -np.log(2) / np.log(abs(rho))
    return np.clip(halflife, 1, 200)


def compute_pairs_features(
    btc_closes: pd.Series,
    eth_closes: pd.Series,
    ols_window: int = 2880,
    zscore_window: int = 672,
) -> pd.DataFrame:
    """
    Compute pairs features for BTC/ETH mean reversion strategy.

    All features are shifted by 1 bar at the very end to prevent look-ahead bias.

    Args:
        btc_closes: DatetimeIndex, float prices for BTC
        eth_closes: DatetimeIndex, float prices for ETH (same index alignment assumed)
        ols_window: rolling OLS window (default 2880 = 30 days at 15M)
        zscore_window: z-score normalisation window (default 672 = 1 week at 15M)

    Returns:
        DataFrame with all feature columns (shifted 1 bar to prevent look-ahead).
        Columns: rolling_beta, rolling_alpha, spread, spread_mean, spread_std,
                 zscore, abs_zscore, zscore_lag4, zscore_lag16,
                 zscore_momentum_4, zscore_momentum_16,
                 spread_std_short, spread_std_long,
                 btc_return_4, eth_return_4, btc_return_16, eth_return_16,
                 btc_eth_corr_60, btc_eth_corr_240,
                 spread_halflife
    """
    # Log prices
    log_btc = np.log(btc_closes)
    log_eth = np.log(eth_closes)

    # Initialize output DataFrame
    out = pd.DataFrame(index=btc_closes.index)

    # --- Rolling OLS: beta and alpha ---
    # Using efficient rolling cov/var method
    rolling_cov = log_btc.rolling(ols_window).cov(log_eth)
    rolling_var = log_eth.rolling(ols_window).var()
    rolling_beta = rolling_cov / rolling_var

    rolling_btc_mean = log_btc.rolling(ols_window).mean()
    rolling_eth_mean = log_eth.rolling(ols_window).mean()
    rolling_alpha = rolling_btc_mean - rolling_beta * rolling_eth_mean

    out["rolling_beta"] = rolling_beta
    out["rolling_alpha"] = rolling_alpha

    # --- Spread: log(btc) - alpha - beta * log(eth) ---
    # Unshifted version used for z-score computation (before shift at the end)
    spread = log_btc - rolling_alpha - rolling_beta * log_eth
    out["spread"] = spread

    # --- Spread rolling statistics (for z-score) ---
    spread_mean = spread.rolling(zscore_window).mean()
    spread_std = spread.rolling(zscore_window).std() + 1e-10

    out["spread_mean"] = spread_mean
    out["spread_std"] = spread_std

    # --- Z-score and derived features ---
    zscore = (spread - spread_mean) / spread_std
    out["zscore"] = zscore
    out["abs_zscore"] = abs(zscore)

    # Z-score lags (for momentum features)
    out["zscore_lag4"] = zscore.shift(4)
    out["zscore_lag16"] = zscore.shift(16)

    # Z-score momentum
    out["zscore_momentum_4"] = zscore - out["zscore_lag4"]
    out["zscore_momentum_16"] = zscore - out["zscore_lag16"]

    # --- Spread volatility at different timescales ---
    out["spread_std_short"] = spread.rolling(60).std()
    out["spread_std_long"] = spread.rolling(240).std()

    # --- BTC and ETH returns at 4-bar and 16-bar lags ---
    out["btc_return_4"] = np.log(btc_closes / btc_closes.shift(4))
    out["eth_return_4"] = np.log(eth_closes / eth_closes.shift(4))
    out["btc_return_16"] = np.log(btc_closes / btc_closes.shift(16))
    out["eth_return_16"] = np.log(eth_closes / eth_closes.shift(16))

    # --- Rolling correlation between BTC and ETH returns ---
    btc_log_ret = np.log(btc_closes / btc_closes.shift(1))
    eth_log_ret = np.log(eth_closes / eth_closes.shift(1))

    out["btc_eth_corr_60"] = btc_log_ret.rolling(60).corr(eth_log_ret)
    out["btc_eth_corr_240"] = btc_log_ret.rolling(240).corr(eth_log_ret)

    # --- Spread half-life (AR1) ---
    # Rolling application of AR1 half-life computation
    out["spread_halflife"] = spread.rolling(120).apply(_ar1_halflife, raw=False)

    # --- Shift ALL features by 1 bar at the very end to prevent look-ahead bias ---
    feature_cols = [c for c in out.columns]
    out[feature_cols] = out[feature_cols].shift(1)

    return out

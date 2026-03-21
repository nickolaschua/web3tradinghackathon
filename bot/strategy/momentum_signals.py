"""
Pure signal functions for cross-sectional momentum rotation.

All functions operate on 4H OHLCV DataFrames. No state, no I/O.
Used by both backtest and live strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def resample_to_4h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample 15M OHLCV to 4H. Input must have DatetimeIndex and lowercase columns."""
    return df_15m.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["close"])


def sharpe_momentum(closes: pd.Series, lookback: int, skip: int = 1) -> float:
    """
    Volatility-adjusted momentum score.

    Args:
        closes: 4H close prices (DatetimeIndex).
        lookback: Number of 4H bars for return calculation (12 = 48H, 42 = 168H).
        skip: Bars to skip from the end (avoid sub-daily reversal). Default 1.

    Returns:
        Sharpe momentum score (return / vol). Higher = stronger momentum.
    """
    if len(closes) < lookback + skip:
        return 0.0
    end = len(closes) - skip
    start = end - lookback
    if start < 0:
        return 0.0
    log_ret = np.log(closes.iloc[end] / closes.iloc[start])
    log_rets = np.log(closes.iloc[start:end] / closes.iloc[start:end].shift(1)).dropna()
    vol = log_rets.std()
    if vol < 1e-10:
        return 0.0
    return float(log_ret / vol)


def nearness_to_high(closes: pd.Series, window: int = 180) -> float:
    """
    Price anchoring signal: current price / rolling max over window.

    Args:
        closes: 4H close prices.
        window: Lookback for rolling max (180 bars ≈ 30 days at 4H).

    Returns:
        Nearness ratio 0-1. Higher = closer to recent high = bullish.
    """
    if len(closes) < window:
        return 0.5  # neutral
    recent_high = closes.iloc[-window:].max()
    if recent_high <= 0:
        return 0.5
    return float(closes.iloc[-1] / recent_high)


def residual_momentum(
    coin_returns: pd.Series,
    btc_returns: pd.Series,
    beta_window: int = 42,
    signal_window: int = 12,
) -> float:
    """
    BTC-beta-stripped momentum.

    Rolling OLS of coin returns on BTC returns over beta_window.
    Cumulate residuals over last signal_window bars, divide by residual vol.

    Args:
        coin_returns: 4H log returns for this coin.
        btc_returns: 4H log returns for BTC (aligned index).
        beta_window: Bars for rolling OLS beta estimation (42 = 7 days).
        signal_window: Bars to accumulate residuals over (12 = 48H).

    Returns:
        Residual momentum score. Higher = coin outperforming vs BTC.
    """
    if len(coin_returns) < beta_window + signal_window:
        return 0.0

    # Rolling OLS: coin = alpha + beta * btc + epsilon
    # Use the most recent beta_window bars for the regression
    y = coin_returns.iloc[-beta_window:].values
    x = btc_returns.iloc[-beta_window:].values

    # Handle NaN
    valid = ~(np.isnan(y) | np.isnan(x))
    if valid.sum() < 10:
        return 0.0

    y_v, x_v = y[valid], x[valid]
    x_mean = x_v.mean()
    y_mean = y_v.mean()
    cov = ((x_v - x_mean) * (y_v - y_mean)).mean()
    var = ((x_v - x_mean) ** 2).mean()
    beta = cov / (var + 1e-10)
    alpha = y_mean - beta * x_mean

    # Compute residuals for the last signal_window bars
    recent_y = coin_returns.iloc[-signal_window:].values
    recent_x = btc_returns.iloc[-signal_window:].values
    residuals = recent_y - alpha - beta * recent_x

    valid_r = ~np.isnan(residuals)
    if valid_r.sum() < 3:
        return 0.0

    residuals = residuals[valid_r]
    resid_vol = residuals.std()
    if resid_vol < 1e-10:
        return 0.0

    return float(residuals.sum() / resid_vol)


def compute_composite_score(
    closes: pd.Series,
    btc_closes: pd.Series,
    weights: dict[str, float],
) -> float:
    """
    Compute the composite momentum score for one coin.

    Args:
        closes: 4H close prices for this coin.
        btc_closes: 4H close prices for BTC (for residual momentum).
        weights: Dict with keys 'sharpe_48h', 'nearness', 'sharpe_168h', 'residual'.
                 Values are IC-derived weights (sum to 1). Missing keys treated as 0.

    Returns:
        Composite score (higher = stronger momentum signal).
    """
    components = {}

    if weights.get("sharpe_48h", 0) > 0:
        components["sharpe_48h"] = sharpe_momentum(closes, lookback=12, skip=1)

    if weights.get("nearness", 0) > 0:
        components["nearness"] = nearness_to_high(closes, window=180)

    if weights.get("sharpe_168h", 0) > 0:
        components["sharpe_168h"] = sharpe_momentum(closes, lookback=42, skip=1)

    if weights.get("residual", 0) > 0:
        coin_ret = np.log(closes / closes.shift(1)).dropna()
        btc_ret = np.log(btc_closes / btc_closes.shift(1)).dropna()
        # Align
        common = coin_ret.index.intersection(btc_ret.index)
        if len(common) > 60:
            components["residual"] = residual_momentum(
                coin_ret.reindex(common), btc_ret.reindex(common)
            )
        else:
            components["residual"] = 0.0

    score = sum(weights.get(k, 0) * v for k, v in components.items())
    return score


def compute_regime_flag(
    btc_closes_4h: pd.Series,
    median_vol: float | None = None,
) -> str:
    """
    Compute market regime from BTC 30-day return and 7-day volatility.

    Args:
        btc_closes_4h: BTC 4H close prices (at least 180 bars).
        median_vol: Precomputed median vol for threshold. If None, uses 0.03.

    Returns:
        One of: "HIGH_VOL_TREND", "LOW_VOL_TREND", "BEARISH", "SIDEWAYS"
    """
    if len(btc_closes_4h) < 180:
        return "SIDEWAYS"

    ret_30d = np.log(btc_closes_4h.iloc[-1] / btc_closes_4h.iloc[-180])
    rets_7d = np.log(btc_closes_4h.iloc[-42:] / btc_closes_4h.iloc[-42:].shift(1)).dropna()
    vol_7d = rets_7d.std()

    if median_vol is None:
        median_vol = 0.03  # ~48% annualized, reasonable crypto default

    if ret_30d > 0.05 and vol_7d > median_vol:
        return "HIGH_VOL_TREND"
    elif ret_30d > 0 and vol_7d <= median_vol:
        return "LOW_VOL_TREND"
    elif ret_30d < -0.05:
        return "BEARISH"
    else:
        return "SIDEWAYS"


def adjust_weights_for_regime(
    base_weights: dict[str, float],
    regime: str,
    boost_factor: float = 1.3,
) -> dict[str, float]:
    """
    Apply regime-based weight adjustment (multiplicative + re-normalize).

    Args:
        base_weights: IC-derived weights (sum to 1).
        regime: Output of compute_regime_flag().
        boost_factor: Multiplier for the favored component (default 1.3).

    Returns:
        Adjusted weights (sum to 1).
    """
    w = dict(base_weights)

    boost_key = {
        "HIGH_VOL_TREND": "sharpe_48h",
        "LOW_VOL_TREND": "nearness",
        "BEARISH": None,   # no boost, crash protection handles this
        "SIDEWAYS": None,  # use base weights
    }.get(regime)

    if boost_key and boost_key in w:
        w[boost_key] *= boost_factor

    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    return w


def compute_ic(
    scores: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """
    Spearman rank IC between scores and subsequent returns.

    Args:
        scores: Signal scores (one per coin per timestamp).
        forward_returns: Realized returns over the next period.

    Returns:
        Spearman correlation (IC). Positive = signal has predictive power.
    """
    valid = scores.notna() & forward_returns.notna()
    if valid.sum() < 10:
        return 0.0
    corr, _ = stats.spearmanr(scores[valid], forward_returns[valid])
    return float(corr) if not np.isnan(corr) else 0.0

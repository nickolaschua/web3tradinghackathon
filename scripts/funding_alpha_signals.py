#!/usr/bin/env python3
"""
Advanced Funding Rate Alpha Signals - Empirically Calibrated
Based on 2022-2025 BTC/ETH/SOL perpetual funding data analysis
"""
import numpy as np
import pandas as pd

# ============================================================================
# EMPIRICAL PARAMETERS (from analyze_funding_distributions.py)
# ============================================================================
EMPIRICAL_PARAMS = {
    "BTC": {
        "funding_mean": 0.000064, "funding_std": 0.000086,
        "zscore_threshold": 1.50, "cum24h_p95": 0.000915,
        "cum24h_mean": 0.000383, "cum24h_std": 0.000465,
    },
    "ETH": {
        "funding_mean": 0.000059, "funding_std": 0.000118,
        "zscore_threshold": 1.50, "cum24h_p95": 0.001043,
        "cum24h_mean": 0.000348, "cum24h_std": 0.000654,
    },
    "SOL": {
        "funding_mean": -0.000051, "funding_std": 0.000992,
        "zscore_threshold": 1.50, "cum24h_p95": 0.001240,
        "cum24h_mean": -0.000032, "cum24h_std": 0.003304,
    },
    "ETH_BTC_SPREAD": {
        "mean": -0.0000006, "std": 0.000094,
    }
}

# ============================================================================
# SIGNAL 1: Cross-Asset Funding Leadership (ETH→BTC Momentum)
# ============================================================================
def signal_funding_leadership(btc_funding, eth_funding, btc_price_ret=None):
    """
    ETH funding leads BTC funding by 12-24H
    Empirical finding: Q5 spread → +0.20 bps BTC funding change vs -0.11 for Q1

    Args:
        btc_funding: BTC funding rate series
        eth_funding: ETH funding rate series
        btc_price_ret: Optional BTC returns for momentum alignment

    Returns:
        DataFrame with leadership signals
    """
    # ETH-BTC spread
    spread = eth_funding - btc_funding
    spread_mean = EMPIRICAL_PARAMS["ETH_BTC_SPREAD"]["mean"]
    spread_std = EMPIRICAL_PARAMS["ETH_BTC_SPREAD"]["std"]
    spread_zscore = (spread - spread_mean) / spread_std

    # Core signal: positive spread (ETH more aggressive) predicts BTC funding increase
    leadership_raw = spread_zscore * (spread_zscore > 0.5).astype(float)

    # Interaction with price momentum (if provided)
    if btc_price_ret is not None:
        price_momentum = (btc_price_ret > 0).astype(float)
        leadership_aligned = leadership_raw * (1 + price_momentum)  # Amplify when price confirms
    else:
        leadership_aligned = leadership_raw

    return pd.DataFrame({
        "funding_leadership_raw": leadership_raw,
        "funding_leadership_aligned": leadership_aligned,
    }, index=btc_funding.index)


# ============================================================================
# SIGNAL 2: Leverage Imbalance Persistence (Extreme Regime)
# ============================================================================
def signal_leverage_persistence(funding_rate, asset="BTC", volatility=None):
    """
    Persistent extreme funding (3-bar) signals structural imbalance
    Empirical: 3-bar persistence occurs ~4% of time (not noise)
    Mean reversion is WEAK (+0.15 to +1.33 bps), so use as regime filter

    Args:
        funding_rate: Funding rate series
        asset: "BTC", "ETH", or "SOL" for parameter lookup
        volatility: Optional ATR for volatility scaling

    Returns:
        DataFrame with persistence signals
    """
    params = EMPIRICAL_PARAMS.get(asset, EMPIRICAL_PARAMS["BTC"])

    # Z-score using 90-period rolling (empirically validated)
    rolling_mean = funding_rate.rolling(90, min_periods=30).mean()
    rolling_std = funding_rate.rolling(90, min_periods=30).std()
    zscore = (funding_rate - rolling_mean) / rolling_std.replace(0, np.nan)

    # Extreme flag (1.5σ based on empirical p95)
    extreme_flag = (np.abs(zscore) > params["zscore_threshold"]).astype(float)

    # Persistence: count extreme bars in last 3
    persistence = extreme_flag.rolling(3, min_periods=1).sum()
    persistence_signal = (persistence >= 2).astype(float) * np.sign(zscore)

    # Since mean reversion is weak, use this as a REGIME indicator, not directional
    # High persistence = high volatility regime incoming
    regime_flag = (persistence >= 2).astype(float)

    # Volatility interaction
    if volatility is not None:
        vol_zscore = (volatility - volatility.rolling(60).mean()) / volatility.rolling(60).std()
        high_vol_regime = (vol_zscore > 0.5).astype(float)
        regime_amplified = regime_flag * (1 + high_vol_regime)
    else:
        regime_amplified = regime_flag

    return pd.DataFrame({
        "funding_persistence_regime": regime_flag,
        "funding_persistence_vol_adjusted": regime_amplified,
        "funding_zscore": zscore,
    }, index=funding_rate.index)


# ============================================================================
# SIGNAL 3: Cumulative Funding Divergence (Price vs Funding)
# ============================================================================
def signal_funding_divergence(funding_rate, price_returns, asset="BTC"):
    """
    When cumulative 24H funding diverges from price direction
    Empirical: P95 cumulative = 9.15 bps (BTC), 10.43 (ETH), 12.40 (SOL)

    Intuition: Rising prices + falling funding = shorts getting squeezed (bullish)
              Falling prices + rising funding = longs getting squeezed (bearish)

    Args:
        funding_rate: Funding rate series (already 4H aligned)
        price_returns: Price returns (4H)
        asset: "BTC", "ETH", or "SOL"

    Returns:
        DataFrame with divergence signals
    """
    params = EMPIRICAL_PARAMS.get(asset, EMPIRICAL_PARAMS["BTC"])

    # Cumulative 24H funding (6 bars at 4H)
    cum_24h = funding_rate.rolling(6).sum()

    # Standardize
    cum_zscore = (cum_24h - params["cum24h_mean"]) / params["cum24h_std"]

    # Cumulative 24H price returns
    cum_price_ret = price_returns.rolling(6).sum()
    price_direction = np.sign(cum_price_ret)

    # Divergence: price up + funding down (or vice versa)
    # Normalize price returns to [-1, 1] scale
    price_zscore = cum_price_ret / (cum_price_ret.rolling(60).std() + 1e-8)
    price_zscore = price_zscore.clip(-3, 3) / 3  # Scale to [-1, 1]

    # Divergence score: negative correlation = divergence
    divergence = -price_zscore * cum_zscore

    # Strong divergence when |divergence| > 1
    strong_divergence = divergence * (np.abs(divergence) > 1.0).astype(float)

    return pd.DataFrame({
        "funding_divergence_raw": divergence,
        "funding_divergence_strong": strong_divergence,
        "funding_cum24h_zscore": cum_zscore,
    }, index=funding_rate.index)


# ============================================================================
# SIGNAL 4: Volatility-Adjusted Funding Shock (High Vol Amplification)
# ============================================================================
def signal_vol_adjusted_funding(funding_rate, volatility, volume_ratio=None, asset="BTC"):
    """
    Extreme funding matters MORE in high volatility regimes
    Empirical: Extreme events (6% of data) cluster during vol spikes

    Args:
        funding_rate: Funding rate series
        volatility: ATR or realized volatility
        volume_ratio: Optional volume spike indicator
        asset: "BTC", "ETH", or "SOL"

    Returns:
        DataFrame with vol-adjusted signals
    """
    params = EMPIRICAL_PARAMS.get(asset, EMPIRICAL_PARAMS["BTC"])

    # Funding z-score
    rolling_mean = funding_rate.rolling(90, min_periods=30).mean()
    rolling_std = funding_rate.rolling(90, min_periods=30).std()
    funding_zscore = (funding_rate - rolling_mean) / rolling_std.replace(0, np.nan)

    # Volatility regime (z-score)
    vol_mean = volatility.rolling(60, min_periods=20).mean()
    vol_std = volatility.rolling(60, min_periods=20).std()
    vol_zscore = (volatility - vol_mean) / vol_std.replace(0, np.nan)

    # Amplification: extreme funding * high vol
    # Use multiplicative (not additive) to capture regime shift
    vol_amplified = funding_zscore * (1 + vol_zscore.clip(0, 3))

    # Volume confirmation (if provided)
    if volume_ratio is not None:
        volume_spike = (volume_ratio > 1.5).astype(float)
        vol_volume_amplified = vol_amplified * (1 + volume_spike)
    else:
        vol_volume_amplified = vol_amplified

    return pd.DataFrame({
        "funding_vol_amplified": vol_amplified,
        "funding_vol_volume_amplified": vol_volume_amplified,
    }, index=funding_rate.index)


# ============================================================================
# SIGNAL 5: Multi-Asset Funding Composite (Weighted Sentiment Index)
# ============================================================================
def signal_composite_funding_sentiment(btc_funding, eth_funding, sol_funding=None,
                                       btc_vol=None, eth_vol=None, sol_vol=None):
    """
    Combine BTC/ETH/SOL funding into a sentiment index
    Empirical weights from correlation: BTC-ETH: 0.62, BTC-SOL: 0.39

    Weight by inverse volatility (SOL is 10x more volatile, so downweight)

    Args:
        btc_funding, eth_funding, sol_funding: Funding rate series
        btc_vol, eth_vol, sol_vol: Optional volatilities for adaptive weighting

    Returns:
        DataFrame with composite signals
    """
    # Standardize each funding series
    def standardize(fr, mean, std):
        return (fr - mean) / std

    btc_std = standardize(btc_funding,
                          EMPIRICAL_PARAMS["BTC"]["funding_mean"],
                          EMPIRICAL_PARAMS["BTC"]["funding_std"])
    eth_std = standardize(eth_funding,
                          EMPIRICAL_PARAMS["ETH"]["funding_mean"],
                          EMPIRICAL_PARAMS["ETH"]["funding_std"])

    # Base weights (inverse volatility from empirical data)
    # BTC: 0.86 bps, ETH: 1.18 bps, SOL: 9.92 bps
    btc_weight = 1.0 / 0.86
    eth_weight = 1.0 / 1.18
    total_weight = btc_weight + eth_weight

    if sol_funding is not None:
        sol_std = standardize(sol_funding,
                              EMPIRICAL_PARAMS["SOL"]["funding_mean"],
                              EMPIRICAL_PARAMS["SOL"]["funding_std"])
        sol_weight = 1.0 / 9.92  # Much lower weight due to high noise
        total_weight += sol_weight
        composite = (btc_weight * btc_std + eth_weight * eth_std + sol_weight * sol_std) / total_weight
    else:
        composite = (btc_weight * btc_std + eth_weight * eth_std) / total_weight

    # Extreme composite signal
    composite_extreme = composite * (np.abs(composite) > 1.5).astype(float)

    # Directional consensus: all 3 agree
    if sol_funding is not None:
        consensus = ((np.sign(btc_std) == np.sign(eth_std)) &
                    (np.sign(eth_std) == np.sign(sol_std))).astype(float)
        consensus_signal = composite * consensus
    else:
        consensus = (np.sign(btc_std) == np.sign(eth_std)).astype(float)
        consensus_signal = composite * consensus

    return pd.DataFrame({
        "funding_composite": composite,
        "funding_composite_extreme": composite_extreme,
        "funding_consensus": consensus_signal,
    }, index=btc_funding.index)


# ============================================================================
# MAIN: Compute All Signals
# ============================================================================
def compute_all_funding_signals(btc_df, funding_dfs, ohlcv_dfs=None):
    """
    Compute all 5 advanced funding signals

    Args:
        btc_df: BTC OHLCV dataframe (4H, must have 'close', 'atr_proxy' if available)
        funding_dfs: Dict of {symbol: funding_df} for BTCUSDT, ETHUSDT, SOLUSDT
        ohlcv_dfs: Optional dict of OHLCV dataframes for volume features

    Returns:
        DataFrame with all signals aligned to btc_df index
    """
    # Align funding to 4H
    btc_funding_4h = funding_dfs.get("BTCUSDT", pd.DataFrame())["funding_rate"].resample("4h").ffill().reindex(btc_df.index, method="ffill")
    eth_funding_4h = funding_dfs.get("ETHUSDT", pd.DataFrame())["funding_rate"].resample("4h").ffill().reindex(btc_df.index, method="ffill")
    sol_funding_4h = funding_dfs.get("SOLUSDT", pd.DataFrame())["funding_rate"].resample("4h").ffill().reindex(btc_df.index, method="ffill") if "SOLUSDT" in funding_dfs else None

    # BTC price returns
    btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1))

    # Volatility (use atr_proxy if available, else compute)
    if "atr_proxy" in btc_df.columns:
        btc_vol = btc_df["atr_proxy"]
    else:
        btc_vol = btc_ret.rolling(14).std() * btc_df["close"]

    # Volume ratio (if available)
    if ohlcv_dfs and "quote_volume" in btc_df.columns:
        volume_ratio = btc_df["quote_volume"] / btc_df["quote_volume"].rolling(20).mean()
    else:
        volume_ratio = None

    # Compute all signals
    signals = btc_df[[]].copy()  # Empty df with same index

    # Signal 1: Leadership
    if not eth_funding_4h.isna().all():
        sig1 = signal_funding_leadership(btc_funding_4h, eth_funding_4h, btc_ret)
        signals = signals.join(sig1)

    # Signal 2: Persistence
    sig2 = signal_leverage_persistence(btc_funding_4h, asset="BTC", volatility=btc_vol)
    signals = signals.join(sig2)

    # Signal 3: Divergence
    sig3 = signal_funding_divergence(btc_funding_4h, btc_ret, asset="BTC")
    signals = signals.join(sig3)

    # Signal 4: Vol-adjusted
    sig4 = signal_vol_adjusted_funding(btc_funding_4h, btc_vol, volume_ratio, asset="BTC")
    signals = signals.join(sig4)

    # Signal 5: Composite
    if not eth_funding_4h.isna().all():
        sig5 = signal_composite_funding_sentiment(btc_funding_4h, eth_funding_4h, sol_funding_4h)
        signals = signals.join(sig5)

    # Shift all signals by 1 to avoid lookahead bias
    signals = signals.shift(1)

    return signals


if __name__ == "__main__":
    print("This module provides funding alpha signal computation functions.")
    print("Import and use compute_all_funding_signals() in your pipeline.")

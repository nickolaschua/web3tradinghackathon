"""
15-minute feature computation for correlation, mean-reversion, and rotation strategies.

Combines:
1. Core technical indicators (RSI, MACD, EMAs, Bollinger Bands)
2. Cross-asset correlation features (BTC-ETH correlation, relative returns)
3. Funding rate features (z-score for sentiment)
4. Multi-asset momentum for rotation strategies
5. Volume analysis

All features are shifted 1 bar to prevent look-ahead bias.
Pure functions safe for both backtest and live environments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


def compute_15m_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 15-minute technical indicators with all features for new strategies.

    Adds:
    - RSI (14 and 7 period for faster signals on 15m)
    - MACD (12, 26, 9)
    - EMAs (20, 50)
    - Bollinger Bands (upper, lower, position, width)
    - Volume indicators (MA, ratio)
    - Returns and volatility measures
    - ATR proxy for risk management

    All indicator columns are shifted 1 bar after computation.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with original OHLCV + shifted indicator columns
    """
    ohlcv_cols = {"open", "high", "low", "close", "volume"}
    out = df.copy()
    out.columns = out.columns.str.lower()

    # --- ATR proxy (consistent with core features.py) ---
    log_ret = np.log(out["close"] / out["close"].shift(1))
    out["atr_proxy"] = log_ret.rolling(14).std() * out["close"] * 1.25

    # --- pandas-ta-classic indicators ---
    out.ta.rsi(length=14, append=True)  # RSI_14
    out.ta.rsi(length=7, append=True)   # RSI_7 (faster for 15m)
    out.ta.macd(fast=12, slow=26, signal=9, append=True)  # MACD_12_26_9, MACDs, MACDh
    out.ta.ema(length=20, append=True)  # EMA_20
    out.ta.ema(length=50, append=True)  # EMA_50

    # Rename to lowercase for consistency
    out.rename(columns={
        "RSI_14": "rsi",
        "RSI_7": "rsi_7",
        "EMA_20": "ema_20",
        "EMA_50": "ema_50",
        "MACD_12_26_9": "macd",
        "MACDs_12_26_9": "macd_signal",
        "MACDh_12_26_9": "macd_hist"
    }, inplace=True)

    # --- Bollinger Bands (manual for explicit control) ---
    sma20 = out["close"].rolling(20).mean()
    std20 = out["close"].rolling(20).std()
    out["bb_upper"] = sma20 + 2.0 * std20
    out["bb_lower"] = sma20 - 2.0 * std20
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (sma20 + 1e-10)
    out["bb_pos"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-10)

    # --- Volume indicators ---
    out["volume_ma_20"] = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / (out["volume_ma_20"] + 1e-10)

    # --- Returns and volatility ---
    out["btc_return"] = out["close"].pct_change()
    out["returns_std_20"] = out["btc_return"].rolling(20).std()

    # --- Momentum (12-bar for rotation strategy = 3 hours on 15m) ---
    out["btc_momentum_12"] = out["close"].pct_change(12)

    # --- Shift all indicator columns 1 bar ---
    ind_cols = [c for c in out.columns if c not in ohlcv_cols]
    out[ind_cols] = out[ind_cols].shift(1)

    return out


def compute_cross_asset_15m_features(
    btc_df: pd.DataFrame,
    eth_df: pd.DataFrame,
    sol_df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add cross-asset correlation, relative returns, and funding features to BTC 15m data.

    Features added:
    - btc_eth_corr: 20-bar rolling correlation of BTC-ETH returns
    - btc_sol_corr: 20-bar rolling correlation of BTC-SOL returns
    - eth_return: ETH 1-bar return (aligned to BTC index)
    - sol_return: SOL 1-bar return
    - eth_momentum_12: ETH 12-bar momentum (for rotation)
    - sol_momentum_12: SOL 12-bar momentum
    - eth_volume, eth_volume_ma: ETH volume features
    - sol_volume, sol_volume_ma: SOL volume features
    - btc_volume, btc_volume_ma: BTC volume features (already in btc_df)
    - btc_funding_zscore: Funding rate z-score (90-period, resampled to 15m)

    Args:
        btc_df: BTC 15m DataFrame (already processed by compute_15m_features)
        eth_df: ETH 15m raw OHLCV DataFrame
        sol_df: SOL 15m raw OHLCV DataFrame
        funding_df: Optional BTC funding rate DataFrame (8h intervals)

    Returns:
        BTC DataFrame with cross-asset features added (not shifted again - already shifted)
    """
    out = btc_df.copy()

    # Ensure index is datetime
    out.index = pd.to_datetime(out.index, utc=True)

    # --- ETH features ---
    if not eth_df.empty:
        eth = eth_df.copy()
        eth.columns = eth.columns.str.lower()
        eth.index = pd.to_datetime(eth.index, utc=True)

        # Align to BTC index
        eth_close = eth["close"].reindex(out.index, method="ffill")
        eth_volume = eth["volume"].reindex(out.index, method="ffill")

        # ETH returns and momentum
        out["eth_return"] = eth_close.pct_change()
        out["eth_momentum_12"] = eth_close.pct_change(12)

        # ETH volume features
        out["eth_volume"] = eth_volume
        out["eth_volume_ma"] = eth_volume.rolling(20).mean()

        # BTC-ETH correlation (20-bar rolling correlation of returns)
        out["btc_eth_corr"] = out["btc_return"].rolling(20).corr(out["eth_return"])
    else:
        out["eth_return"] = np.nan
        out["eth_momentum_12"] = np.nan
        out["eth_volume"] = np.nan
        out["eth_volume_ma"] = np.nan
        out["btc_eth_corr"] = np.nan

    # --- SOL features ---
    if not sol_df.empty:
        sol = sol_df.copy()
        sol.columns = sol.columns.str.lower()
        sol.index = pd.to_datetime(sol.index, utc=True)

        # Align to BTC index
        sol_close = sol["close"].reindex(out.index, method="ffill")
        sol_volume = sol["volume"].reindex(out.index, method="ffill")

        # SOL returns and momentum
        out["sol_return"] = sol_close.pct_change()
        out["sol_momentum_12"] = sol_close.pct_change(12)

        # SOL volume features
        out["sol_volume"] = sol_volume
        out["sol_volume_ma"] = sol_volume.rolling(20).mean()

        # BTC-SOL correlation
        out["btc_sol_corr"] = out["btc_return"].rolling(20).corr(out["sol_return"])
    else:
        out["sol_return"] = np.nan
        out["sol_momentum_12"] = np.nan
        out["sol_volume"] = np.nan
        out["sol_volume_ma"] = np.nan
        out["btc_sol_corr"] = np.nan

    # --- BTC volume features (from btc_df, but ensure volume_ma is present) ---
    if "btc_volume" not in out.columns:
        out["btc_volume"] = out["volume"]
    if "btc_volume_ma" not in out.columns:
        out["btc_volume_ma"] = out["volume"].rolling(20).mean()

    # --- Funding rate features ---
    if funding_df is not None and not funding_df.empty:
        funding = funding_df.copy()
        funding.index = pd.to_datetime(funding.index, utc=True)

        # Resample 8h funding to 15m via forward-fill
        funding_rate = funding["funding_rate"].reindex(out.index, method="ffill")

        # Z-score: (current - rolling_mean) / rolling_std
        funding_ma = funding_rate.rolling(90).mean()
        funding_std = funding_rate.rolling(90).std()
        out["btc_funding_zscore"] = ((funding_rate - funding_ma) / funding_std.replace(0, np.nan))
    else:
        out["btc_funding_zscore"] = np.nan

    # Shift all newly added cross-asset features by 1 bar
    cross_asset_cols = [
        "eth_return", "eth_momentum_12", "eth_volume", "eth_volume_ma", "btc_eth_corr",
        "sol_return", "sol_momentum_12", "sol_volume", "sol_volume_ma", "btc_sol_corr",
        "btc_funding_zscore"
    ]
    existing_cols = [c for c in cross_asset_cols if c in out.columns]
    out[existing_cols] = out[existing_cols].shift(1)

    return out


def prepare_15m_features(
    btc_path: str,
    eth_path: str | None = None,
    sol_path: str | None = None,
    funding_path: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load 15m data, compute all features, and return ready-to-use feature matrix.

    Pipeline:
    1. Load BTC 15m parquet
    2. Compute base 15m features on BTC
    3. Load ETH/SOL/funding if provided (will resample 4h to 15m if needed)
    4. Compute cross-asset features
    5. Drop NaN warmup rows
    6. Filter by date range

    Args:
        btc_path: Path to BTC 15m parquet
        eth_path: Optional path to ETH parquet (15m or 4h, will be resampled)
        sol_path: Optional path to SOL parquet (15m or 4h, will be resampled)
        funding_path: Optional path to BTC funding rate parquet
        start: Optional start date (e.g., "2024-01-01")
        end: Optional end date (e.g., "2024-12-31")

    Returns:
        Feature DataFrame with UTC DatetimeIndex, all NaN rows dropped
    """
    # Load BTC
    btc = pd.read_parquet(btc_path)
    btc.columns = btc.columns.str.lower()
    if not isinstance(btc.index, pd.DatetimeIndex):
        btc.index = pd.to_datetime(btc.index, utc=True)

    # Compute base features
    feat = compute_15m_features(btc)

    # Load and resample cross-asset data if needed
    def load_and_resample(path):
        if not path:
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df.columns = df.columns.str.lower()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        # Check if resampling needed (4h data → 15m)
        # Infer frequency by looking at first two timestamps
        if len(df) > 1:
            freq = df.index[1] - df.index[0]
            if freq >= pd.Timedelta(hours=1):
                # Resample 4h → 15m via forward-fill
                df = df.resample("15min").ffill()

        return df

    eth = load_and_resample(eth_path)
    sol = load_and_resample(sol_path)
    funding = pd.read_parquet(funding_path) if funding_path else None

    feat = compute_cross_asset_15m_features(feat, eth, sol, funding)

    # Drop warmup NaN rows
    feat = feat.dropna()

    # Filter by date range
    if start:
        feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]

    return feat


__all__ = [
    "compute_15m_features",
    "compute_cross_asset_15m_features",
    "prepare_15m_features",
]

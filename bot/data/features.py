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
    out.ta.rsi(length=7, append=True)                          # → RSI_7  (faster, useful at 15M)
    out.ta.macd(fast=12, slow=26, signal=9, append=True)       # → MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    out.ta.ema(length=20, append=True)                         # → EMA_20
    out.ta.ema(length=50, append=True)                         # → EMA_50

    # --- EMA slope (manual — df.ta.slope(close="EMA_20") kwarg form is unverified) ---
    # Rate of change of EMA_20: (EMA_20[t] - EMA_20[t-1]) / EMA_20[t-1]
    out["ema_slope"] = (out["EMA_20"] - out["EMA_20"].shift(1)) / out["EMA_20"].shift(1)

    # --- Bollinger Band width and position (manual to avoid bbands column-name ambiguity) ---
    sma20 = out["close"].rolling(20).mean()
    std20 = out["close"].rolling(20).std()
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    # bb_width: normalized band width — high = volatile regime, low = squeeze
    out["bb_width"] = (bb_upper - bb_lower) / (sma20 + 1e-10)
    # bb_pos: percent B — 0=at lower band, 0.5=at mid, 1=at upper band, can exceed [0,1]
    out["bb_pos"] = (out["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # --- Volume ratio: current bar volume vs 20-bar rolling mean ---
    # Detects volume spikes that often precede or confirm breakouts.
    # Safe on Roostoo (volume=0): 0 / 0 → NaN, handled by dropna() downstream.
    out["volume_ratio"] = out["volume"] / (out["volume"].rolling(20).mean() + 1e-10)

    # --- Candle body ratio: measures directional conviction of each bar ---
    # +1 = full bullish body (close=high, open=low), -1 = full bearish, 0 = doji.
    # Normalized by high-low range; NaN when H=L (Roostoo synthetic candles).
    hl_range = out["high"] - out["low"]
    out["candle_body"] = (out["close"] - out["open"]) / (hl_range + 1e-10)

    # --- Shift all indicator columns 1 bar to prevent look-ahead bias ---
    # CRITICAL: shift AFTER computing indicators, never before.
    # Shifting close prices first would make indicators compute on bar N-1 close
    # but label it as bar N — a subtle off-by-one that causes garbage signal values.
    ind_cols = [c for c in out.columns if c not in ohlcv_cols]
    out[ind_cols] = out[ind_cols].shift(1)

    return out


def compute_cross_asset_features(
    btc_df: pd.DataFrame,
    other_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Inject cross-asset lagged log-return features into the BTC feature DataFrame.

    For each pair in `other_dfs`, computes 1-bar and 2-bar lagged log returns
    and appends them as columns to `btc_df`. These represent prior-period
    movement in correlated assets — a common momentum/spillover feature.

    IMPORTANT: This function must be called AFTER compute_features() but BEFORE
    dropna(). If dropna() runs first, the ETH/SOL columns do not yet exist and
    pandas silently drops every row.

    Column naming:
    - "ETH/USD" → prefix "eth": eth_return_lag1, eth_return_lag2
    - "SOL/USD" → prefix "sol": sol_return_lag1, sol_return_lag2
    - General: f"{pair.split('/')[0].lower()}_return_lag{n}"

    Cross-asset columns are NOT shifted an additional bar — the lag is already
    encoded in the column name (lag1 = yesterday's return, lag2 = two days ago).
    These are aligned to the same index as btc_df via pandas reindex.

    Args:
        btc_df: BTC feature DataFrame (already processed by compute_features).
                Must have a DatetimeIndex.
        other_dfs: Dict mapping pair symbol → raw OHLCV DataFrame.
                   Empty dict is valid — function returns btc_df unchanged.

    Returns:
        btc_df with additional cross-asset lag columns (new DataFrame, not in-place).
    """
    out = btc_df.copy()

    for pair, df in other_dfs.items():
        if df.empty:
            continue

        # Normalize column names (Binance Parquet may be capitalized)
        df = df.copy()
        df.columns = df.columns.str.lower()

        prefix = pair.split("/")[0].lower()   # "ETH/USD" → "eth"

        # Log returns for this asset
        log_ret = np.log(df["close"] / df["close"].shift(1))

        # lag1 = yesterday's return; lag2 = two days ago
        lag1 = log_ret.shift(1)
        lag2 = log_ret.shift(2)

        # Align to btc_df index — fills NaN where timestamps don't overlap
        out[f"{prefix}_return_lag1"] = lag1.reindex(out.index)
        out[f"{prefix}_return_lag2"] = lag2.reindex(out.index)

    return out


def compute_btc_context_features(
    btc_feat_df: pd.DataFrame,
    eth_df: pd.DataFrame,
    sol_df: pd.DataFrame,
    window: int = 180,
) -> pd.DataFrame:
    """
    Add BTC/altcoin context features based on rolling correlation, beta, and return divergence.

    All new columns are shifted 1 bar to match the convention in compute_features().

    New columns added:
      eth_btc_corr        : rolling {window}-bar Spearman correlation of ETH and BTC log-returns
      sol_btc_corr        : rolling {window}-bar Spearman correlation of SOL and BTC log-returns
      eth_btc_beta        : rolling OLS beta of ETH returns on BTC returns (cov/var)
      sol_btc_beta        : same for SOL
      eth_btc_divergence  : ETH log-return minus BTC log-return (lag-1 bar), captures ETH leading BTC
      sol_btc_divergence  : same for SOL

    Args:
        btc_feat_df : Output of compute_features() + compute_cross_asset_features().
                      Must have a DatetimeIndex (UTC) and a 'close' column (unshifted).
        eth_df      : Raw OHLCV DataFrame for ETH (any capitalisation).
        sol_df      : Raw OHLCV DataFrame for SOL (any capitalisation).
        window      : Rolling window in bars for correlation and beta (default 20 = ~3.3 days at 4H).

    Returns:
        New DataFrame with six additional columns.
    """
    out = btc_feat_df.copy()

    btc_ret = np.log(out["close"] / out["close"].shift(1))

    for symbol, raw_df in [("eth", eth_df), ("sol", sol_df)]:
        raw = raw_df.copy()
        raw.columns = raw.columns.str.lower()

        alt_ret = np.log(raw["close"] / raw["close"].shift(1)).reindex(out.index, method="ffill")

        # Rolling correlation (Pearson on log returns; shift 1 for look-ahead prevention)
        corr = btc_ret.rolling(window).corr(alt_ret)
        out[f"{symbol}_btc_corr"] = corr.shift(1)

        # Rolling beta: cov(btc, alt) / var(btc) — how much alt moves per unit of BTC
        cov = btc_ret.rolling(window).cov(alt_ret)
        var = btc_ret.rolling(window).var()
        out[f"{symbol}_btc_beta"] = (cov / (var + 1e-10)).shift(1)

        # Return divergence: alt outperformed/underperformed BTC last bar (lead-lag signal)
        out[f"{symbol}_btc_divergence"] = (alt_ret - btc_ret).shift(1)

    return out


def compute_market_context_features(
    coin_feat_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    eth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add BTC and ETH lagged returns as market context features.

    These are the same values for all coins at each timestamp — they capture
    "what did the market leaders just do?" which is predictive for all coins.

    New columns (all shifted 1 bar):
      btc_return_4h : BTC 16-bar lagged log return
      btc_return_1d : BTC 96-bar lagged log return
      eth_return_4h : ETH 16-bar lagged log return
      eth_return_1d : ETH 96-bar lagged log return
    """
    out = coin_feat_df.copy()

    for prefix, df in [("btc", btc_df), ("eth", eth_df)]:
        raw = df.copy()
        raw.columns = raw.columns.str.lower()
        log_ret = np.log(raw["close"] / raw["close"].shift(1))
        out[f"{prefix}_return_4h"] = log_ret.shift(16).reindex(out.index)
        out[f"{prefix}_return_1d"] = log_ret.shift(96).reindex(out.index)

    return out


def compute_coin_identity_features(
    coin_feat_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    liquidity_tier: int,
    all_atr_proxies: dict[str, pd.Series] | None = None,
    window: int = 2880,
) -> pd.DataFrame:
    """
    Add coin-to-market identity features that tell the model what kind of coin
    it is looking at.

    New columns:
      btc_corr_30d    : Rolling correlation of this coin's returns vs BTC (shifted 1 bar)
      relative_vol    : This coin's atr_proxy / BTC's atr_proxy (shifted 1 bar)
      vol_rank        : Percentile rank of this coin's atr_proxy across all coins (shifted 1 bar)
      liquidity_tier  : Static categorical (1=mega, 2=large, 3=mid)
    """
    out = coin_feat_df.copy()

    # BTC correlation
    raw_btc = btc_df.copy()
    raw_btc.columns = raw_btc.columns.str.lower()
    btc_ret = np.log(raw_btc["close"] / raw_btc["close"].shift(1)).reindex(out.index)
    coin_ret = np.log(out["close"] / out["close"].shift(1))
    out["btc_corr_30d"] = coin_ret.rolling(window).corr(btc_ret).shift(1)

    # Relative volatility
    btc_atr = (
        np.log(raw_btc["close"] / raw_btc["close"].shift(1))
        .rolling(14).std() * raw_btc["close"] * 1.25
    ).reindex(out.index)
    out["relative_vol"] = (out["atr_proxy"] / (btc_atr + 1e-10)).shift(1)

    # Vol rank (cross-sectional percentile)
    if all_atr_proxies is not None and len(all_atr_proxies) > 1:
        atr_panel = pd.DataFrame(all_atr_proxies).reindex(out.index)
        coin_col = None
        for col, series in all_atr_proxies.items():
            if col == "self":
                coin_col = col
                break
        if coin_col is not None:
            out["vol_rank"] = atr_panel.rank(axis=1, pct=True)[coin_col].shift(1)
        else:
            out["vol_rank"] = atr_panel.rank(axis=1, pct=True).iloc[:, 0].shift(1)
    else:
        out["vol_rank"] = 0.5

    # Liquidity tier (static)
    out["liquidity_tier"] = liquidity_tier

    return out


__all__ = [
    "compute_features",
    "compute_cross_asset_features",
    "compute_btc_context_features",
    "compute_market_context_features",
    "compute_coin_identity_features",
]

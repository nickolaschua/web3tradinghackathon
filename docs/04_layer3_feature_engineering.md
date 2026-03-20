# Layer 3 — Feature Engineering

## What This Layer Does

Feature Engineering transforms raw OHLCV price data into the set of numerical inputs that your strategy logic uses to make decisions. It sits between the Data Pipeline and the Strategy Engine, and is called identically by both the backtesting system and the live bot.

This layer computes technical indicators, cross-asset signals, sentiment inputs, and rolling statistical features. It handles all normalisation, NaN cleaning, and look-ahead bias prevention. By the time data leaves this layer, it is a clean, shift-corrected feature matrix ready for signal thresholding or model input.

**This layer is deployed on EC2.** The same `features.py` file runs in both research and production.

---

## What This Layer Is Trying to Achieve

1. Produce a consistent, reproducible set of features from raw price data
2. Prevent look-ahead bias at the source — features should only use information available at the time the signal would have been generated
3. Ensure every feature is normalised appropriately so that threshold values validated in backtesting transfer to live trading
4. Make the feature set extensible — adding a new feature during the competition should require only adding it to this file without touching the strategy layer

---

## How It Contributes to the Bigger Picture

This is the layer where your intellectual edge is defined. The API client, data pipeline, and infrastructure are table stakes. The feature set is where you make informed choices about what signals have predictive power for crypto returns.

Every feature in this layer was either validated by IC analysis in the research phase (Spearman correlation with forward returns > 0.02 consistently) or serves a structural purpose (e.g., ATR for position sizing, ADX for regime detection). Nothing is here because it seemed like a good idea — everything should be justifiable by backtested evidence.

---

## Files in This Layer

```
data/
└── features.py         Shared indicator library (historical + live)

backtesting/
└── ic_analysis.py      Research tool: measures predictive power of each feature
```

---

## `data/features.py`

### Architecture Principles

**Single shared file:** features.py is called by `vbt_sweep.py`, `walk_forward.py`, `bt_validator.py`, and `live_fetcher.py` (via the main loop). If you ever find yourself writing feature computation logic twice, stop and refactor it into this file.

**Shift enforcement:** Every feature series is shifted forward by 1 bar before it is returned. This ensures that signal generated at the close of bar N uses data from bar N, not bar N+1. Forgetting this single rule is the most common source of look-ahead bias.

**Expanding normalisation:** Where z-scoring is needed, use expanding window mean and std (growing from the start of the series), never global mean and std computed on the full dataset. Global normalisation uses future data.

**dropna enforcement:** After all computation, drop NaN rows. This is the safety net for indicator warmup periods. Log the number of rows dropped so you can verify the warmup is as expected.

```python
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

def compute_features(df: pd.DataFrame, pair: str = None) -> pd.DataFrame:
    """
    Compute full feature matrix from OHLCV DataFrame.
    
    Input:  DataFrame with columns [open, high, low, close, volume]
            indexed by UTC datetime, monotonic, no NaN
    Output: DataFrame of features, shift(1) applied, dropna enforced
    
    CRITICAL: This function must be called identically in backtesting and live.
    Do not add conditional logic that differs between environments.
    """
    feat = pd.DataFrame(index=df.index)

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Momentum indicators ────────────────────────────────────────────────

    # RSI: overbought/oversold. Values < 30 = oversold, > 70 = overbought.
    feat["rsi_14"] = ta.rsi(c, length=14)

    # MACD histogram: momentum direction and acceleration
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    feat["macd_hist"] = macd["MACDh_12_26_9"]
    feat["macd_line"] = macd["MACD_12_26_9"]
    feat["macd_signal"] = macd["MACDs_12_26_9"]

    # ── Volatility indicators ──────────────────────────────────────────────

    # ATR: absolute volatility in price units. Used for stop sizing, NOT normalised.
    feat["atr_14"] = ta.atr(h, l, c, length=14)

    # Bollinger Band %B: where price sits within the bands. 0=lower, 1=upper.
    bb = ta.bbands(c, length=20, std=2)
    feat["bb_pct_b"] = bb["BBP_20_2.0"]
    feat["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]

    # Rolling volatility: standard deviation of log returns
    log_returns = np.log(c / c.shift(1))
    feat["vol_4h"] = log_returns.rolling(6).std()     # 24H lookback on 4H bars
    feat["vol_24h"] = log_returns.rolling(24).std()   # 4D lookback on 4H bars

    # ── Trend indicators ───────────────────────────────────────────────────

    # ADX: trend strength. > 25 = trending, < 20 = ranging.
    adx = ta.adx(h, l, c, length=14)
    feat["adx_14"] = adx["ADX_14"]
    feat["dmp_14"] = adx["DMP_14"]  # Directional movement positive
    feat["dmn_14"] = adx["DMN_14"]  # Directional movement negative

    # EMA crossovers: directional bias
    feat["ema_20"] = ta.ema(c, length=20)
    feat["ema_50"] = ta.ema(c, length=50)
    feat["ema_200"] = ta.ema(c, length=200)
    feat["ema_20_50_spread"] = (feat["ema_20"] - feat["ema_50"]) / c
    feat["price_vs_ema200"] = (c - feat["ema_200"]) / c

    # ── Volume indicators ──────────────────────────────────────────────────

    # OBV: cumulative volume direction
    feat["obv"] = ta.obv(c, v)
    feat["obv_ema"] = ta.ema(feat["obv"], length=20)
    feat["obv_signal"] = feat["obv"] - feat["obv_ema"]  # OBV divergence from its trend

    # Volume z-score: how unusual is current volume relative to recent history
    v_roll_mean = v.rolling(20).mean()
    v_roll_std = v.rolling(20).std()
    feat["volume_zscore"] = (v - v_roll_mean) / (v_roll_std + 1e-10)

    # ── Return features ────────────────────────────────────────────────────

    # Multi-horizon returns: where did price come from
    feat["return_1"] = c.pct_change(1)    # 4H (one bar on 4H data)
    feat["return_6"] = c.pct_change(6)    # 24H
    feat["return_42"] = c.pct_change(42)  # 1W (42 × 4H bars)
    feat["return_126"] = c.pct_change(126) # 3W

    # ── Cross-asset features (BTC as leading indicator for alts) ──────────
    # Only populated if pair is provided and is NOT BTC.
    # BTC lead: BTC's 4H return tends to predict alt moves 1–3 bars ahead.
    # This is computed externally and injected here if available.
    # Placeholder columns — filled by compute_cross_asset_features() below.
    feat["btc_return_lag1"] = np.nan
    feat["btc_return_lag2"] = np.nan

    # ── Expanding normalisation for z-scored features ──────────────────────
    # CRITICAL: Use expanding() not global mean/std. Global uses future data.
    for col in ["rsi_14", "macd_hist", "bb_pct_b", "volume_zscore",
                "ema_20_50_spread", "price_vs_ema200", "obv_signal"]:
        if col in feat.columns:
            exp_mean = feat[col].expanding().mean()
            exp_std = feat[col].expanding().std()
            feat[f"{col}_z"] = (feat[col] - exp_mean) / (exp_std + 1e-10)

    # ── CRITICAL: Shift all features forward by 1 bar ─────────────────────
    # This prevents look-ahead bias. A signal using bar N's close
    # should only see features computed from bar N-1 and earlier.
    n_cols_before = len(feat.columns)
    feat = feat.shift(1)
    assert len(feat.columns) == n_cols_before, "Column count changed after shift"

    # ── Drop NaN rows from indicator warmup period ────────────────────────
    n_before = len(feat)
    feat = feat.dropna()
    n_after = len(feat)
    warmup_bars = n_before - n_after
    if warmup_bars > 0:
        logger.debug(f"Dropped {warmup_bars} warmup bars ({warmup_bars * 4}H on 4H data)")

    return feat


def compute_cross_asset_features(btc_df: pd.DataFrame,
                                  target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add BTC return lag features to a non-BTC pair's feature DataFrame.
    Must be called after compute_features() for both BTC and the target pair.
    """
    btc_return = np.log(btc_df["close"] / btc_df["close"].shift(1))
    target_df["btc_return_lag1"] = btc_return.shift(1).reindex(target_df.index)
    target_df["btc_return_lag2"] = btc_return.shift(2).reindex(target_df.index)
    return target_df.dropna()
```

---

## `backtesting/ic_analysis.py`

Before building your strategy, run IC analysis to determine which features actually have predictive power. This is what separates a strategy built on evidence from one built on intuition.

**Information Coefficient (IC)** is the Spearman rank correlation between a feature value at time T and the actual return from T to T+N. An IC of 0.02 means 2% rank correlation — small but statistically meaningful across thousands of observations. An IC of 0.05+ is considered strong for liquid markets.

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

def compute_ic(features: pd.DataFrame, forward_returns: pd.Series,
               horizon_label: str = "4H") -> pd.DataFrame:
    """
    Compute Spearman IC for each feature column against forward returns.
    Returns DataFrame with IC, p-value, and t-statistic per feature.
    """
    results = []
    for col in features.columns:
        aligned = features[col].dropna().align(forward_returns.dropna(), join="inner")
        f_vals, r_vals = aligned
        if len(f_vals) < 30:
            continue
        ic, pval = stats.spearmanr(f_vals, r_vals)
        t_stat = ic * np.sqrt((len(f_vals) - 2) / (1 - ic**2 + 1e-10))
        results.append({
            "feature": col,
            "horizon": horizon_label,
            "ic": ic,
            "abs_ic": abs(ic),
            "pval": pval,
            "t_stat": t_stat,
            "n_obs": len(f_vals),
            "significant": pval < 0.05,
        })
    return pd.DataFrame(results).sort_values("abs_ic", ascending=False)

def ic_decay_analysis(features: pd.DataFrame, prices: pd.Series,
                      horizons: list[int] = [1, 3, 6, 12, 24, 42]) -> pd.DataFrame:
    """
    Compute IC at multiple forward horizons to understand signal persistence.
    A signal with IC that decays slowly is more useful for swing trading.
    """
    all_results = []
    for h in horizons:
        fwd_returns = prices.shift(-h) / prices - 1
        ic_df = compute_ic(features, fwd_returns, horizon_label=f"{h}bar")
        ic_df["horizon_bars"] = h
        all_results.append(ic_df)
    return pd.concat(all_results, ignore_index=True)

def plot_ic_summary(ic_df: pd.DataFrame, top_n: int = 10):
    """Plot IC bar chart for top features."""
    top = ic_df.nlargest(top_n, "abs_ic")
    colors = ["green" if v > 0 else "red" for v in top["ic"]]
    plt.figure(figsize=(12, 6))
    plt.barh(top["feature"], top["ic"], color=colors, alpha=0.7)
    plt.axvline(x=0.02, color="blue", linestyle="--", label="IC=0.02 threshold")
    plt.axvline(x=-0.02, color="blue", linestyle="--")
    plt.xlabel("Spearman IC")
    plt.title("Feature Information Coefficient vs Forward 4H Returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ic_analysis.png", dpi=150)
    plt.show()
```

**What to look for:**
- Any feature with consistent `abs_ic > 0.02` across multiple time periods is worth keeping
- Features where IC decays slowly (still meaningful at 6+ bar horizon) are ideal for 4H swing trading
- Features where IC spikes at 1 bar but collapses at 3+ bars are noise trading signals — avoid
- Negative IC is fine — it just means the signal needs to be inverted
- If two features have IC > 0.02 but correlate > 0.8 with each other, keep only one

**How to use the results:**
Run this analysis on 2020–2023 data (your in-sample research period). The top 5–7 features by consistent IC become the inputs to your strategy's signal score. Do not use all 30+ features — more features does not mean better signals, it means more overfitting surface.

---

## Feature Quick-Reference

| Feature | What It Measures | Typical Use |
|---|---|---|
| `rsi_14` | Momentum oscillator, 30/70 levels | Entry filter (don't buy overbought) |
| `macd_hist` | Momentum direction and acceleration | Primary directional signal |
| `atr_14` | Absolute price volatility | Stop-loss sizing (NOT normalised) |
| `bb_pct_b` | Price position within Bollinger Bands | Entry timing, mean reversion signal |
| `bb_width` | Volatility regime (contraction → expansion) | Breakout detection |
| `adx_14` | Trend strength | Regime filter (trend vs range) |
| `ema_20_50_spread` | Medium-term trend direction | Regime confirmation |
| `price_vs_ema200` | Long-term trend position | Bull/bear regime |
| `volume_zscore` | Unusual volume relative to recent history | Conviction filter for entries |
| `obv_signal` | Volume-price divergence | Trend confirmation |
| `return_1` to `return_126` | Price momentum at multiple horizons | Cross-sectional momentum ranking |
| `btc_return_lag1` | BTC's recent momentum (for alts) | Leading indicator for alt entries |

---

## Failure Modes This Layer Prevents

| Failure | Prevention |
|---|---|
| Look-ahead bias from same-bar feature/signal alignment | `shift(1)` applied to all features universally |
| Look-ahead bias from global normalisation | `expanding()` mean/std, never global |
| NaN propagation from indicator warmup | `dropna()` enforced as final step |
| Strategy divergence between backtest and live | Single shared `features.py` called identically |
| Overfitting from too many uninformative features | IC analysis gates which features enter the strategy |
| ATR-based stops invalidated by normalisation | `atr_14` is intentionally NOT normalised |

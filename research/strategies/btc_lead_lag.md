# Strategy 1: BTC Lead-Lag Features

## What it is

BTC processes market information faster than any altcoin. When BTC moves, altcoins follow with a
delay of minutes to hours — a structural result of retail attention fragmentation and BTC being the
reserve asset of the entire crypto market. Encoding BTC's recent return history as features in an
altcoin XGBoost model lets the model learn:

1. The magnitude and direction of BTC's recent move
2. How correlated this specific altcoin is with BTC (high correlation = follow BTC closely)
3. The rolling beta (how much the altcoin amplifies BTC moves)
4. Asymmetry: BTC declines hit altcoins harder than BTC rises (Demir et al., 2021)

This is pure OHLCV. No external data required. It's a cross-asset extension of the feature
pipeline already partially built — `compute_cross_asset_features` in `features.py` already
computes lagged log returns for ETH and SOL, but doesn't yet include BTC return lags or
rolling correlation/beta.

---

## How to implement in this codebase

### New features to add to `bot/data/features.py`

The existing `compute_cross_asset_features(btc_df, other_dfs)` function already injects ETH/SOL
lag1 and lag2 log returns into the BTC feature DataFrame. A parallel function is needed that works
the other direction: injects BTC features into *altcoin* feature DataFrames.

Add a new function `compute_btc_context_features`:

```python
import numpy as np
import pandas as pd

def compute_btc_context_features(
    altcoin_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    corr_window: int = 42,   # ~7 days of 4H bars
    beta_window: int = 42,
) -> pd.DataFrame:
    """
    Inject BTC context features into an altcoin feature DataFrame.

    Features added (all shifted 1 bar — no look-ahead):
      btc_ret_lag1        : BTC log return, 1 bar ago
      btc_ret_lag2        : BTC log return, 2 bars ago
      btc_ret_6bar        : BTC 24h cumulative log return (sum of last 6 bars)
      btc_ret_42bar       : BTC 7d cumulative log return (sum of last 42 bars)
      btc_alt_corr        : Rolling 42-bar Pearson correlation of altcoin vs BTC log returns
      alt_beta            : Rolling 42-bar beta (altcoin return / BTC return covariance)

    Args:
        altcoin_df: Feature DataFrame for the altcoin (already processed by compute_features).
                    Must have a DatetimeIndex and a 'close' column.
        btc_df:     Raw OHLCV DataFrame for BTC/USD.
                    Must have a DatetimeIndex and a 'close' column.
        corr_window: Rolling window in bars for correlation and beta.
        beta_window: Rolling window in bars for beta (can differ from corr_window).

    Returns:
        New DataFrame with BTC context columns appended. Columns are shifted 1 bar.
    """
    out = altcoin_df.copy()

    # Normalize BTC column names
    btc = btc_df.copy()
    btc.columns = btc.columns.str.lower()

    # BTC log returns aligned to altcoin index
    btc_log_ret = np.log(btc["close"] / btc["close"].shift(1)).reindex(out.index)
    alt_log_ret = np.log(out["close"] / out["close"].shift(1))

    # --- Lagged BTC returns ---
    out["btc_ret_lag1"]  = btc_log_ret.shift(1)   # shift again: already 1-bar lag
    out["btc_ret_lag2"]  = btc_log_ret.shift(2)

    # --- Cumulative BTC returns over longer horizons ---
    out["btc_ret_6bar"]  = btc_log_ret.rolling(6).sum().shift(1)
    out["btc_ret_42bar"] = btc_log_ret.rolling(42).sum().shift(1)

    # --- Rolling correlation: altcoin return vs BTC return ---
    combined = pd.concat([alt_log_ret, btc_log_ret], axis=1)
    combined.columns = ["alt", "btc"]
    out["btc_alt_corr"] = (
        combined["alt"].rolling(corr_window).corr(combined["btc"]).shift(1)
    )

    # --- Rolling beta: cov(alt, btc) / var(btc) ---
    def rolling_beta(alt_ret: pd.Series, btc_ret: pd.Series, window: int) -> pd.Series:
        cov = alt_ret.rolling(window).cov(btc_ret)
        var = btc_ret.rolling(window).var()
        return (cov / (var + 1e-12)).shift(1)

    out["alt_beta"] = rolling_beta(alt_log_ret, btc_log_ret, beta_window)

    return out
```

### Integration point

In `main.py` or wherever per-coin features are assembled before model inference, call this after
`compute_features()`:

```python
# Existing: compute per-coin features
features = compute_features(coin_df)

# New: inject BTC context if this is an altcoin
if pair != "BTC/USD":
    features = compute_btc_context_features(features, btc_df)
```

The BTC DataFrame (`btc_df`) is already fetched by `live_fetcher.py` — it's the reference asset
for the 4H model. No extra API calls needed.

### Column names in the XGBoost feature matrix

After integration, the model sees 6 new columns per altcoin:
- `btc_ret_lag1`
- `btc_ret_lag2`
- `btc_ret_6bar`
- `btc_ret_42bar`
- `btc_alt_corr`
- `alt_beta`

These need to be present in the training data when the model was trained. If adding to existing
models, you must retrain — XGBoost will error if inference columns differ from training columns.
Treat this as a feature set expansion that requires a model retrain.

---

## How to check for correctness

### Sanity checks on feature values

```python
# After computing features for ETH with BTC context:
f = compute_btc_context_features(eth_features, btc_raw_df)

# 1. Lag alignment check: btc_ret_lag1 at row N should equal BTC log return at row N-2
btc_logret = np.log(btc_raw_df["close"] / btc_raw_df["close"].shift(1))
aligned = btc_logret.reindex(f.index)
mismatch = (f["btc_ret_lag1"] - aligned.shift(2)).abs().max()
assert mismatch < 1e-10, f"Lag alignment off by {mismatch}"

# 2. Correlation bounds: must be in [-1, 1]
assert f["btc_alt_corr"].dropna().between(-1, 1).all()

# 3. Beta sign: most altcoins have positive beta to BTC; flag negatives
neg_beta_pct = (f["alt_beta"].dropna() < 0).mean()
print(f"Negative beta fraction: {neg_beta_pct:.2%}")  # Should be < 20%

# 4. No look-ahead: btc_ret_lag1 at bar T should NOT know bar T's close
# Check by verifying the shift was applied
last_btc_ret = np.log(btc_raw_df["close"].iloc[-1] / btc_raw_df["close"].iloc[-2])
assert f["btc_ret_lag1"].iloc[-1] != last_btc_ret  # must be shifted
```

### Backtest feature importance check

After retraining the model with BTC context features:
```python
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model("models/xgb_btc_4h.pkl")  # retrained model

importance = dict(zip(feature_names, model.feature_importances_))
btc_features = {k: v for k, v in importance.items() if k.startswith("btc_")}
print(sorted(btc_features.items(), key=lambda x: -x[1]))
# btc_ret_lag1 and btc_alt_corr should consistently appear in top 10
```

### Walk-forward predictive validity

A simple correlation test — does `btc_ret_lag1` have a statistically significant correlation
with next-bar altcoin return?

```python
f["next_bar_ret"] = np.log(f["close"].shift(-1) / f["close"])  # forward return (label only)
corr = f[["btc_ret_lag1", "btc_ret_6bar", "btc_alt_corr", "alt_beta"]].corrwith(
    f["next_bar_ret"]
)
print(corr)
# Expect btc_ret_lag1 correlation ~0.05-0.20 for liquid altcoins
# btc_alt_corr × btc_ret_lag1 interaction is the key non-linear signal XGBoost captures
```

---

## Maximizing value

### Use interaction implicitly through XGBoost, not manually

The real edge is not `btc_ret_lag1` alone — it's `btc_ret_lag1 × btc_alt_corr`. When BTC moves up
*and* the altcoin is highly correlated with BTC, the altcoin signal is strong. When correlation is
low, the BTC move is irrelevant to this altcoin. XGBoost captures this automatically via tree
splits, so no manual feature interaction engineering is needed. Just provide both features.

### Regime asymmetry: BTC declines vs BTC rises

Demir et al. (2021) document that BTC declines affect altcoins more than BTC rises. Add a feature
that flags the sign:

```python
out["btc_ret_lag1_neg"] = out["btc_ret_lag1"].clip(upper=0)  # only negative BTC moves
out["btc_ret_lag1_pos"] = out["btc_ret_lag1"].clip(lower=0)  # only positive BTC moves
```

This lets XGBoost learn that large negative BTC returns have outsized downside impact on
altcoin returns.

### Window selection

The research finds BTC lead-lag works on 1-bar to ~6-bar horizons (4H to 24H). The 42-bar
(7-day) cumulative return captures the trend component that the momentum strategy already uses.
Don't add longer windows — they overlap with TSMOM features already in the model.

### Per-altcoin beta stratification

High-beta altcoins (DOGE, PEPE, BONK — beta > 1.5) amplify BTC moves dramatically. Low-beta
coins (stablecoins, wrapped tokens) barely react. `alt_beta` lets the model learn this per coin.
Don't compute one global BTC beta — compute it per coin, per rolling window.

---

## Common pitfalls

### Pitfall 1: Double-shifting

`compute_cross_asset_features` in `features.py` already shifts ETH/SOL returns by an extra lag.
If you call `compute_btc_context_features` *after* `compute_features` has already applied
`shift(1)` to all indicator columns, you must be careful:
- The BTC log return series is computed fresh from raw BTC OHLCV (not from the shifted feature
  DataFrame), so it's correctly lagged by 1 within the function via `.shift(1)`.
- Do NOT pass `btc_features` (already-shifted feature DataFrame) as the BTC input. Always pass
  the raw `btc_df` OHLCV.

### Pitfall 2: Non-stationarity of correlation

Rolling correlation is regime-dependent. In March 2020 and May 2021 crashes, crypto correlations
spiked toward 1.0. In altcoin-specific news events, correlation briefly collapses. A 42-bar
window is a reasonable medium; don't use very short windows (< 20 bars) which are too noisy.

### Pitfall 3: BTC features in the BTC model itself

If the model also trades BTC/USD, `btc_ret_lag1` becomes a self-referential feature — you're
predicting BTC from its own lagged return. This is fine (it's autocorrelation), but it's a
different signal than the lead-lag effect. Consider using different feature sets for BTC vs.
altcoin models, or at minimum noting this difference in interpretation.

### Pitfall 4: Overfitting on BTC-dominated periods

In high-correlation regimes (all coins moving with BTC), BTC features dominate the model and
individual coin signals become useless. The model may learn to just follow BTC and discard all
other features. Check SHAP values periodically — if `btc_ret_lag1` SHAP contribution exceeds
60% of total signal, the model has degenerated to "just follow BTC."

### Pitfall 5: Missing BTC data at startup

The live fetcher may not have a full 42-bar BTC buffer ready at bot startup. Add a warmup guard:

```python
if len(btc_df) < 50:
    logger.warning("BTC buffer too short for lead-lag features — skipping")
    return features  # return without BTC context
```

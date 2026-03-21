# Strategy 2: Cross-Sectional Rank Features

## What it is

Cross-sectional momentum ranks each coin by its recent return *relative to the rest of the
universe* at every bar. A coin with a `ret_7d_rank` of 0.95 is in the top 5% of performers over
the last 7 days across all 39 coins. This is fundamentally different from time-series momentum
(TSMOM), which looks at one coin's own history: cross-sectional rank answers "how is this coin
doing *compared to its peers right now?*"

The academic case (Fieberg et al., 2025 — CTREND factor) is that aggregating multiple technical
indicators via cross-sectional rank captures **relative strength**: the coins that are relatively
strongest tend to stay strongest for 1-4 weeks. The research shows this subsumes raw TSMOM and
achieves Sharpe ~1.8 with proper implementation.

In practice, you're adding 3-4 columns per coin per bar — percentile ranks of 7d, 14d, and 28d
returns across the full coin universe. XGBoost uses these to learn "this coin is outperforming —
buy" vs "this coin is underperforming the pack — skip".

---

## How to implement in this codebase

### Where to add this

This is a **universe-level** computation that cannot live inside the per-coin `compute_features`
function (which only sees one coin's OHLCV at a time). It needs to sit in the data pipeline where
all coin DataFrames are available simultaneously — likely in `live_fetcher.py` or a new
`universe_features.py` module.

### Step 1: Build the return matrix

```python
import numpy as np
import pandas as pd

def compute_cross_sectional_ranks(
    coin_dfs: dict[str, pd.DataFrame],
    lookbacks: list[int] = [42, 84, 168],   # 7d, 14d, 28d in 4H bars
) -> dict[str, pd.DataFrame]:
    """
    Compute cross-sectional return rank features for each coin.

    For each lookback period, ranks each coin's cumulative log return
    as a percentile (0–1) within the universe at each bar.

    Args:
        coin_dfs: Dict mapping pair symbol → raw OHLCV DataFrame.
                  All DataFrames must share the same DatetimeIndex.
        lookbacks: List of bar lookback windows. 42 bars = 7d at 4H.

    Returns:
        Same dict as input but with rank columns added to each DataFrame.
        Columns added per lookback: f"ret_{n}bar_rank", f"ret_{n}bar_zscore"
        All rank/zscore columns are shifted 1 bar (no look-ahead).
    """
    # Step 1: compute cumulative log returns for each coin at each lookback
    return_matrices = {}
    for n in lookbacks:
        # One column per coin, one row per bar
        ret_matrix = pd.DataFrame(
            {
                pair: np.log(df["close"] / df["close"].shift(n))
                for pair, df in coin_dfs.items()
                if not df.empty
            }
        )
        return_matrices[n] = ret_matrix

    # Step 2: compute percentile rank across coins at each timestamp
    rank_matrices = {}
    zscore_matrices = {}
    for n, ret_matrix in return_matrices.items():
        # rank(axis=1, pct=True): for each row (timestamp), rank across columns (coins)
        rank_matrices[n] = ret_matrix.rank(axis=1, pct=True)
        # z-score across coins at each timestamp
        mean = ret_matrix.mean(axis=1)
        std  = ret_matrix.std(axis=1)
        zscore_matrices[n] = ret_matrix.sub(mean, axis=0).div(std + 1e-10, axis=0)

    # Step 3: inject back into per-coin DataFrames with 1-bar shift
    result = {}
    for pair, df in coin_dfs.items():
        out = df.copy()
        for n in lookbacks:
            if pair in rank_matrices[n].columns:
                out[f"ret_{n}bar_rank"]   = rank_matrices[n][pair].shift(1)
                out[f"ret_{n}bar_zscore"] = zscore_matrices[n][pair].shift(1)
        result[pair] = out

    return result
```

### Step 2: Integration in the data pipeline

Call `compute_cross_sectional_ranks` **after** all per-coin `compute_features()` calls complete,
but **before** model inference:

```python
# In live_fetcher.py or main.py:

# 1. Fetch all coin OHLCV DataFrames
coin_dfs = {pair: fetcher.get_ohlcv(pair) for pair in UNIVERSE}

# 2. Per-coin technical features (existing)
feature_dfs = {pair: compute_features(df) for pair, df in coin_dfs.items()}

# 3. Cross-sectional rank features (new)
feature_dfs = compute_cross_sectional_ranks(feature_dfs)

# 4. BTC context features per altcoin (Strategy 1)
btc_raw = coin_dfs["BTC/USD"]
for pair in UNIVERSE:
    if pair != "BTC/USD":
        feature_dfs[pair] = compute_btc_context_features(feature_dfs[pair], btc_raw)

# 5. Run XGBoost inference per coin
for pair, features in feature_dfs.items():
    signal = model.predict_proba(features.iloc[-1:])
```

### Column names added

For lookbacks [42, 84, 168] (7d, 14d, 28d), each coin gets 6 new columns:
- `ret_42bar_rank`   — percentile rank of 7d return (0=worst, 1=best in universe)
- `ret_42bar_zscore` — z-score of 7d return vs. universe
- `ret_84bar_rank`
- `ret_84bar_zscore`
- `ret_168bar_rank`
- `ret_168bar_zscore`

These should be added to the training data and the model retrained. The feature names must be
identical in train and live.

---

## How to check for correctness

### Universe coverage check

At each timestamp, the sum of all coins' ranks for a given lookback should be ~uniform:

```python
# Verify ranks are correctly computed at a single timestamp
timestamp = feature_dfs["BTC/USD"].index[-2]  # second-to-last bar (last is current bar)

ranks_at_t = {
    pair: feature_dfs[pair].loc[timestamp, "ret_42bar_rank"]
    for pair in UNIVERSE
    if "ret_42bar_rank" in feature_dfs[pair].columns
}
ranks_series = pd.Series(ranks_at_t).dropna()

# Should have ~N coins; mean should be ~0.5; min ~0, max ~1
print(f"N coins ranked: {len(ranks_series)}")
print(f"Mean rank: {ranks_series.mean():.3f}")  # expect ~0.5
print(f"Min: {ranks_series.min():.3f}, Max: {ranks_series.max():.3f}")
assert ranks_series.between(0, 1).all(), "Rank out of [0, 1] bounds"
```

### No look-ahead check

```python
# The rank at bar T must be computed from returns ending at bar T-1 (due to shift(1))
# Verify by comparing manually

pair = "ETH/USD"
n = 42

# Manual: return from bar T-n-1 to bar T-1 (shifted)
expected_ret = np.log(
    coin_dfs[pair]["close"].iloc[-2] / coin_dfs[pair]["close"].iloc[-2 - n]
)
stored_rank = feature_dfs[pair]["ret_42bar_rank"].iloc[-1]

# Cross-check: what rank would that expected_ret get in the universe at that bar?
# (This is an indirect check — exact equality not guaranteed due to float precision)
print(f"ETH 7d return at T-1: {expected_ret:.4f}")
print(f"ETH 7d rank stored:   {stored_rank:.3f}")
# If ETH return is above median, rank should be > 0.5
```

### Predictive validity

Before retraining the model, do a quick correlation check:

```python
for pair in ["BTC/USD", "ETH/USD", "SOL/USD"]:
    df = feature_dfs[pair].copy()
    df["fwd_ret"] = np.log(df["close"].shift(-6) / df["close"])  # 24h forward return
    ic = df[["ret_42bar_rank", "ret_84bar_rank", "fwd_ret"]].corr()["fwd_ret"]
    print(f"{pair}: IC(ret_42bar_rank, fwd_ret_24h) = {ic['ret_42bar_rank']:.3f}")
    # Expect positive IC: ~0.05-0.15 for momentum
```

---

## Maximizing value

### Don't replace TSMOM — augment it

`ret_42bar_rank` is correlated with `ret_42bar` (raw return), but not identical. The rank
normalizes away level effects: a +5% return looks different when the universe median is +10% vs.
+0%. Including both raw returns (already in the model via EMA slope, MACD) and cross-sectional
ranks makes the feature set richer without perfect redundancy.

### Use rank for coin selection, z-score for signal strength

- Rank = which coins to trade (coins ranked above 0.7 are candidates for the long book)
- Z-score = how strong the signal is (z-score > 1.5 is a high-conviction momentum signal)

XGBoost will find this naturally, but you can also hard-code a pre-filter: only run model
inference on coins where `ret_42bar_rank > 0.6`, saving compute.

### Spread between top and bottom: universe dispersion feature

Add one more universe-level feature that every model receives — the return spread between the
top and bottom tercile. When this spread is high, cross-sectional momentum is "hot"; when low,
the universe is correlated and CS signals are noisy:

```python
def compute_universe_spread(coin_dfs, n=42):
    ret_matrix = pd.DataFrame({
        pair: np.log(df["close"] / df["close"].shift(n))
        for pair, df in coin_dfs.items()
    })
    top_tercile_mean = ret_matrix.apply(
        lambda row: row[row >= row.quantile(0.67)].mean(), axis=1
    )
    bot_tercile_mean = ret_matrix.apply(
        lambda row: row[row <= row.quantile(0.33)].mean(), axis=1
    )
    return (top_tercile_mean - bot_tercile_mean).shift(1).rename("universe_spread_42bar")
```

Inject this single series into every coin's feature DataFrame. It's a regime indicator —
when spread is high, CS momentum works better; when near zero, switch to other signals.

---

## Common pitfalls

### Pitfall 1: Incomplete bar at computation time

At the live 4H bar close, some exchange feeds have a processing delay. If BTC's bar has closed
but PEPE's bar is still 1 second behind, the return matrix at that timestamp has a missing value
for PEPE. The rank then gets computed on N-1 coins. Use `min_count` or `dropna(axis=1, how='any')`
before ranking to ensure consistent universe size:

```python
ret_matrix.rank(axis=1, pct=True)  # NaN propagates naturally — no special handling needed
# but log that the universe size shrank:
n_coins = ret_matrix.notna().sum(axis=1)
if n_coins.min() < len(coin_dfs) * 0.8:
    logger.warning("Universe shrinkage: some coins missing data at rank computation")
```

### Pitfall 2: The universe must be identical in train and live

If you train the model with 39 coins and run live with 35 coins (4 were delisted or removed),
rank percentiles shift: a coin that was 70th percentile in the 39-coin universe might be 80th
percentile in the 35-coin universe. This creates train/live distribution shift. Keep a fixed
`UNIVERSE` list in config and use it in both training and live.

### Pitfall 3: Rank features are strongly autocorrelated

A coin that ranked 90th percentile yesterday will almost certainly rank ~90th percentile today.
XGBoost handles autocorrelation fine, but walk-forward validation will inflate apparent accuracy
if train/test windows overlap in rank values. Use a minimum 2-week gap (embargo) between train
end and test start when validating these features.

### Pitfall 4: Small universe = quantization

With 39 coins, `pct=True` rank produces values like 0.026, 0.051, 0.077, ... (1/39 increments).
There's no continuous signal, just 39 discrete levels. This is fine for XGBoost tree splits but
be aware that a "rank of 0.90" means "36th out of 39 coins" — high granularity claims are
statistically weak. The z-score column is more continuous and often more informative.

### Pitfall 5: Look-ahead in z-score normalization

When computing the cross-sectional mean and std across coins, the normalization itself uses
future information if done naively on the full sample. In the code above, the z-score is computed
at each timestamp using only that timestamp's cross-sectional values (axis=1), which is correct —
it's comparing coins to each other at the same time, not normalizing across time.

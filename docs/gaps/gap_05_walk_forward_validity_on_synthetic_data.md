# Gap 05: Walk-Forward Optimization Validity When Live Indicators Differ from Backtest

## Why This Is a Gap
Walk-forward optimization (when implemented — see Issue 08) trains parameters on historical Binance OHLCV data. But live trading uses synthetic flat candles from Roostoo (H=L=O=C=LastPrice). The ATR, ADX, and OBV values computed from these two data sources are fundamentally different.

## The Core Problem
If I optimize ATR multiplier = 2.0 using historical OHLCV data where ATR ≈ $1,500, but in live trading ATR ≈ $0 (synthetic flat candles), the stop levels will be completely wrong regardless of the optimized multiplier.

## What I Need to Know

1. **Can walk-forward optimization be done on close-only data?**
   - If I use only close prices (no high/low) during backtesting, do the results transfer to live trading?
   - This means replacing ATR with close-to-close volatility in backtesting too.

2. **Is there a way to simulate synthetic candle behavior during backtesting?**
   - Downsample OHLCV bars to simulate "what ATR would look like if all bars were flat"
   - This would give pessimistic ATR estimates but would match live behavior

3. **Should optimization focus on close-only indicators (MACD, RSI, EMA crossovers) exclusively?**
   - These transfer perfectly from backtest to live since they only use close prices
   - ATR-based stops could use a fixed-dollar fallback calibrated from historical volatility

4. **What is the recommended approach in the academic literature for optimizing live systems with limited market data?**

## Priority
**High** — all optimized parameters are potentially invalid in live trading due to this mismatch.

---

## Research Findings (Context7, 2026-03-12)

### TimeSeriesSplit Walk-Forward Structure (sklearn)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(df):
    # Expanding window: train sets grow, test always after train
    # Fold 1: train[0:20%], test[20%:40%]
    # Fold 2: train[0:40%], test[40%:60%]
    # Fold 3: train[0:60%], test[60%:80%]
    # etc.
    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx]
```

Key property: the test set always comes **after** the train set, no data leakage.

### Close-Only Indicator Transfer Analysis

| Indicator | Uses H/L? | Transfers to live? | Action |
|-----------|-----------|-------------------|--------|
| RSI_14 | Close only | ✅ Yes — perfect transfer | Keep as-is |
| MACD_12_26_9 | Close only | ✅ Yes — perfect transfer | Keep as-is |
| EMA crossover | Close only | ✅ Yes — perfect transfer | Keep as-is |
| ATR_14 | High-Low-Close | ❌ Near-zero on synthetic | Replace with close-to-close vol |
| ADX_14 | High-Low-Close | ❌ Near-zero on synthetic | Replace with EMA-slope or fixed threshold |
| OBV | Close + Volume | ❌ Volume=0 on synthetic | Remove from live feature set |
| Bollinger Bands | Close only (std) | ✅ Yes — close-based std | Keep as-is |

### Recommended Resolution

**Use close-to-close volatility for ATR in BOTH backtest and live trading.**

This is the critical insight: the fix is not to use different ATR computations
in backtest vs live — it is to use the **same** close-to-close formula in both.

When the backtest is run on Binance historical data (which HAS high/low), we
deliberately ignore high/low and use only close prices. This produces slightly
noisier ATR estimates in backtest, but the values will be calibrated correctly
for live trading.

```python
# In compute_features() — change ONCE and it fixes both backtest and live:
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Close-to-close ATR proxy (works on both OHLCV and synthetic candles)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["ATR_14"] = log_ret.rolling(14).std() * df["close"] * 1.25

    # Close-only indicators — transfer perfectly
    df.ta.rsi(length=14, append=True)      # RSI_14
    df.ta.macd(fast=12, slow=26, signal=9, append=True)  # MACD_12_26_9
    df.ta.bbands(length=20, append=True)   # BBL_20_2.0, BBU_20_2.0

    # Remove volume-dependent indicators for live compatibility
    # df.ta.obv(append=True)  # DISABLED: volume=0 in synthetic candles
    # df.ta.adx(append=True)  # DISABLED: H=L on synthetic candles
```

### ADX Replacement

ADX measures trend strength from high-low directional movement. With synthetic
flat candles, ADX ≈ 0 always. Replacement options:
1. **EMA slope**: `ema_slope = df["close"].ewm(span=14).mean().diff(5)` — measures momentum direction
2. **RSI regime**: RSI > 60 = trending up, RSI < 40 = trending down, 40-60 = ranging
3. **Remove ADX entirely**: The regime detector already uses EMA crossovers (Gap 06)

### Walk-Forward Parameter Robustness Check

After optimization, perform a robustness check on close-only features:
```python
# Verify that optimized params produce consistent Sharpe across all 5 folds
# Reject parameter sets with high fold-to-fold variance (overfit indicator)
fold_sharpes = [sharpe_fold_1, sharpe_fold_2, ..., sharpe_fold_5]
if np.std(fold_sharpes) > 0.5:
    # Parameters are overfit — they only work in specific market conditions
    # Consider wider parameter priors or simpler strategy
    pass
```

This is a standard robustness check: optimized parameters should produce
positive Sharpe in **all** folds, not just on average.

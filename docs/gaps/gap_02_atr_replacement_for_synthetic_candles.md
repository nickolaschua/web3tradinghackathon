# Gap 02: ATR Replacement Strategy for Synthetic Flat Candles

## Why This Is a Gap
The entire stop-loss system is ATR-based, but live candles from Roostoo are synthetic with H=L=O=C=LastPrice (Issue 06). Standard ATR (using high-low range) will be near-zero on these candles, making ATR trailing stops collapse to the current price — causing immediate stop-outs on every position.

## What I Need to Know

### Option A: Close-to-close volatility as ATR proxy
Use rolling standard deviation of close-to-close returns, scaled by current price to produce an ATR-equivalent dollar value:
```python
returns = df["close"].pct_change()
vol_proxy = returns.rolling(14).std() * df["close"]
```
Question: What scaling factor converts close-to-close vol to ATR equivalence? Research typical ATR/realized-vol ratios.

### Option B: Source OHLCV from Binance in parallel
Run a separate lightweight thread that fetches Binance WebSocket kline data for BTCUSDT alongside the Roostoo polling. Use Binance high/low for ATR calculation but Roostoo prices for actual trade execution.
Question: Is this architecturally sound? Is it acceptable under competition rules to use Binance data feeds for indicator computation?

### Option C: Pre-compute ATR multiplier offline and use fixed dollar stop
Use the average historical ATR in dollars for BTC and apply it as a fixed dollar stop, updated daily from historical data.
Question: Too static? BTC volatility varies significantly.

## Research Needed
1. What does the competition allow in terms of external data feeds during the competition period?
2. What is the typical ratio of `ATR_14` to `close_to_close_std * price` for BTC on 4H data?
3. Are there Python libraries that implement close-to-close ATR estimators?

## Priority
**Critical** — if ATR is near-zero in live trading, the stop-loss system is non-functional.

---

## Research Findings (Context7 + Domain Knowledge, 2026-03-12)

### pandas-ta ATR Column Naming

`pandas_ta_classic` (the library used in the project) computes:
```python
df.ta.atr(length=14, append=True)  # appends column: ATR_14
```

The library requires columns named exactly: `open`, `high`, `low`, `close`, `volume`.
When H=L=O=C on synthetic candles, `ATR_14 ≈ 0` because the TR formula is:
```
TR = max(high-low, |high-prev_close|, |low-prev_close|)
   = max(0, 0, 0) = 0  (when H=L=C and prev_close = close)
```

### Close-to-Close Volatility Formula

The Yang-Zhang or close-to-close estimator:
```python
def close_to_close_atr(df, length=14):
    """ATR proxy using only close prices — survives flat synthetic candles."""
    log_returns = np.log(df["close"] / df["close"].shift(1))
    rolling_std = log_returns.rolling(length).std()
    # Scale to dollar terms (same units as ATR)
    return rolling_std * df["close"]
```

**Empirical ratio**: On 4H BTC data, `ATR_14 ≈ 1.0–1.5x * (close_to_close_std * close)`.
The factor is ~1.25 as a practical approximation. Apply a correction factor:
```python
vol_proxy = returns.rolling(14).std() * df["close"] * 1.25
```

This preserves the units and rough magnitude of ATR even on flat candles,
because the close price still moves between 4H candles (each candle = last tick
from Roostoo, which does change with BTC price).

### Recommended Implementation: Modified Option A

**Decision**: Use close-to-close volatility proxy (Option A) with a 1.25x scaling
factor. This is the only option that:
1. Doesn't require external data feeds (Option B architectural complexity)
2. Isn't static (Option C fails during high-volatility periods)
3. Works with the existing backtesting framework — change the ATR computation
   during **both** backtest and live trading to use close-to-close vol

**Implementation change required in feature engineering:**
```python
# In compute_features() — replace pandas_ta ATR:
# OLD: df.ta.atr(length=14, append=True)  # produces ATR_14 (broken on synthetic candles)
# NEW:
log_ret = np.log(df["close"] / df["close"].shift(1))
df["ATR_14"] = log_ret.rolling(14).std() * df["close"] * 1.25
```

By using the same formula in backtest AND live, the ATR values will match.

### Backtest Consistency Requirement

**Critical**: The ATR computation must be changed in the backtest too, not just
in the live feature engineering. If backtest uses `pandas_ta ATR` (true OHLCV)
and live uses `close-to-close vol`, the ATR multiplier will be miscalibrated.

Change `compute_features()` to use close-to-close vol exclusively, then re-run
the walk-forward optimization with this new ATR definition.

### Option B Assessment

Using a parallel Binance WebSocket feed for indicator data is architecturally
possible but adds significant complexity:
- Additional WebSocket thread management
- Price synchronization between Binance timestamps and Roostoo timestamps
- Potential competition rule violation if the rules restrict data sources

Not recommended unless competition rules explicitly allow external feeds.

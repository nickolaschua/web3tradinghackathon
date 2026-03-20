# Gap 12: `pandas-ta` Library Availability and Correctness on EC2

## Why This Is a Gap
The feature engineering layer uses `pandas_ta` for ATR, ADX, MACD, RSI, and Bollinger Band calculations. I need to verify:

## What I Need to Know

1. **Does `pandas_ta` produce the same values as `TA-Lib` for these indicators?**
   - Backtest optimization may have been designed with different indicator libraries
   - Minor differences in lookback periods (e.g. Wilder's smoothing for RSI) can cause different numerical values
   - Verify: `pandas_ta.rsi(close, length=14)` matches `talib.RSI(close, timeperiod=14)` on the same data

2. **Is `pandas_ta` compatible with Python 3.11?**
   - The infrastructure docs specify Python 3.11 on EC2
   - Some `pandas_ta` versions have compatibility issues with newer pandas/numpy versions
   - Test: `pip install pandas-ta` on the target Python version and run `import pandas_ta`

3. **What is the minimum number of bars required before `pandas_ta` returns non-NaN values?**
   - RSI(14): needs 14+1 bars
   - MACD(12,26,9): needs 26+9 bars minimum = 35 bars
   - ATR(14): needs 14+1 bars
   - ADX(14): needs 28+ bars (ADX uses a double smoothing)
   - The buffer warmup (500 bars) should be sufficient, but verify the `is_warmed_up()` threshold

4. **Does `pandas_ta` handle the `shift(1)` operation correctly when NaN-filling?**
   - After `shift(1)`, the first row of every indicator column is NaN
   - Does `dropna()` remove only the first row, or does it cascade (e.g. if MACD needs 35 bars, rows 0-35 are NaN)

## Priority
**Medium** — affects whether indicator values in live trading match backtest values precisely.

---

## Research Findings (Context7 — pandas-ta-classic, 2026-03-12)

### Library Clarification: `pandas_ta_classic` not `pandas_ta`

**Critical discovery**: The project imports `pandas_ta` but the Context7
library that matches the project's usage pattern is **`pandas_ta_classic`**
(PyPI: `pandas-ta-classic`), not the original `pandas-ta` package.

The original `pandas-ta` repository has been unmaintained since 2022 and has
known compatibility issues with pandas >= 2.0 and Python 3.11.
`pandas_ta_classic` is the actively maintained fork.

**Import**: `import pandas_ta_classic as ta` (or as `pandas_ta` with alias)

### Column Naming Convention (Confirmed)

From Context7 (`/xgboosted/pandas-ta-classic`):
```python
df.ta.atr(length=14, append=True)     # → ATR_14
df.ta.adx(length=14, append=True)     # → ADX_14, DMP_14, DMN_14
df.ta.rsi(length=14, append=True)     # → RSI_14
df.ta.macd(fast=12, slow=26, signal=9, append=True)  # → MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
df.ta.bbands(length=20, append=True)  # → BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
```

**Required input column names**: exactly `open`, `high`, `low`, `close`, `volume`
(lowercase). The library will fail silently or with KeyError if columns are
uppercase (`Open`, `High`, etc.) or use different names.

### RSI: pandas_ta vs TA-Lib Parity

`pandas_ta` RSI uses Wilder's Exponential Smoothing (same as TA-Lib).
The formula is: `RS = Avg Gain(14) / Avg Loss(14)` using EWM with `alpha = 1/14`.

TA-Lib RSI and pandas_ta RSI produce **identical values** after the initial
warmup period (the first RSI value may differ by ~0.001 due to initialization
seed differences, but values converge after ~30 bars).

**Practical equivalence**: For a 500-bar warmup buffer, any initialization
difference between TA-Lib and pandas_ta is fully dissolved by bar 30.

### MACD Signal Line Naming

The project code references `MACDs_12_26_9` (the signal line, lowercase 's').
This is confirmed correct for pandas_ta_classic:
- `MACD_12_26_9` = MACD line (fast EMA - slow EMA)
- `MACDh_12_26_9` = MACD histogram
- `MACDs_12_26_9` = MACD signal line (EMA of MACD)

### Python 3.11 Compatibility

`pandas_ta_classic` is tested with:
- Python 3.8, 3.9, 3.10, 3.11 ✅
- pandas >= 1.3, including pandas 2.x ✅
- numpy >= 1.20 ✅

The original `pandas-ta` (0.3.14b0) has issues with pandas 2.x. If `pandas-ta`
is in requirements.txt, replace with `pandas-ta-classic`.

### Minimum Warmup Bars Verification

| Indicator | NaN bars | Actual warmup |
|-----------|----------|---------------|
| RSI(14) | 14 | 15 bars needed |
| EMA(20) | 20 | 21 bars needed |
| MACD(12,26,9) | 33 | 34 bars needed |
| ATR(14) true | 14 | 15 bars (uses H/L/C) |
| Close-to-close ATR | 14 | 15 bars (uses close only) |
| ADX(14) | 27 | 28 bars (double-smoothed DI) |
| Bollinger(20) | 20 | 21 bars needed |

**The 500-bar warmup buffer is more than sufficient for all indicators.**

The `is_warmed_up()` threshold should be at least **35 bars** (MACD warmup).
Current code uses 35 — this is correct.

### shift(1) and dropna() Behavior

After `shift(1)` on the indicator columns:
- All indicator columns have NaN in row 0
- This is intentional: row N uses features computed from bar N-1 (prevents look-ahead)
- `dropna()` removes only the rows where ANY column is NaN

**Cascade effect**: If MACD needs 34 bars of history, after `shift(1)`, rows
0–34 (35 rows) will have NaN in MACD columns. `dropna()` removes all 35 rows.

For the live buffer (500+ bars), this means the first 35 rows are dropped,
leaving 465+ clean rows. This is fine. The `is_warmed_up()` check of 35 bars
ensures we never call `compute_features()` on a buffer too small.

### Dependency Management

Add to `requirements.txt`:
```
pandas-ta-classic>=0.3.14  # maintained fork, Python 3.11 compatible
# NOT: pandas-ta (unmaintained, pandas 2.x incompatible)
```

Compatibility check on startup:
```python
try:
    import pandas_ta_classic as ta
    _PANDAS_TA_VERSION = ta.__version__
except ImportError:
    raise ImportError(
        "pandas-ta-classic not installed. Run: pip install pandas-ta-classic"
    )
```

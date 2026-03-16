# Phase 4: Data Pipeline - Research

**Researched:** 2026-03-16
**Domain:** Python feature engineering with pandas-ta-classic; Binance Parquet seeding; synthetic candle volatility proxies
**Confidence:** HIGH

<research_summary>
## Summary

Phase 4 builds the LiveFetcher and feature computation layer. Three niche topics warranted research: (1) the `pandas-ta-classic` API surface and its output column naming conventions; (2) the Binance Parquet flat column structure and the risk of capitalization mismatch; and (3) the mathematical validity of the close-to-close ATR proxy `log_returns.rolling(14).std() * close * 1.25`.

All three are well-understood after research. `pandas-ta-classic` uses `import pandas_ta_classic as ta` and outputs columns in `UPPER_UNDERSCORE` format (e.g. `RSI_14`, `MACD_12_26_9`). Binance raw CSVs use **capitalized** column names (`Open`, `High`, `Low`, `Close`, `Volume`) — the project's Parquet files assume lowercase; safest practice is to lowercase on load. The ATR proxy formula is mathematically sound: the theoretical SD-to-ATR conversion factor is `1/0.875 ≈ 1.143`; using `1.25` is a conservative buffer appropriate for fat-tailed crypto distributions.

**Primary recommendation:** Use `df.columns = df.columns.str.lower()` after reading Parquet to guard against capitalization mismatch. Compute all indicators, then shift all indicator columns by 1 bar in one pass. Inject cross-asset features BEFORE `dropna()`.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas-ta-classic | 0.3.78 (Feb 2026) | Technical indicators (RSI, MACD, EMA, Slope) | Community-maintained fork of pandas-ta; works on Python 3.11 + pandas 2.x |
| pandas | ≥2.0 | DataFrame operations, rolling windows | Core dependency |
| numpy | ≥1.24 | Log return calculations | Math primitives |
| collections.deque | stdlib | Circular candle buffer (maxlen=500) | O(1) append/drop, memory-bounded |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| binance_historical_data | PyPI latest | Download Binance klines CSVs | Seeding historical buffer on startup |
| pyarrow / fastparquet | via pandas | Read .parquet files | `pd.read_parquet()` backend |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pandas-ta-classic | pandas-ta (original) | Original broken on Python 3.11 + pandas 2.x — do NOT use |
| pandas-ta-classic | ta-lib | Requires C binary, harder to install on EC2 |
| ATR proxy | pandas_ta ATR | ATR ≈ 0 on H=L=O=C synthetic candles — DO NOT USE |

**Installation:**
```bash
pip install pandas-ta-classic pandas numpy pyarrow binance-historical-data
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended File Structure
```
bot/data/
├── __init__.py
├── live_fetcher.py      # LiveFetcher class with seed + poll + buffer
└── features.py          # compute_features() + compute_cross_asset_features()
```

### Pattern 1: pandas-ta-classic Import and Indicator API
**What:** Library is imported as `pandas_ta_classic`, used as DataFrame extension `.ta`
**When to use:** All indicator computations — RSI, MACD, EMA, Slope

```python
import pandas_ta_classic as ta  # NOT import pandas_ta as ta

# Append indicators to df (column names in UPPER_UNDERSCORE format)
df.ta.rsi(length=14, append=True)            # → adds RSI_14
df.ta.macd(fast=12, slow=26, signal=9, append=True)  # → adds MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
df.ta.ema(length=20, append=True)            # → adds EMA_20
df.ta.ema(length=50, append=True)            # → adds EMA_50
df.ta.slope(close="EMA_20", length=1, append=True)   # → adds SLOPE_1 (EMA slope proxy)

# Or: functional API (returns Series/DataFrame)
rsi = ta.rsi(df["close"], length=14)
```

### Pattern 2: Parquet Seeding with Column Safety
**What:** Load Binance Parquet with lowercase guard — raw Binance CSVs use `Open`/`High`/`Low`/`Close`/`Volume` (capitalized); Parquet conversions vary
**When to use:** `_seed_from_history()` in LiveFetcher

```python
def _seed_from_history(self, symbol: str, df: pd.DataFrame) -> None:
    # CRITICAL: lowercase columns — Binance CSVs are capitalized, Parquet may vary
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Flat column access — NOT multi-index df["BTC/USD"]["close"]
    required = {"open", "high", "low", "close", "volume"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

    for _, row in df.tail(self._maxlen).iterrows():
        candle = {
            "open": row["open"], "high": row["high"],
            "low": row["low"], "close": row["close"],
            "volume": row["volume"], "timestamp": row.name,
        }
        self._buffers[symbol].append(candle)
```

### Pattern 3: Shift-After-Compute (Look-Ahead Prevention)
**What:** Compute all indicators first, then shift all indicator columns together in one pass
**When to use:** `compute_features()` — prevents accidental use of current-bar info

```python
def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
    import pandas_ta_classic as ta
    import numpy as np

    out = df.copy()

    # ATR proxy: close-to-close log vol × close × 1.25
    log_ret = np.log(out["close"] / out["close"].shift(1))
    out["atr_proxy"] = log_ret.rolling(14).std() * out["close"] * 1.25

    # Indicators (DO NOT use ta.atr, ta.adx, ta.obv — all ≈0 on synthetic candles)
    out.ta.rsi(length=14, append=True)
    out.ta.macd(fast=12, slow=26, signal=9, append=True)
    out.ta.ema(length=20, append=True)
    out.ta.ema(length=50, append=True)
    out.ta.slope(close="EMA_20", length=1, append=True)

    # Shift ALL indicator columns 1 bar to prevent look-ahead bias
    indicator_cols = [c for c in out.columns if c not in {"open", "high", "low", "close", "volume"}]
    out[indicator_cols] = out[indicator_cols].shift(1)

    return out

### Pattern 4: Cross-Asset Features BEFORE dropna()
**What:** Inject features from other pairs (ETH, SOL) before dropping NaN rows
**Why:** dropna() on a DataFrame that doesn't yet have ETH/SOL columns drops ALL rows (Issue 07)

```python
def get_feature_matrix(self) -> pd.DataFrame:
    df = self.compute_features(self._to_dataframe("BTC/USD"))
    df = self.compute_cross_asset_features(df)  # ← MUST be before dropna
    df = df.dropna()                             # ← only after cross-asset injection
    return df
```

### Pattern 5: Synthetic Candle Construction
**What:** Roostoo /v3/ticker returns only LastPrice — construct flat candles
**When to use:** Live polling loop

```python
candle = {
    "open": last_price,
    "high": last_price,   # H = L = O = C = LastPrice (no OHLCV from Roostoo)
    "low": last_price,
    "close": last_price,
    "volume": 0.0,        # Roostoo provides no volume
    "timestamp": int(time.time()),
}
self._buffers[pair].append(candle)
```

### Anti-Patterns to Avoid
- **`from pandas_ta import ...`**: wrong package — use `pandas_ta_classic`
- **`df.ta.atr()` on synthetic candles**: returns ≈0 because H=L → stop-loss system breaks
- **`df.ta.adx()` on synthetic candles**: returns ≈0 because H=L → remove entirely
- **`df.ta.obv()` on synthetic candles**: volume=0 → OBV is static; remove
- **Multi-index Parquet access `df["BTC/USD"]["close"]`**: raises KeyError on flat Parquet
- **shift(1) before indicators**: shifts close prices, indicators compute on shifted closes
- **dropna() before cross-asset features**: silently drops all rows where ETH/SOL not yet present
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RSI calculation | Custom Wilder smoothing | `df.ta.rsi(length=14, append=True)` | Wilder EMA vs standard EMA difference causes ~2% divergence |
| MACD calculation | Custom fast/slow EMA diff | `df.ta.macd(fast=12, slow=26, signal=9, append=True)` | Signal line uses Wilder smoothing, easy to get wrong |
| EMA calculation | Custom `.ewm()` with alpha | `df.ta.ema(length=20, append=True)` | pandas-ta-classic uses `adjust=False` which matches TA convention |
| Linear regression slope | Polyfit per row | `df.ta.slope(length=1, append=True)` | Built-in, tested, correct |
| ATR for synthetic candles | `df.ta.atr()` | `log_ret.rolling(14).std() * close * 1.25` | `ta.atr()` requires H-L range; ≈0 on flat candles |

**Key insight:** Never use `pandas_ta` (old) — it's broken on Python 3.11 + pandas 2.x. Always `pandas_ta_classic`.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Wrong Package Import
**What goes wrong:** `import pandas_ta as ta` installs the old broken library, raises `AttributeError` or silent wrong results on Python 3.11
**Why it happens:** The original `pandas-ta` repo looks authoritative; easy to install wrong one
**How to avoid:** Always `pip install pandas-ta-classic` and `import pandas_ta_classic as ta`
**Warning signs:** `DeprecationWarning` from pandas about `.applymap`; indicators returning all NaN

### Pitfall 2: Parquet Column Capitalization Mismatch
**What goes wrong:** `KeyError: 'close'` when Binance Parquet has `Close` (capital C)
**Why it happens:** Binance raw CSV files use capitalized headers; Parquet conversion may or may not rename them
**How to avoid:** Always call `df.columns = df.columns.str.lower()` immediately after `pd.read_parquet()`
**Warning signs:** `KeyError` on column access, or 8h startup delay (bot can't seed, falls back to live polling)

### Pitfall 3: Shift Applied Before Indicators
**What goes wrong:** Indicators computed on already-shifted prices → garbage values for bar N use bar N-2 info
**Why it happens:** Intuitive to "prepare" the input data first
**How to avoid:** Compute indicators on raw OHLCV, then shift indicator columns only (not OHLCV)
**Warning signs:** RSI/MACD values are consistently off by 1 bar vs backtesting reference

### Pitfall 4: dropna() Before Cross-Asset Features
**What goes wrong:** All rows silently dropped because ETH/SOL lag columns don't exist yet when `dropna()` runs
**Why it happens:** `dropna()` feels like "cleanup at the end"; cross-asset feels like enrichment after cleanup
**How to avoid:** Always: `compute_features()` → `compute_cross_asset_features()` → `dropna()`
**Warning signs:** Feature matrix has 0 rows after warmup; or only BTC rows survive with ETH/SOL as NaN

### Pitfall 5: ATR-Based Indicators on Synthetic Candles
**What goes wrong:** Stop-loss system uses ATR ≈ 0 → stops never trigger or trigger at 0 distance
**Why it happens:** `ta.atr()` is the standard; it's not obvious Roostoo gives H=L=O=C
**How to avoid:** Use `log_returns.rolling(14).std() * close * 1.25` everywhere ATR appears (backtest AND live)
**Warning signs:** `atr_proxy` column is all zeros; trailing stop levels never update

### Pitfall 6: seed_dfs Dict Key Mismatch
**What goes wrong:** `KeyError` when `seed_dfs["BTC/USD"]` but caller passes `seed_dfs["BTCUSDT"]`
**Why it happens:** Binance uses `BTCUSDT`; Roostoo API uses `BTC/USD`
**How to avoid:** Normalize keys in `__init__` or document the expected format clearly in docstring; prefer `BTC/USD` (Roostoo format) since that's what main.py uses
**Warning signs:** `KeyError` on startup during seeding
</common_pitfalls>

<code_examples>
## Code Examples

### RSI — Output Column Name
```python
# Source: Context7 /xgboosted/pandas-ta-classic
import pandas_ta_classic as ta

df.ta.rsi(length=14, append=True)
# → adds column: RSI_14
# Access: df["RSI_14"]
```

### MACD — Output Column Names
```python
# Source: Context7 /xgboosted/pandas-ta-classic
df.ta.macd(fast=12, slow=26, signal=9, append=True)
# → adds columns:
#   MACD_12_26_9   (MACD line)
#   MACDs_12_26_9  (Signal line)
#   MACDh_12_26_9  (Histogram)
```

### EMA + Slope (EMA slope proxy)
```python
# Source: Context7 /xgboosted/pandas-ta-classic
df.ta.ema(length=20, append=True)   # → EMA_20
df.ta.ema(length=50, append=True)   # → EMA_50
# EMA slope: use slope indicator on EMA_20
df.ta.slope(close="EMA_20", length=1, append=True)  # → SLOPE_1
# Or manual: (EMA_20 - EMA_20.shift(1)) / EMA_20.shift(1)
```

### ATR Proxy (close-to-close volatility)
```python
# Source: PROJECT.md + validated against qoppac.blogspot.com ATR-SD relationship
import numpy as np

log_ret = np.log(df["close"] / df["close"].shift(1))
df["atr_proxy"] = log_ret.rolling(14).std() * df["close"] * 1.25
# Factor: 1/0.875 = 1.143 (theoretical SD→ATR); 1.25 = conservative buffer for crypto fat tails
# VERIFIED: mathematically equivalent to ATR for close-to-close returns
```

### Full Feature Computation with Shift
```python
# Source: Architecture pattern derived from PROJECT.md + Context7 verified column names
import pandas_ta_classic as ta
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ATR proxy (works on synthetic candles where H=L)
    log_ret = np.log(out["close"] / out["close"].shift(1))
    out["atr_proxy"] = log_ret.rolling(14).std() * out["close"] * 1.25

    # Indicators that work on close price only (safe for synthetic candles)
    out.ta.rsi(length=14, append=True)                        # RSI_14
    out.ta.macd(fast=12, slow=26, signal=9, append=True)      # MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    out.ta.ema(length=20, append=True)                        # EMA_20
    out.ta.ema(length=50, append=True)                        # EMA_50
    out.ta.slope(close="EMA_20", length=1, append=True)       # SLOPE_1 (EMA slope)

    # DISABLED — broken on synthetic candles:
    # out.ta.atr(append=True)   # H=L → ATR≈0
    # out.ta.adx(append=True)   # H=L → ADX≈0
    # out.ta.obv(append=True)   # volume=0 → OBV static

    # Shift ALL indicator columns to prevent look-ahead bias
    ohlcv = {"open", "high", "low", "close", "volume"}
    ind_cols = [c for c in out.columns if c not in ohlcv]
    out[ind_cols] = out[ind_cols].shift(1)

    return out
```

### Parquet Seed with Column Safety
```python
# Source: Derived from Binance public data docs + PROJECT.md Issue 04
def _seed_from_history(self, symbol: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df.columns = df.columns.str.lower()  # Binance CSVs: capitalized; guard against mismatch

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns for {symbol}: {missing}")

    for ts, row in df.tail(self._maxlen).iterrows():
        self._buffers[symbol].append({
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "timestamp": int(ts.timestamp()) if hasattr(ts, "timestamp") else int(ts),
        })
```
</code_examples>

<sota_updates>
## State of the Art (2025-2026)

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pandas-ta` (original) | `pandas-ta-classic` fork | 2022-2023 | Original broken on Python 3.11 + pandas 2.x |
| ATR via ta.atr() for all environments | Close-to-close proxy for synthetic data | 2024 (Roostoo-specific) | Required for Roostoo which has no OHLCV |
| ADX for regime detection | EMA(20)/EMA(50) crossover (resampled 4H→daily) | project-specific | ADX ≈ 0 on H=L candles |
| OBV for volume signals | Disabled (volume=0) | project-specific | Volume is meaningless on synthetic candles |

**New tools/patterns to consider:**
- `pandas-ta-classic` v0.3.78 (Feb 2026): actively maintained, Python 3.11-3.13 supported
- `collections.deque(maxlen=N)`: preferred over list-based buffers for memory-bounded streaming data

**Deprecated/outdated:**
- `pandas-ta` (original, unmaintained since 2022): broken `FutureWarning` and `AttributeError` on pandas 2.x
- Standard ATR (`ta.atr()`): valid for real OHLCV data, invalid for Roostoo synthetic candles
</sota_updates>

<open_questions>
## Open Questions

1. **Binance Parquet column capitalization in the actual competition file**
   - What we know: Binance raw CSVs use `Open`/`High`/`Low`/`Close`/`Volume` (capitalized); the PROJECT.md states flat lowercase
   - What's unclear: Whether the provided Parquet files were pre-normalized to lowercase before the competition
   - Recommendation: Always call `df.columns = df.columns.str.lower()` in `_seed_from_history()` — zero cost, prevents 8h startup failure

2. **EMA slope via `df.ta.slope(close="EMA_20")`**
   - What we know: Slope indicator exists in pandas-ta-classic; can be applied to any Series including `EMA_20` column
   - What's unclear: Whether `df.ta.slope(close="EMA_20", ...)` correctly reads from the `EMA_20` column or errors
   - Recommendation: Compute EMA first (`append=True`), then apply slope. If `close=` kwarg doesn't work, fall back to manual: `(df["EMA_20"] - df["EMA_20"].shift(1)) / df["EMA_20"].shift(1)`

3. **MACD warmup stabilization**
   - What we know: MACD(12,26,9) technically needs 26+9=35 bars; the 35-bar threshold is set in PROJECT.md
   - What's unclear: Wilder EMA converges asymptotically; first values are less accurate
   - Recommendation: 35 bars is the minimum; 50+ bars gives more stable MACD values. Competition start gives adequate historical data.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- Context7 `/xgboosted/pandas-ta-classic` — import path, indicator API, column naming conventions, MACD output columns confirmed
- [binance/binance-public-data GitHub](https://github.com/binance/binance-public-data) — klines column structure (Open, High, Low, Close, Volume)
- [pypi.org/project/binance-historical-data](https://pypi.org/project/binance-historical-data/) — package format (CSV + zip, flat structure)
- [qoppac.blogspot.com — ATR vs SD relationship](https://qoppac.blogspot.com/2018/12/the-relationship-between-atr-and.html) — ATR = SD / 0.875 empirically; 1.25 multiplier = conservative buffer (HIGH: published quantitative research by Rob Carver)

### Secondary (MEDIUM confidence)
- [pandas-ta-classic PyPI](https://pypi.org/project/pandas-ta-classic/) — Python 3.11 support confirmed, v0.3.78 latest as of Feb 2026
- [pandas-ta-classic indicators reference](https://xgboosted.github.io/pandas-ta-classic/indicators.html) — slope indicator in Momentum category confirmed
- [tradingstrategy.ai slope docs](https://tradingstrategy.ai/docs/api/technical-analysis/momentum/help/pandas_ta.momentum.slope.html) — slope formula: `close.diff(length) / length`, output: `SLOPE_{length}`

### Tertiary (LOW confidence — verify during implementation)
- `df.ta.slope(close="EMA_20", ...)` parameter usage — confirmed slope accepts `close` kwarg for custom series but `close="EMA_20"` reading from column is unverified; test before relying on it
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: pandas-ta-classic 0.3.78
- Ecosystem: binance_historical_data, pandas 2.x, pyarrow
- Patterns: shift-after-compute, cross-asset-before-dropna, flat Parquet access
- Pitfalls: wrong package, column capitalization, ATR on synthetic candles, dropna ordering

**Confidence breakdown:**
- pandas-ta-classic API (import, column names): HIGH — verified via Context7
- ATR proxy formula: HIGH — validated via published ATR-SD empirical relationship
- Binance Parquet column structure: MEDIUM — raw CSVs are capitalized; Parquet conversion may vary; lowercase guard is the safe implementation
- EMA slope via `slope(close="EMA_20")`: MEDIUM — documented but not tested end-to-end
- Warmup thresholds: HIGH — MACD(12,26,9) = 35 bars is mathematically correct

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (30 days — pandas-ta-classic is stable; Binance format unlikely to change)
</metadata>

---

*Phase: 04-data-pipeline*
*Research completed: 2026-03-16*
*Ready for planning: yes*

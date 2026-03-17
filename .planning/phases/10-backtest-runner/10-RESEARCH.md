# Phase 10: Backtest Runner + Feature Prep - Research

**Researched:** 2026-03-17
**Domain:** Python ML backtesting — XGBoost inference + bar-by-bar simulation + financial metrics
**Confidence:** HIGH

<research_summary>
## Summary

Phase 10 requires three capabilities: (1) loading the existing feature pipeline and running it on historical Parquet data, (2) loading a pre-trained XGBoost `.pkl` model and running bar-by-bar inference, and (3) computing a comprehensive stats report including Sharpe, Sortino, and max drawdown correctly annualized for 4H bars.

The standard approach for this type of ML backtest is bare-pandas simulation — no backtesting framework needed. The feature pipeline already exists in `bot/data/features.py` and is designed for this use. `quantstats` handles Sharpe/Sortino/drawdown with a `periods=2190` parameter for 4H crypto bars. XGBoost loads cleanly from `.pkl` via `pickle.load()`.

**Primary recommendation:** Hand-roll the bar-by-bar simulation loop (it's ~50 lines), reuse `bot/data/features.py` exactly as-is, and use `quantstats` for financial metrics. Do NOT use vectorbt, backtesting.py, or backtrader — they fight the custom feature pipeline rather than helping it.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| xgboost | >=2.0 | Load `.pkl` model, `predict_proba()` | The model format; needed for sklearn API deserialization |
| quantstats | latest | Sharpe, Sortino, max_drawdown, drawdown_details | Industry standard for quant reporting; correct annualization API |
| pandas | >=2.0 (installed) | Data manipulation, returns series | Already in requirements.txt |
| numpy | >=2.0 (installed) | Numerical ops | Already in requirements.txt |
| pyarrow | >=14.0 (installed) | Parquet loading | Already in requirements.txt |
| argparse | stdlib | CLI args (`--model`, `--data`, `--start`, `--end`, `--threshold`) | No extra dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | — | Not required for inference | Only needed if you retrain; xgboost sklearn API loads without it |
| scipy | — | Not required | quantstats handles all stats |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| quantstats | hand-rolled formulas | quantstats handles edge cases (zero std, first negative return) correctly; don't reinvent |
| bare pandas loop | vectorbt | vectorbt is powerful but fights our custom feature pipeline; overkill for 1 model/1 asset |
| bare pandas loop | backtesting.py | backtesting.py wraps strategies in a class that doesn't fit our pipeline's design |
| pickle | model.load_model() .json | Both work; if Phase 11 saves as .pkl, stick with pickle |

**New dependencies to add to requirements.txt:**
```bash
pip install xgboost quantstats
# Requirements to add:
# xgboost>=2.0
# quantstats>=0.0.62
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```
scripts/
└── backtest.py          # Single script, ~200 lines

# Imports from existing bot package:
# bot/data/features.py  — compute_features(), compute_cross_asset_features()
```

### Pattern 1: Pre-compute all features, then iterate bar-by-bar

**What:** Run the entire feature pipeline on all historical data at once, THEN loop bar-by-bar feeding each row to the model. Do NOT re-compute features inside the loop.

**Why:** The feature pipeline's `shift(1)` already prevents lookahead — bar N's row only contains bar N-1's indicator values. Re-running features per-bar would be computationally wasteful and wouldn't add any additional lookahead protection.

**When to use:** Always. This is the correct architecture for offline backtesting when features are already lookahead-safe.

```python
# Source: bot/data/features.py design + standard ML backtest pattern
import pandas as pd
import pickle
import quantstats as qs
from bot.data.features import compute_features, compute_cross_asset_features

# Step 1: Load all parquet files
btc = pd.read_parquet("data/BTCUSDT_4h.parquet")
eth = pd.read_parquet("data/ETHUSDT_4h.parquet")
sol = pd.read_parquet("data/SOLUSDT_4h.parquet")

# Step 2: Run feature pipeline ONCE on full dataset
btc_feat = compute_features(btc)
btc_feat = compute_cross_asset_features(btc_feat, {"ETH/USD": eth, "SOL/USD": sol})
btc_feat = btc_feat.dropna()

# Step 3: Filter to backtest date range
btc_feat = btc_feat.loc[start_date:end_date]

# Step 4: Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Step 5: Bar-by-bar simulation
FEATURE_COLS = [c for c in btc_feat.columns if c not in {"open","high","low","close","volume"}]
for idx, row in btc_feat.iterrows():
    features = pd.DataFrame([row[FEATURE_COLS]])
    proba_buy = model.predict_proba(features)[0][1]
    # ... generate signal, track position, track PnL
```

### Pattern 2: Signal generation with probability threshold

**What:** XGBoost binary classifier outputs `predict_proba()` → `[P(class=0), P(class=1)]`. Column index `[1]` is P(BUY). Apply threshold to convert probability to BUY/SELL/HOLD.

**Standard threshold:** 0.6 for BUY, < 0.4 for SELL, 0.4–0.6 is HOLD. Make `--threshold` a CLI arg.

```python
# Source: XGBoost docs + community pattern for binary trading classifiers
def get_signal(proba_buy: float, threshold: float = 0.6) -> str:
    if proba_buy >= threshold:
        return "BUY"
    elif proba_buy <= (1.0 - threshold):
        return "SELL"
    else:
        return "HOLD"
```

### Pattern 3: PnL tracking and returns series construction

**What:** Track position (LONG/FLAT), entry price, and compute bar-by-bar returns. Build a returns Series that quantstats can consume.

**Critical:** Returns Series must be indexed by timestamp (DatetimeIndex) for quantstats to work correctly.

```python
# Standard pattern for position tracking in bar-by-bar backtests
position = 0  # 0 = flat, 1 = long, -1 = short
entry_price = None
returns = []
timestamps = []

for idx, row in btc_feat.iterrows():
    close = row["close"]
    bar_return = 0.0

    if position == 1:
        bar_return = (close - prev_close) / prev_close  # long return

    signal = get_signal(model.predict_proba(pd.DataFrame([row[FEATURE_COLS]]))[0][1], threshold)

    if signal == "BUY" and position == 0:
        position = 1
        entry_price = close
    elif signal == "SELL" and position == 1:
        position = 0
        entry_price = None

    returns.append(bar_return)
    timestamps.append(idx)
    prev_close = close

returns_series = pd.Series(returns, index=timestamps)
```

### Pattern 4: Financial metrics with correct 4H annualization

**What:** quantstats `sharpe()` and `sortino()` accept a `periods` parameter for non-daily data. For 4H crypto (always-on market): **2190 bars/year** (365.25 × 24 / 4 = 2191.5).

**CRITICAL:** Default `periods=252` (daily equity market) produces a Sharpe ~9× too low for 4H data. Always override.

```python
# Source: quantstats GitHub source code — sharpe/sortino both use sqrt(periods) annualization
import quantstats as qs

PERIODS_4H = 2190  # 365.25 * 24 / 4 = 2191.5 ≈ 2190

sharpe = qs.stats.sharpe(returns_series, periods=PERIODS_4H)
sortino = qs.stats.sortino(returns_series, periods=PERIODS_4H)
max_dd = qs.stats.max_drawdown(returns_series)   # returns negative number, e.g. -0.23 = -23%
```

### Anti-Patterns to Avoid

- **Re-computing features inside the bar loop:** Wasteful; features must be computed once on full history (MACD needs 35+ bars of history per computation)
- **Using quantstats win_rate() for trade win rate:** `qs.stats.win_rate()` is PERIOD-based (% of bars with positive PnL) not TRADE-based (% of closed trades profitable). Compute trade win rate yourself.
- **Using default periods=252 for 4H data:** Sharpe/Sortino will be wrong by a factor of ~3 (sqrt(2190/252) ≈ 2.95)
- **Ignoring warmup period:** Skip first 50 rows after dropna to ensure MACD(12,26,9) has full warmup. The pipeline's dropna() handles NaN removal but warmup quality improves after ~50 bars.
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sharpe ratio | Custom formula | `qs.stats.sharpe(returns, periods=2190)` | Edge cases: zero std, single-period, first negative return phantom peak |
| Sortino ratio | Custom formula | `qs.stats.sortino(returns, periods=2190)` | Downside deviation has subtle implementation variants; quantstats uses the standard |
| Max drawdown | Custom formula | `qs.stats.max_drawdown(returns)` | "Phantom baseline" handling for first-bar negative returns; quantstats handles correctly |
| Drawdown details | Custom formula | `qs.stats.drawdown_details(qs.stats.to_drawdown_series(returns))` | Returns start/end/duration of each drawdown period |
| Feature pipeline | Custom reimplementation | Reuse `bot/data/features.py` directly | Guarantees backtest and live bot use identical features — calibration consistency |

**Key insight:** The feature pipeline was explicitly designed in `features.py` with a docstring that says "safe to call in both backtest and live environments. The output is identical in both contexts." This is the calibration consistency guarantee — DO NOT reimplement.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Wrong annualization factor (4H vs daily)
**What goes wrong:** Sharpe of 0.4 gets reported as 0.4 instead of the correct ~1.2, misleading evaluation.
**Why it happens:** Default `periods=252` is hardcoded for daily equity markets; crypto 4H has 2190 bars/year.
**How to avoid:** Always pass `periods=2190` to `qs.stats.sharpe()` and `qs.stats.sortino()`. Define `PERIODS_4H = 2190` as a constant.
**Warning signs:** Sharpe ratios that seem 3× lower than expected; identical strategy gives wildly different results in different frameworks.

### Pitfall 2: XGBoost feature name/order mismatch
**What goes wrong:** `UserWarning: X does not have valid feature names` or silent wrong predictions.
**Why it happens:** If you pass a numpy array instead of a named DataFrame to `predict_proba()`, XGBoost loses column name validation. If column ORDER differs from training, predictions are wrong with no error.
**How to avoid:** Always pass a `pd.DataFrame` with column names when calling `predict_proba()`. After loading the model, check `model.feature_names_in_` and use it to select/order columns.
**Warning signs:** UserWarning about feature names; suspiciously flat signal distribution (all HOLDs or all BUYs).

```python
# Safe prediction pattern
FEATURE_COLS = list(model.feature_names_in_)  # use model's expected order
features_row = pd.DataFrame([row[FEATURE_COLS]])
proba = model.predict_proba(features_row)[0][1]
```

### Pitfall 3: quantstats win_rate is period-based, not trade-based
**What goes wrong:** win_rate reports 52% meaning "52% of 4H bars had positive PnL", not "52% of closed trades were winners".
**Why it happens:** quantstats is built for always-invested strategies; it doesn't know about discrete trades.
**How to avoid:** Compute trade win rate manually from the list of closed trades. Track `(entry_price, exit_price)` pairs.
**Warning signs:** win_rate is very close to 50% even for a high-signal strategy (most bars are HOLD = 0 return).

```python
# Trade-based win rate — hand-roll this one
trade_pnls = [exit_price - entry_price for entry_price, exit_price in closed_trades]
win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) if trade_pnls else 0.0
```

### Pitfall 4: Returns index not DatetimeIndex
**What goes wrong:** `quantstats` functions fail or produce NaN metrics.
**Why it happens:** quantstats expects a DatetimeIndex on the returns Series to handle resampling internally.
**How to avoid:** Ensure the Parquet file has a DatetimeIndex (or convert after loading), and preserve it through the feature pipeline.
**Warning signs:** `AttributeError` in quantstats; monthly_returns heatmap fails.

### Pitfall 5: Not handling HOLD signal for position tracking
**What goes wrong:** HOLD is treated as SELL, causing strategy to exit every bar that isn't BUY.
**Why it happens:** Sloppy if/elif in signal handler; `signal != "BUY"` accidentally triggers exit.
**How to avoid:** Three-way state machine: BUY enters long, SELL exits long, HOLD does nothing.
**Warning signs:** Hundreds of tiny trades instead of sustained positions; high turnover in trade log.
</common_pitfalls>

<code_examples>
## Code Examples

### Loading XGBoost Model from .pkl
```python
# Source: Context7 /dmlc/xgboost — sklearn_examples.md (HIGH confidence)
import pickle
import xgboost  # must be imported so pickle can deserialize XGBClassifier

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Verify feature names match
print("Model expects features:", list(model.feature_names_in_))
print("Model classes:", model.classes_)  # [0, 1] — 0=no-buy, 1=buy
```

### Predict Signal from Feature Row
```python
# Source: XGBoost docs + verified pattern (HIGH confidence)
import pandas as pd

def predict_signal(model, row: pd.Series, feature_cols: list, threshold: float = 0.6) -> str:
    X = pd.DataFrame([row[feature_cols]])  # named DataFrame preserves column order
    proba_buy = model.predict_proba(X)[0][1]  # P(class=1)
    if proba_buy >= threshold:
        return "BUY"
    elif proba_buy <= (1.0 - threshold):
        return "SELL"
    else:
        return "HOLD"
```

### Financial Metrics with quantstats (4H Annualization)
```python
# Source: quantstats GitHub source code — stats.py (HIGH confidence)
import quantstats as qs
import pandas as pd

PERIODS_4H = 2190  # 365.25 * 24 / 4

def compute_stats_report(returns: pd.Series) -> dict:
    """Returns a dict of all key stats for the backtest report."""
    returns = returns[returns.index.notna()]  # guard against NaT index

    return {
        "total_return": float((1 + returns).prod() - 1),
        "sharpe": qs.stats.sharpe(returns, periods=PERIODS_4H),
        "sortino": qs.stats.sortino(returns, periods=PERIODS_4H),
        "max_drawdown": qs.stats.max_drawdown(returns),  # negative, e.g. -0.23
        "win_rate_periods": qs.stats.win_rate(returns),  # % of bars with positive return
        "volatility": qs.stats.volatility(returns, periods=PERIODS_4H),
        "cagr": qs.stats.cagr(returns, periods=PERIODS_4H),
    }
```

### Feature Pipeline Invocation (reusing existing code)
```python
# Source: bot/data/features.py — existing codebase (HIGH confidence)
import pandas as pd
from bot.data.features import compute_features, compute_cross_asset_features

def prepare_features(btc_path: str, eth_path: str, sol_path: str) -> pd.DataFrame:
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)
    sol = pd.read_parquet(sol_path)

    # Ensure DatetimeIndex
    for df in [btc, eth, sol]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    # Run feature pipeline — SAME as live bot
    feat = compute_features(btc)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    feat = feat.dropna()  # after cross-asset injection, per features.py design

    return feat
```

### Complete Bar-by-Bar Loop Skeleton
```python
# Source: Synthesized from patterns above (HIGH confidence for structure)
def run_backtest(feat_df: pd.DataFrame, model, threshold: float = 0.6):
    feature_cols = list(model.feature_names_in_)  # safe column ordering

    position = 0  # 0=flat, 1=long
    entry_price = None
    prev_close = None
    returns = []
    timestamps = []
    closed_trades = []

    for idx, row in feat_df.iterrows():
        close = row["close"]

        # Compute bar return for current position
        bar_return = 0.0
        if position == 1 and prev_close is not None:
            bar_return = (close - prev_close) / prev_close

        # Generate signal
        X = pd.DataFrame([row[feature_cols]])
        proba_buy = model.predict_proba(X)[0][1]
        if proba_buy >= threshold:
            signal = "BUY"
        elif proba_buy <= (1.0 - threshold):
            signal = "SELL"
        else:
            signal = "HOLD"

        # State machine
        if signal == "BUY" and position == 0:
            position = 1
            entry_price = close
        elif signal == "SELL" and position == 1:
            closed_trades.append({"entry": entry_price, "exit": close,
                                   "pnl_pct": (close - entry_price) / entry_price})
            position = 0
            entry_price = None

        returns.append(bar_return)
        timestamps.append(idx)
        prev_close = close

    returns_series = pd.Series(returns, index=pd.DatetimeIndex(timestamps))
    return returns_series, closed_trades
```
</code_examples>

<sota_updates>
## State of the Art (2025-2026)

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| backtesting.py for ML strategies | bare-pandas loop + quantstats | Ongoing | Frameworks fight custom pipelines; hand-roll is simpler |
| Manual Sharpe implementation | quantstats with `periods` param | 2022+ | quantstats handles edge cases (phantom peak, zero-std) correctly |
| numpy-only feature computation | pandas-ta-classic with shift(1) | Already done | Consistent with live bot |
| DMatrix for XGBoost inference | sklearn API `predict_proba()` | Ongoing | sklearn API is simpler; DMatrix only needed for performance at scale |

**New tools/patterns to consider:**
- **vectorbt PRO:** For parameter sweeps (hundreds of thresholds), but overkill here — we have 1 model, 1 asset
- **mlflow / wandb:** For experiment tracking across Phase 11 training runs — not needed for Phase 10

**Deprecated/outdated:**
- **backtrader:** Heavy, old, fights ML pipelines. Community has moved to vectorbt or bare-pandas.
- **pyfolio:** Dependencies broken on Python 3.11+; replaced by quantstats.
</sota_updates>

<open_questions>
## Open Questions

1. **Long/short or long-only?**
   - What we know: CONTEXT says XGBoost outputs BUY/SELL/HOLD; signal routing unclear
   - What's unclear: Does SELL mean "go short" or "close long"?
   - Recommendation: Implement long/short with a flag `--long-only` defaulting to long-only (simpler, lower risk in hackathon context); model trainer (Phase 11) decides

2. **Position sizing**
   - What we know: RiskManager uses ATR-based sizing; CONTEXT says "simple backtest sim is fine"
   - What's unclear: Should backtest use 100% capital per trade or fraction?
   - Recommendation: Default to 100% capital per trade for simplicity; add `--position-size` arg as optional

3. **Transaction costs**
   - What we know: Roostoo is a hackathon exchange; likely 0 maker/taker fees
   - What's unclear: Does backtest need to model slippage?
   - Recommendation: Add a `--fee-bps 0` arg (default 0) so it can be toggled if needed

4. **Multi-pair backtesting**
   - What we know: The bot trades BTC/ETH/SOL; the model may only be trained for BTC
   - What's unclear: Will Phase 11 train separate models per pair?
   - Recommendation: Implement for BTC only in Phase 10; add pair selection arg for future extension
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- Context7 `/dmlc/xgboost` — sklearn API, pickle loading, predict_proba interface
- Context7 `/ranaroussi/quantstats` — stats function list, sharpe/sortino/max_drawdown API
- GitHub `ranaroussi/quantstats/blob/main/quantstats/stats.py` (WebFetch) — sharpe/sortino `periods` parameter, win_rate implementation, max_drawdown implementation
- `bot/data/features.py` (codebase read) — feature column names, shift(1) behaviour, cross-asset pattern

### Secondary (MEDIUM confidence)
- XGBoost readthedocs prediction page — predict_proba output format [n_samples, n_classes], threshold patterns
- vectorbt.dev — confirmed overkill for single-model backtest; quantstats adapter noted

### Tertiary (LOW confidence - needs validation)
- Sharpe annualization factor 2190 for 4H crypto: derived from first principles (365.25×24/4), not found in official source. Validate against: `qs.stats.sharpe(returns, periods=2190)` produces sensible numbers during testing.
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: XGBoost sklearn API (model loading + prediction)
- Ecosystem: quantstats (metrics), pandas (returns series), pyarrow (parquet)
- Patterns: Bar-by-bar simulation, lookahead prevention, position tracking
- Pitfalls: Annualization, feature name mismatch, win_rate definition, HOLD signal handling

**Confidence breakdown:**
- XGBoost model loading: HIGH — Context7 confirms pickle pattern with source examples
- Quantstats API (sharpe/sortino/drawdown): HIGH — GitHub source code verified
- Annualization factor 2190: MEDIUM — derived from first principles; verify empirically
- Bar-by-bar simulation pattern: HIGH — standard, no gotchas beyond documented pitfalls
- Trade-based win rate: HIGH — quantstats explicitly period-based; hand-roll is required

**Research date:** 2026-03-17
**Valid until:** 2026-04-17 (30 days — stable ecosystem)
</metadata>

---

*Phase: 10-backtest-runner*
*Research completed: 2026-03-17*
*Ready for planning: yes*

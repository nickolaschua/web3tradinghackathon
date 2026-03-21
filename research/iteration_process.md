# Strategy Iteration Process: Developing Better Features Without Overfitting

## The core problem

Every time you run a backtest, look at the results, and change something, you are implicitly
fitting to that backtest period. Do this enough times and the strategy performs well on that
period not because it has genuine alpha, but because you have memorized the noise. The process
below is designed to make this systematic error visible and costly.

The existing codebase already has the right pieces:
- `train_model.py` has 5-fold `TimeSeriesSplit` CV with `gap=24` bars
- `backtest_15m.py` has `TRAIN_CUTOFF = "2024-01-01"` — a held-out test set
- The backtest has `--sweep` and `--atr-sweep` for parameter scanning

The process here builds on those foundations and specifies exactly when to use each tool,
in what order, and what the acceptance criteria are.

---

## 1. The data partition (set once, never change)

The most important rule: **decide your data splits before you start iterating, then lock them**.

```
Full history (e.g. 2022-01-01 → today)
│
├── DEVELOPMENT WINDOW: 2022-01-01 → 2023-06-30
│   Used for: feature engineering, IC testing, hypothesis generation
│   You may look at this freely — it is your scratchpad
│
├── VALIDATION WINDOW: 2023-07-01 → 2023-12-31
│   Used for: walk-forward CV, hyperparameter selection
│   You may look at this only through the CV output, never directly
│
└── TEST WINDOW: 2024-01-01 → present   ← TRAIN_CUTOFF in backtest_15m.py
    LOCKED. Run the final backtest here exactly ONCE per candidate model.
    Never use test-period results to change strategy parameters.
    If you peek here and then adjust, this data is compromised.
```

The test window (post-2024) is already protected in `backtest_15m.py` via `TRAIN_CUTOFF`.
The development/validation split within pre-2024 data is what you need to enforce manually.

**Practical rule:** Never run `backtest_15m.py` on the full dataset to guide a parameter decision.
Only run it as a final confirmation after all parameter decisions are made on the development
window.

---

## 2. Before adding any feature: the IC test

IC = Information Coefficient = Spearman correlation between a feature value at bar T and the
forward return at bar T+horizon. A feature with IC = 0 is random noise. A feature with
|IC| > 0.03 consistently is worth investigating.

**Run this before writing a single line of integration code:**

```python
# scripts/test_ic.py — run this for every candidate feature

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from bot.data.features import compute_features

def compute_ic(feature_series: pd.Series, close: pd.Series, horizon: int = 6) -> float:
    """
    Spearman IC: correlation between feature[t] and log_return[t → t+horizon].
    Shift feature forward by horizon to avoid look-ahead.
    """
    fwd_ret = np.log(close.shift(-horizon) / close)
    aligned = pd.concat([feature_series, fwd_ret], axis=1).dropna()
    if len(aligned) < 50:
        return float("nan")
    ic, pvalue = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(ic)

# Test a candidate feature across rolling 90-day windows
def rolling_ic(feature_series: pd.Series, close: pd.Series, horizon: int = 6, window: int = 360) -> pd.Series:
    """Rolling IC over 90-day windows (360 × 15M = 90 days)."""
    results = {}
    for end in range(window, len(feature_series), window // 3):  # step = 30 days
        start = end - window
        ic = compute_ic(
            feature_series.iloc[start:end],
            close.iloc[start:end],
            horizon=horizon,
        )
        results[feature_series.index[end]] = ic
    return pd.Series(results)

# Example: test a new feature "rsi_zscore"
btc = pd.read_parquet("data/btc_15m.parquet")
feat = compute_features(btc)

# Compute the candidate feature
feat["rsi_zscore"] = (feat["RSI_14"] - feat["RSI_14"].rolling(42).mean()) / \
                     (feat["RSI_14"].rolling(42).std() + 1e-8)

ric = rolling_ic(feat["rsi_zscore"], btc["close"], horizon=6)
print(f"Rolling IC stats:")
print(f"  Mean:     {ric.mean():.4f}  (need > 0.03 to add value)")
print(f"  Positive: {(ric > 0).mean():.0%}  (need > 60% of windows)")
print(f"  Std:      {ric.std():.4f}  (lower = more stable)")
print(f"  Min:      {ric.min():.4f}  Max: {ric.max():.4f}")
```

**Acceptance criteria for a new feature:**
- Mean IC > 0.03 (or < -0.03 for contrarian features)
- Positive (or negative) in > 60% of rolling windows — signal is stable across regimes
- IC is NOT concentrated in one or two windows — that's memorization, not signal

Features that fail this test should not be added to the model regardless of how good they
look in a backtest. A backtest can appear good for a feature with IC = 0 purely by chance,
especially in a short test window.

**The IC test uses only the development window (pre-2023-07-01).** Do not compute IC on the
full dataset — that leaks test-set information into your feature selection.

---

## 3. Walk-forward cross-validation (the main validation tool)

`train_model.py` already runs `TimeSeriesSplit(n_splits=5, gap=24)`. This is good. Use it
correctly:

**Purging:** The `gap=24` bars prevents label leakage from the 6-bar forward horizon. But
for the new features (BTC lead-lag with 42-bar lookbacks, cross-sectional ranks with 168-bar
lookbacks), the gap should be at least as large as the longest feature lookback:

```python
# In train_model.py, when adding Strategy 1 or 2 features:
# Change gap to match your longest rolling feature window
LONGEST_LOOKBACK = 168   # 168-bar rank window from Strategy 2

tscv = TimeSeriesSplit(n_splits=5, gap=LONGEST_LOOKBACK + 6)  # lookback + label horizon
```

Without this, the validation fold's features overlap with the training fold's label period —
the model sees "future" information during training, and CV scores look better than they
actually are.

**Embargo:** Add an additional buffer after each training fold ends:

```python
# Manual embargo: skip first EMBARGO bars of each validation fold
EMBARGO = 42   # 7 days of 4H bars

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    val_idx = val_idx[EMBARGO:]   # skip the first 42 bars of each val fold
    if len(val_idx) < 100:
        continue
    # ... rest of fold training
```

Embargo prevents leakage from features that are computed over rolling windows — if a training
sample ends on day 100 and a feature uses a 7-day window, the feature values on days 101-107
in the validation fold still partially reflect training data.

**What to accept from walk-forward CV:**

| CV Mean AP | Interpretation |
|---|---|
| < 0.35 | Model has no predictive power — don't run backtest |
| 0.35–0.50 | Marginal signal — carefully check feature importances |
| 0.50–0.65 | Good signal — proceed to backtest |
| > 0.65 | Suspiciously good — check for look-ahead bias first |

Mean AP is the `average_precision_score` across folds, already printed by `train_model.py`.
The baseline for random guessing is the positive class rate (roughly 40-50% for the current
label threshold). AP significantly above baseline is the target.

**CV consistency check:** The AP across 5 folds should not vary wildly:

```
Acceptable:  Fold 0: 0.54  Fold 1: 0.51  Fold 2: 0.56  Fold 3: 0.49  Fold 4: 0.52
Suspicious:  Fold 0: 0.72  Fold 1: 0.38  Fold 2: 0.69  Fold 3: 0.41  Fold 4: 0.71
```

High variance across folds means the model only works in specific market conditions. It may
backtest well if those conditions happen to dominate the test period, but it will fail live.

---

## 4. Parameter sensitivity testing (before locking any threshold)

Every threshold in the strategy is a parameter. RSI < 30, bb_pos < 0.15, ENTRY_ZSCORE = 1.5 —
each was chosen for a reason, but none should be fragile to small perturbations.

**The perturbation rule:** A good parameter setting should produce similar results when
perturbed ±20%. If performance collapses at RSI < 28 or RSI < 32, the threshold at 30 is
cherry-picked and will not generalize.

```python
# scripts/sensitivity_test.py — run on development window only

import itertools

# For each threshold in the strategy, define a ±20% range
param_grid = {
    "rsi_entry": [24, 27, 30, 33, 36],        # ±20% around 30
    "bb_pos_entry": [0.08, 0.12, 0.15, 0.18, 0.22],   # ±20% around 0.15
    "entry_zscore": [1.2, 1.35, 1.5, 1.65, 1.8],      # ±20% around 1.5 (pairs)
}

results = []
for params in itertools.product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    sharpe = run_dev_window_backtest(**config)   # run on 2022-2023 only
    results.append({**config, "sharpe": sharpe})

df = pd.DataFrame(results)
print(df.sort_values("sharpe", ascending=False).head(10))

# A robust parameter: Sharpe stays within 0.3 of the best value across all perturbations
# A fragile parameter: Sharpe drops by 1.0+ when moved by one step
```

**Do not pick the best-performing parameter from the grid.** Pick the most stable one —
the value where performance is consistent across neighbors. Robustness is alpha; specific
parameters are memorization.

After sensitivity testing, fix the parameters. Do not re-run sensitivity testing after
seeing the test-window backtest.

---

## 5. The overfitting detection checklist

Run this before declaring any iteration "done" and before running `backtest_15m.py` on test data.

```
OVERFITTING RED FLAGS — if any of these are true, stop and investigate:

□ Walk-forward CV Mean AP > 0.70
  → Suspiciously good. Verify gap/embargo settings. Check for column leakage
    (did a feature accidentally encode the label?).

□ A single feature has importance > 50%
  → Run: model.feature_importances_ — if one feature dominates, the model has
    degenerated to a one-feature rule. Check whether that feature has look-ahead bias.
  → Acceptable: top feature at 25-35% when the model has 15+ features.

□ Entry threshold is not a round number
  → RSI threshold 27.3 instead of 27 or 30 is a sign it was found by search.
    If you ran more than 20 parameter trials, you have found noise.

□ Backtest Sharpe on development window > 2.5
  → Real crypto strategies have Sharpe 0.8-1.8 net of fees. Higher implies
    either look-ahead bias or extreme overfitting to the development window.
  → The existing backtest BEST is Sharpe 1.141 on OOS data. Use that as calibration.

□ Win rate > 75% or < 35%
  → Win rate > 75%: likely look-ahead bias. Win rate < 35%: the profit comes from
    a few large winners masking many small losers — fragile in live trading.

□ CV fold variance > 0.15 (max AP fold - min AP fold > 0.15)
  → Model only works in specific conditions.

□ Test-window performance > 30% better than development-window performance
  → The test window happened to be an unusual period. Do not update strategy
    parameters to "fix" the performance gap.

□ You have run more than 30 distinct backtests on the development window
  → Apply multiple testing correction (Section 6).

□ A feature was added because the backtest looked good, not because it passed IC test
  → This is the most common error. Always IC test first.
```

---

## 6. Multiple testing correction

Every time you run a backtest with a modified strategy and see a Sharpe improvement, you are
conducting a hypothesis test. If you run 50 backtests and keep the best, you expect to find
a Sharpe of approximately 2.5 purely by chance (the maximum of 50 random draws from a Sharpe
distribution). This is the Deflated Sharpe Ratio (DSR) problem from Bailey et al. (2016).

**Track every trial you run:**

```python
# Keep a research log — one row per backtest run (even failures)
# Save to research/backtest_log.csv

import csv
from datetime import datetime

def log_backtest(description: str, sharpe: float, sortino: float,
                 max_dd: float, changes_made: str):
    with open("research/backtest_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            description,
            round(sharpe, 3),
            round(sortino, 3),
            round(max_dd, 4),
            changes_made,
        ])
```

**Applying the DSR correction:** Given N total trials with average Sharpe `sr_bar` and
standard deviation `sr_std`, the expected maximum Sharpe from noise is approximately:

```python
import numpy as np
from scipy.stats import norm

def deflated_sharpe(sr_obs: float, sr_bar: float, sr_std: float,
                    n_trials: int, t: int) -> float:
    """
    Probability that observed Sharpe is above the expected maximum from noise.
    sr_obs:  Sharpe of the best strategy found
    sr_bar:  mean Sharpe across all N trials
    sr_std:  std of Sharpe across all N trials
    n_trials: total number of strategies/param combinations tested
    t:       number of bars in the backtest
    Returns DSR in [0, 1] — values > 0.95 indicate genuine alpha.
    """
    # Expected maximum Sharpe from N independent trials:
    e_max = sr_bar + sr_std * (
        (1 - np.euler_gamma) * norm.ppf(1 - 1.0/n_trials)
        + np.euler_gamma * norm.ppf(1 - 1.0/(n_trials * np.e))
    )
    # Sharpe ratio standard error under IID assumption:
    sr_se = np.sqrt((1 + 0.5 * sr_obs**2) / t)
    # DSR: probability that sr_obs exceeds the noise maximum
    dsr = norm.cdf((sr_obs - e_max) / sr_se)
    return dsr

# Example:
# You ran 40 backtests, mean Sharpe = 0.5, std = 0.3, best Sharpe = 1.4, 35040 bars
dsr = deflated_sharpe(sr_obs=1.4, sr_bar=0.5, sr_std=0.3, n_trials=40, t=35040)
print(f"DSR: {dsr:.3f}  ({'genuine alpha' if dsr > 0.95 else 'possibly noise'})")
```

Accept a strategy only if DSR > 0.95 once all trials are accounted for. If DSR < 0.95,
the strategy's apparent Sharpe is explained by the number of searches conducted.

---

## 7. The concrete iteration cycle

Each session of strategy development should follow this order. Do not skip steps.

### Step 1: Form a hypothesis (5 minutes)

Write down, in one sentence, the market inefficiency you believe the new feature captures.
"BTC's 1-bar lagged return predicts altcoin returns because BTC processes information faster
than individual altcoins."

If you cannot write this sentence, you are data mining, not doing research. Do not proceed.

### Step 2: IC test on development window (15 minutes)

Run `test_ic.py` on the development window only (pre-2023-07-01). Check:
- Mean IC > 0.03
- Positive in > 60% of rolling windows
- IC is not concentrated in 1-2 windows

If the feature fails: record why in a notes file. Stop. Do not integrate the feature.

### Step 3: Feature integration (30-60 minutes)

Add the feature to `features.py`. Verify:
- All columns are shifted 1 bar after computation (no look-ahead)
- Column names are consistent between training and backtest
- `compute_features()` still passes existing unit tests (if any)
- Run a spot check: `features.iloc[-1]` should return sensible values, not NaN

### Step 4: Walk-forward CV with new features (10 minutes)

Run `train_model.py` with updated `FEATURE_COLS`. Check:
- Mean AP is not worse than baseline (existing features without the new ones)
- CV variance is acceptable (< 0.15 spread across folds)
- Feature importance: does the new feature appear in the top 10?

If CV AP drops: the new feature is hurting, not helping. Remove it. Do not proceed to backtest.

### Step 5: Parameter sensitivity test on development window (20 minutes)

For any new thresholds introduced (entry conditions, exit conditions, window sizes): run the
sensitivity grid on the development window. Pick the most stable parameter, not the best one.
Log the full grid results.

### Step 6: Freeze parameters, retrain final model

With parameters fixed, train the final model on all development + validation data
(pre-2024-01-01) using the fixed `TRAIN_CUTOFF = "2024-01-01"` in `train_model.py`.
Save the model with a versioned name:

```bash
python scripts/train_model.py \
    --btc-path data/btc_4h.parquet \
    --eth-path data/eth_4h.parquet \
    --sol-path data/sol_4h.parquet \
    --save models/xgb_btc_4h_v2.pkl   # increment version
```

### Step 7: Run the test-window backtest ONCE

Run `backtest_15m.py` (or the 4H equivalent) on the full OOS window. This is the report card.
You get to run this **once per model version**. Record the result in `research/backtest_log.csv`.

```bash
python scripts/backtest_15m.py \
    --model models/xgb_btc_15m_v2.pkl \
    --threshold 0.70 --atr-mult 10
```

If the result is worse than the previous version: do NOT go back and adjust parameters to
improve it. That would compromise the test set. Instead, investigate the CV folds to understand
what changed and start a new iteration from Step 1 with a new hypothesis.

### Step 8: Sanity check against benchmarks

The test-window result is only meaningful relative to baselines. Run these comparisons:

```python
# Benchmark 1: BTC buy-and-hold over the same test window
btc_test = btc_close["2024-01-01":]
btc_sharpe = (btc_test.pct_change().mean() / btc_test.pct_change().std()) * np.sqrt(35040)

# Benchmark 2: Equal-weight universe hold
# (average Sharpe of holding all 39 coins equally weighted)

# Your strategy should beat both on:
# 1. Sharpe ratio (risk-adjusted return)
# 2. Maximum drawdown (drawdown should be lower than BTC buy-and-hold)
# 3. Calmar ratio (return / max_drawdown)
```

BTC buy-and-hold is the hardest benchmark to beat in a bull market — if the test period was
2024 (BTC +150%), your strategy must generate comparable returns with lower drawdown to be
worth the complexity.

### Step 9: Run the DSR check

With the full backtest log, compute DSR for the best strategy found so far:

```python
log = pd.read_csv("research/backtest_log.csv")
dsr = deflated_sharpe(
    sr_obs=log["sharpe"].max(),
    sr_bar=log["sharpe"].mean(),
    sr_std=log["sharpe"].std(),
    n_trials=len(log),
    t=35040,   # bars in the test window
)
print(f"DSR: {dsr:.3f}")
```

If DSR < 0.95: you have found noise, not alpha. The correct response is to stop looking
for improvements on this feature set and instead think harder about the economic mechanism
behind the next hypothesis.

---

## 8. Feature set management

### The feature budget

XGBoost overfits less than linear models but still overfits. A rough rule for this dataset:

```
Maximum features = min(n_positive_labels / 20, 30)
```

With ~40-50% positive rate on 35,000 bars, you have ~15,000-17,500 positive labels.
Maximum sensible features: min(17500/20, 30) = **30 features**.

The existing model has 12 features. The 6 strategies add up to roughly:
- Strategy 1 (BTC lead-lag): +6 features
- Strategy 2 (cross-sectional rank): +6 features
- Strategy 3 (funding rate): +4 features
- Mean reversion gate: uses existing features, no new XGBoost features

Total: ~28 features. This is at the upper limit. Do not add all at once — add in order of
IC strength and stop when CV AP stops improving.

### Feature redundancy check

Before adding a new feature, check its correlation with existing features:

```python
# After computing candidate features:
corr_matrix = feature_df[FEATURE_COLS + [new_feature]].corr()
max_corr = corr_matrix[new_feature].drop(new_feature).abs().max()
print(f"Max correlation with existing features: {max_corr:.3f}")
# If max_corr > 0.85: the feature is nearly redundant with an existing one
# XGBoost will split between them unpredictably → feature importance becomes unstable
# Consider: replace the weaker correlated feature rather than adding a duplicate
```

### Track feature importance across versions

After each model retrain, save the feature importance:

```python
import json

importance = dict(zip(model.feature_names_in_, model.feature_importances_))
with open(f"research/feature_importance_v{version}.json", "w") as f:
    json.dump(importance, f, indent=2)

# Flags to check:
# 1. Did any new feature enter the top 5? If yes, it's carrying real signal.
# 2. Did any existing feature drop below 1% importance? If yes, consider removing it.
# 3. Did the top feature's importance increase above 40%? If yes, the model is concentrating.
```

---

## 9. Strategy-specific iteration notes

### For XGBoost features (Strategies 1, 2, 3)

- The IC test is mandatory before integration
- Always retrain with updated `FEATURE_COLS` in `train_model.py`
- Walk-forward CV is the primary gate — backtest is just confirmation

### For rule-based filters (Strategy 4: token unlock, Strategy 5: mean reversion gate)

Rule-based strategies do not have model parameters to tune, but they have **threshold parameters**
(RSI < 30, ENTRY_ZSCORE = 1.5). These still require sensitivity testing. The acceptance criterion
is: does removing or loosening this rule make results noticeably worse? If results are unchanged
without the rule, the rule is not adding value.

```python
# Test rule ablation: run backtest with rule disabled and compare
# If Sharpe without the rule ≈ Sharpe with the rule → rule is not contributing
# If Sharpe without the rule drops significantly → rule is earning its keep
```

### For pairs trading (Strategy 6)

Pairs trading has the highest overfitting risk of all six strategies because:
1. You select which pairs to trade (selection bias)
2. The cointegration test can give false positives with short samples
3. Half-life is estimated on historical data that may not match the competition window

Additional safeguards:
- Only deploy pairs where the Engle-Granger p-value < 0.05 (not the 0.10 code default) when
  running the pre-competition screen
- Require half-life < 20 bars with 95% confidence interval entirely below 30 bars
- Do not update beta during the competition if it has been stable historically — frequent
  re-fitting of the hedge ratio is fitting to noise

---

## 10. What good iteration looks like in practice

The goal is not to maximize the backtest Sharpe. It is to find strategies where:
1. You understand the economic mechanism
2. The feature has positive IC before integration
3. Walk-forward CV shows stable, if modest, improvement
4. Performance is robust to ±20% parameter perturbation
5. DSR > 0.95 accounting for all trials run

A strategy that achieves Sharpe 0.9 through this process is worth more than a strategy
that achieves Sharpe 2.0 through 60 backtests on the same data. The 0.9 will likely be
closer to 0.7 live. The 2.0 will likely be negative.

**The calibration from the existing codebase:** The best confirmed result in `backtest_15m.py`
is Sharpe 1.141 on OOS 2024-2026 data. Any new strategy iteration should be benchmarked
against this. An improvement of +0.1 Sharpe after following this full process is meaningful.
An improvement of +0.5 Sharpe from a single feature addition should be treated with extreme
suspicion until the DSR check passes.

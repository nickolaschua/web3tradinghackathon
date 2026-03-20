# Gap 01: How to Implement `backtest_fold()` Correctly

## Why This Is a Gap
`backtest_fold()` is the critical unimplemented function that blocks all parameter optimization (Issue 08). Before writing it, I need to decide the implementation approach and understand the tradeoffs.

## What I Need to Know

### Option A: Use `backtesting.py` `Backtest` class
The project already has `backtesting/bt_validator.py` with a `MomentumStrategy` class. Can this be called directly from `backtest_fold()` with date-sliced data? What does `Backtest.run()` return and how do I extract Sharpe ratio from it?

### Option B: Manually simulate the strategy loop
Write a pandas-based loop that iterates over the test period bar by bar, applies `generate_signal()`, applies risk rules, tracks position P&L, then computes Sharpe. More flexible but more code.

### Option C: Use vectorbt's built-in portfolio simulation
vectorbt's `Portfolio.from_signals()` can compute Sharpe ratio directly from signal arrays. Faster but requires restructuring the strategy output.

## Research Needed
1. What metrics does `backtesting.py`'s `backtest.stats()` return — specifically the Sharpe ratio key name?
2. What is the correct way to slice a time-indexed DataFrame for train/test splits with TimeSeriesSplit?
3. How does Optuna's `trial.suggest_float()` interact with the objective function return value for maximization?
4. Should the objective be Sharpe ratio, Calmar ratio, or total return? For a competition maximizing ROI, Sharpe may under-optimize for absolute return.

## Priority
**Critical** — all parameter values are guesses without this. Must be resolved before competition parameters are finalized.

---

## Research Findings (Context7, 2026-03-12)

### backtesting.py Stats Keys

The exact Sharpe Ratio key is `'Sharpe Ratio'` (with a space):
```python
from backtesting import Backtest, Strategy

bt = Backtest(df, MomentumStrategy, cash=50000, commission=0.001)
stats = bt.run()
sharpe = stats['Sharpe Ratio']          # exact key name confirmed
equity_final = stats['Equity Final [$]']  # for total return objective
```

Walk-forward pattern using `bt.optimize()`:
```python
stats, heatmap = bt.optimize(
    n1=range(10, 30),
    n2=range(20, 50),
    maximize='Sharpe Ratio',
    constraint=lambda p: p.n1 < p.n2,
    return_heatmap=True
)
```

Walk-forward retraining pattern: subclass Strategy and retrain using
`self.data.df[-N_TRAIN:]` inside `next()` for a rolling-window approach.

### Optuna Objective Function Pattern

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    atr_mult = trial.suggest_float("atr_mult", 1.0, 4.0)
    rsi_threshold = trial.suggest_float("rsi_threshold", 55.0, 75.0)

    # Run backtest folds with TimeSeriesSplit
    sharpe_scores = []
    for train_idx, test_idx in tscv.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        score = run_single_fold(test_df, atr_mult, rsi_threshold)
        sharpe_scores.append(score)

    return np.mean(sharpe_scores)  # Must return a single float

study = optuna.create_study(direction="maximize")  # maximize Sharpe
study.optimize(objective, n_trials=100)
best_sharpe = study.best_trial.value
best_params = study.best_trial.params  # dict of param_name -> value
```

Log-scale suggestion for parameters that vary by orders of magnitude:
```python
atr_mult = trial.suggest_float("atr_mult", 0.5, 5.0, log=True)
```

### TimeSeriesSplit for Walk-Forward Folds

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    # NOTE: successive training sets are supersets (expanding window)
    # test set always comes AFTER train set — no look-ahead bias
```

Key property: each fold's test set is after the train set. Training sets grow
cumulatively (fold 1: 20% train, fold 2: 40% train, etc.) which is correct
for time-series data — we have more history as time progresses.

### Recommended Implementation: Option A with Optuna

**Decision**: Use Option A (backtesting.py) wrapped in Optuna for hyperparameter
search, with TimeSeriesSplit for walk-forward folds.

Reasons:
- `backtesting.py` is already in the project (`bt_validator.py`)
- `stats['Sharpe Ratio']` is confirmed and correct
- Competition ranks by total return, so use `'Equity Final [$]'` as objective
  OR use Sharpe for robustness and accept slightly lower absolute return
- `TimeSeriesSplit(n_splits=5)` ensures no look-ahead bias

**On objective choice**: For a competition maximizing ROI, consider using a
composite objective: `equity_final * (1 + sharpe / 10)` to avoid pure
return-chasing while still rewarding high absolute return.

### vectorbt Alternative for Speed

If backtesting.py is too slow for 100 Optuna trials × 5 folds = 500 runs:
```python
import vectorbt as vbt

pf = vbt.Portfolio.from_signals(price, entries, exits, freq='4h')
sharpe = pf.sharpe_ratio()          # annualized
max_dd = pf.max_drawdown()
total_return = pf.total_return()
```

vectorbt is ~100x faster than backtesting.py for signal-array-based backtests.
If run time is an issue, restructure `backtest_fold()` to use vectorbt.

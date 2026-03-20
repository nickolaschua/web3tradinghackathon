# Layer 4 — Backtesting Engine

## What This Layer Does

The Backtesting Engine validates trading strategies against historical data before any live capital is risked. It answers one question: does this strategy have genuine, out-of-sample predictive power, or does it just look good because the parameters were tuned to fit the historical data?

This layer runs entirely pre-hackathon on your local machine or a research EC2 instance. It is never deployed to the live trading EC2. Its only output is a set of validated strategy parameters written to `config.yaml` and documented evidence of out-of-sample performance.

**This layer is NOT deployed to production.** It is pure research infrastructure.

---

## What This Layer Is Trying to Achieve

1. Rapidly sweep thousands of parameter combinations to identify promising regions (vectorbt)
2. Deeply validate top candidates with realistic trade simulation and visual analysis (backtesting.py)
3. Prevent overfitting by optimising on rolling train windows and evaluating on held-out test windows (walk-forward)
4. Verify that the strategy survives all market regimes, not just bull markets (regime stress test)
5. Produce a single honest performance estimate on a completely untouched holdout set

---

## How It Contributes to the Bigger Picture

Without this layer, you are deploying a strategy you believe in but haven't verified. In the competition, every other team is essentially doing this — building something that looks good in their heads or in a quick backtest and deploying it. Your edge is that you have rigorous evidence your strategy works across market regimes and out-of-sample periods.

The most important output of this layer is not the Sharpe ratio. It is confidence — specifically, the confidence to let the bot run autonomously during the competition without second-guessing it based on short-term results.

---

## Files in This Layer

```
backtesting/
├── vbt_sweep.py        vectorbt rapid parameter sweeps
├── bt_validator.py     backtesting.py detailed single-strategy validation
├── walk_forward.py     Optuna + TimeSeriesSplit walk-forward optimisation
├── regime_stress.py    Performance isolated by labeled market regime
└── ic_analysis.py      (Documented in Layer 3)
```

---

## The Two-Phase Approach

### Phase 1: vectorbt — Rapid Parameter Sweeps

vectorbt runs vectorized NumPy operations that test thousands of parameter combinations simultaneously. A 5-year backtest with 400 parameter combinations runs in under 5 seconds. Use this for exploration: finding which parameter regions are promising before committing to expensive walk-forward optimisation.

```python
import vectorbt as vbt
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path

def run_momentum_sweep(pair: str = "BTCUSDT", interval: str = "4h"):
    """
    Sweep all combinations of fast/slow EMA + RSI threshold.
    Returns a DataFrame of results sorted by Sharpe ratio.
    """
    df = pd.read_parquet(f"data/parquet/{pair}_{interval}.parquet")
    price = df["close"]

    # Parameter grid
    fast_windows = [8, 10, 12, 15, 20]
    slow_windows = [30, 40, 50, 75, 100]
    rsi_buy_levels = [40, 45, 50]

    results = []

    for fast, slow, rsi_buy in product(fast_windows, slow_windows, rsi_buy_levels):
        if fast >= slow:
            continue

        fast_ma = vbt.MA.run(price, fast, short_name="fast")
        slow_ma = vbt.MA.run(price, slow, short_name="slow")
        rsi = vbt.RSI.run(price, 14)

        # Entry: fast MA crosses above slow MA AND RSI below threshold
        entries = fast_ma.ma_crossed_above(slow_ma) & (rsi.rsi < rsi_buy)
        # Exit: fast MA crosses below slow MA
        exits = fast_ma.ma_crossed_below(slow_ma)

        # CRITICAL: shift(1) is already applied in features.py
        # vectorbt uses bar-close execution by default — do not override

        pf = vbt.Portfolio.from_signals(
            price,
            entries,
            exits,
            init_cash=50_000,
            fees=0.00012,   # Taker fee
            slippage=0.0002,
            freq=interval,
        )

        results.append({
            "fast": fast,
            "slow": slow,
            "rsi_buy": rsi_buy,
            "total_return": pf.total_return(),
            "sharpe": pf.sharpe_ratio(),
            "max_drawdown": pf.max_drawdown(),
            "n_trades": pf.stats()["Total Trades"],
            "win_rate": pf.stats()["Win Rate [%]"] / 100,
        })

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(results_df.head(20).to_string())

    # Plot response surface for top parameters
    pivot = results_df.pivot_table(values="sharpe", index="fast", columns="slow", aggfunc="max")
    print("\nSharpe Response Surface (max across RSI thresholds):")
    print(pivot.round(2).to_string())

    return results_df
```

**What to look for in the response surface:**
- A flat plateau where multiple nearby parameter combinations all produce similar good Sharpe ratios — this indicates a robust region
- A sharp spike where only one specific combination performs well — this is overfit, avoid it
- If performance drops sharply when you move fast from 10→12 or slow from 50→55, the strategy is fragile

### Phase 2: backtesting.py — Detailed Validation

After identifying the top 3–5 parameter combinations from the sweep, run them through backtesting.py for detailed trade-level analysis with interactive Bokeh charts.

```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import sys
sys.path.insert(0, "..")
from data.features import compute_features

class MomentumStrategy(Strategy):
    # Parameters — set these from config.yaml values
    fast_period = 12
    slow_period = 50
    rsi_buy = 45
    atr_multiplier = 2.0
    rsi_exit = 70

    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index)
        self.fast_ma = self.I(lambda x: ta.ema(x, self.fast_period), close)
        self.slow_ma = self.I(lambda x: ta.ema(x, self.slow_period), close)
        self.rsi = self.I(lambda x: ta.rsi(x, 14), close)
        self.atr = self.I(lambda x: ta.atr(
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
            pd.Series(self.data.Close),
            14
        ), close)

    def next(self):
        if not self.position:
            # Entry condition
            if (crossover(self.fast_ma, self.slow_ma) and
                    self.rsi[-1] < self.rsi_buy):
                # Size: use 25% of equity per trade
                size = int(self.equity * 0.25 / self.data.Close[-1])
                if size > 0:
                    self.buy(size=size)
        else:
            # Exit conditions
            atr_stop = self.data.Close[-1] - self.atr_multiplier * self.atr[-1]
            if (crossover(self.slow_ma, self.fast_ma) or
                    self.rsi[-1] > self.rsi_exit or
                    self.data.Close[-1] < atr_stop):
                self.position.close()


def validate_strategy(pair: str = "BTCUSDT", interval: str = "4h"):
    df = pd.read_parquet(f"data/parquet/{pair}_{interval}.parquet")
    # backtesting.py requires specific column names
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Use 80% for validation, hold out 20%
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]

    bt = Backtest(
        train_df,
        MomentumStrategy,
        cash=50_000,
        commission=0.00012,
        exclusive_orders=True,
    )

    # Run with current config params
    stats = bt.run()
    print(stats)
    bt.plot(filename="strategy_validation.html")

    # Optimise (with caution — only to confirm neighbourhood, not to data-mine)
    opt_stats, heatmap = bt.optimize(
        fast_period=range(8, 25, 2),
        slow_period=range(30, 110, 10),
        rsi_buy=range(35, 55, 5),
        maximize="Sharpe Ratio",
        return_heatmap=True,
    )
    print("\nOptimised parameters:", opt_stats._strategy)
    return stats, opt_stats
```

---

## `backtesting/walk_forward.py`

Walk-forward optimisation is the core defence against overfitting. It simulates what actually happens in live trading: you fit a model on past data and use it to trade future data that was not used for fitting.

```python
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import sys
sys.path.insert(0, "..")

optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data(pair: str, interval: str) -> pd.DataFrame:
    return pd.read_parquet(f"data/parquet/{pair}_{interval}.parquet")

def backtest_fold(df: pd.DataFrame, fast: int, slow: int,
                  rsi_buy: float, atr_mult: float) -> dict:
    """Run a single fold backtest. Returns performance metrics."""
    from backtesting import Backtest
    # ... (uses MomentumStrategy from bt_validator.py)
    # Returns {"sharpe": float, "return": float, "drawdown": float, "n_trades": int}
    pass  # Implementation uses MomentumStrategy with injected params

def walk_forward_objective(trial: optuna.Trial, df: pd.DataFrame,
                            n_splits: int = 5) -> float:
    """
    Optuna objective: returns median Sharpe across all walk-forward folds.
    Using MEDIAN (not mean) reduces sensitivity to outlier folds.
    """
    fast = trial.suggest_int("fast_period", 6, 25)
    slow = trial.suggest_int("slow_period", 30, 150)
    rsi_buy = trial.suggest_float("rsi_buy", 30, 55, step=5)
    atr_mult = trial.suggest_float("atr_multiplier", 1.5, 3.5, step=0.5)

    if fast >= slow:
        return -999  # Invalid combination

    # Walk-forward splits with gap to prevent leakage
    # 6 months train (~1080 bars on 4H), 2 months test (~360 bars), gap=30 bars
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=30)
    fold_sharpes = []

    for train_idx, test_idx in tscv.split(df):
        # Only use the last 1080 bars of train (rolling window)
        train_start = max(0, len(train_idx) - 1080)
        train_df = df.iloc[train_idx[train_start:]]
        test_df = df.iloc[test_idx]

        # Require minimum bars in test fold for statistical validity
        if len(test_df) < 300:
            continue

        result = backtest_fold(test_df, fast, slow, rsi_buy, atr_mult)
        if result["n_trades"] < 10:
            fold_sharpes.append(-1)  # Penalise strategies with too few trades to evaluate
        else:
            fold_sharpes.append(result["sharpe"])

    if not fold_sharpes:
        return -999
    return float(np.median(fold_sharpes))


def run_walk_forward(pair: str = "BTCUSDT", interval: str = "4h",
                     n_trials: int = 150, n_splits: int = 5):
    df = load_data(pair, interval)

    # Hold out final 20% — NEVER touch this until final evaluation
    holdout_start = int(len(df) * 0.8)
    research_df = df.iloc[:holdout_start]
    holdout_df = df.iloc[holdout_start:]
    print(f"Research period: {research_df.index[0]} to {research_df.index[-1]}")
    print(f"Holdout period:  {holdout_df.index[0]} to {holdout_df.index[-1]} (LOCKED)")

    # Optimise on research data
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
    )
    study.optimize(
        lambda trial: walk_forward_objective(trial, research_df, n_splits),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    print(f"\nBest parameters: {best_params}")
    print(f"Best median Sharpe: {study.best_value:.3f}")

    # Parameter importance analysis
    importance = optuna.importance.get_param_importances(study)
    print("\nParameter importance:")
    for param, imp in importance.items():
        print(f"  {param}: {imp:.3f}")
        if imp > 0.6:
            print(f"  WARNING: {param} dominates performance — possible overfit")

    # Parameter stability check: do top trials agree?
    top_trials = study.trials_dataframe().nlargest(20, "value")
    for param in best_params:
        col = f"params_{param}"
        std = top_trials[col].std()
        mean = top_trials[col].mean()
        cv = std / (abs(mean) + 1e-10)
        if cv > 0.3:
            print(f"WARNING: {param} has high variance in top trials (CV={cv:.2f}) — fragile")

    # FINAL HOLDOUT EVALUATION — run exactly once
    print("\n" + "="*50)
    print("FINAL HOLDOUT EVALUATION (run once, never re-optimise after this)")
    holdout_result = backtest_fold(holdout_df, **best_params)
    print(f"Holdout Sharpe: {holdout_result['sharpe']:.3f}")
    print(f"Holdout Return: {holdout_result['return']:.1%}")
    print(f"Holdout Max DD: {holdout_result['drawdown']:.1%}")
    print(f"Holdout Trades: {holdout_result['n_trades']}")

    if holdout_result["sharpe"] < 0.5:
        print("CAUTION: Holdout Sharpe < 0.5. Parameters may be overfit.")
    elif holdout_result["sharpe"] > 1.0:
        print("Strong holdout performance. Proceed with confidence.")

    return best_params, holdout_result
```

---

## `backtesting/regime_stress.py`

A strategy that earns 80% in a bull market but loses 35% in a bear market will not win a competition that might span both. Regime stress testing forces you to see the full picture.

```python
import pandas as pd
from dataclasses import dataclass

@dataclass
class Regime:
    name: str
    start: str
    end: str
    type: str  # "bull", "bear", "crash", "sideways", "recovery"

# Manually labeled regimes from BTC price history
REGIMES = [
    Regime("Bull 2020-21",   "2020-10-01", "2021-11-10", "bull"),
    Regime("Crash 2021",     "2021-11-10", "2022-06-18", "crash"),
    Regime("Bear 2022",      "2022-06-18", "2022-11-20", "bear"),
    Regime("Recovery 2023",  "2023-01-01", "2024-01-01", "recovery"),
    Regime("Sideways 2024",  "2024-07-01", "2024-09-30", "sideways"),
    Regime("Bull 2024-25",   "2024-10-01", "2025-06-30", "bull"),
]

def run_regime_stress(params: dict, pair: str = "BTCUSDT", interval: str = "4h"):
    df = pd.read_parquet(f"data/parquet/{pair}_{interval}.parquet")
    results = []
    for regime in REGIMES:
        segment = df.loc[regime.start:regime.end]
        if len(segment) < 50:
            print(f"Skipping {regime.name}: insufficient data")
            continue
        result = backtest_fold(segment, **params)
        result["regime"] = regime.name
        result["type"] = regime.type
        results.append(result)
        print(f"{regime.name:25s} ({regime.type:10s}) | "
              f"Sharpe: {result['sharpe']:+.2f} | "
              f"Return: {result['return']:+.1%} | "
              f"Max DD: {result['drawdown']:.1%} | "
              f"Trades: {result['n_trades']}")

    results_df = pd.DataFrame(results)
    positive_regimes = (results_df["sharpe"] > 0).sum()
    total_regimes = len(results_df)
    print(f"\nPositive Sharpe in {positive_regimes}/{total_regimes} regimes")

    if results_df[results_df["type"] == "crash"]["drawdown"].max() > 0.40:
        print("FAIL: Max drawdown exceeds 40% in crash regime — strategy needs better stops")

    return results_df
```

---

## Pass/Fail Criteria Before Deploying

A strategy is considered ready for deployment when it meets ALL of the following:

| Criterion | Threshold |
|---|---|
| Walk-forward median Sharpe (5 folds) | > 0.8 |
| Holdout Sharpe (evaluated once) | > 0.5 |
| Positive Sharpe in regime stress test | ≥ 4 of 6 regimes |
| Max drawdown in crash regime | < 40% |
| Parameter stability (top 20 trials CV) | < 0.30 for all params |
| No single parameter dominating importance | < 60% importance score |
| Minimum trades per walk-forward fold | > 10 |

If any criterion fails, do not deploy. Go back to feature research or strategy logic and address the root cause. The holdout evaluation is the final gate — once you run it, you cannot re-optimise and re-run it. If the holdout fails, you must start a new optimisation run with a different approach.

---

## Failure Modes This Layer Prevents

| Failure | Prevention |
|---|---|
| Look-ahead bias in backtest | vectorbt bar-close execution + shift(1) in features.py |
| Overfitting to historical data | WFO with held-out test windows + parameter stability checks |
| Strategy only works in one regime | Regime stress test across 6 labeled market periods |
| Holdout set contamination | Holdout evaluated exactly once after all optimisation is complete |
| Overconfidence from single-fold optimisation | Median Sharpe across 5 folds required |
| Fragile parameter sensitivity | Response surface flatness check in vbt_sweep |

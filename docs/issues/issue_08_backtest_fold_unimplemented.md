# Issue 08: `backtest_fold()` is an Unimplemented Stub

## Layer
Layer 4 — Backtesting (`backtesting/walk_forward.py`, `backtesting/regime_stress.py`)

## Description
The `backtest_fold()` function in `backtesting/walk_forward.py` is a stub with only `pass` as its body. Both `walk_forward.py` (Optuna optimization) and `regime_stress.py` (regime-specific stress testing) import and call this function.

This means:
1. Walk-forward parameter optimization cannot run at all
2. Regime stress testing cannot run at all
3. All `config.yaml` parameter values are unvalidated — there is no way to know if the parameters are optimal without implementing this function

## Code Location
`backtesting/walk_forward.py` → `backtest_fold()` function body

## What It Needs to Do
`backtest_fold()` should:
1. Accept a train/test date split and a set of parameters (ATR mult, stop pct, regime thresholds)
2. Compute features on the test period data
3. Run the strategy logic over the test period
4. Return a Sharpe ratio (or other metric) for Optuna to optimize

The typical implementation uses either `backtesting.py`'s `Backtest` class or manually simulates the strategy loop.

## Impact
**Critical** — without this, the system has no validated parameters. All default values in `config.yaml` are guesses. The entire research phase (walk-forward optimization) described in the docs is blocked.

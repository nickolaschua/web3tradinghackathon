# Issue 09: vectorbt Sweep Bypasses `features.py` shift(1) — Potential Look-Ahead Bias

## Layer
Layer 4 — Backtesting (`backtesting/vbt_sweep.py`)

## Description
`vbt_sweep.py` computes moving averages and RSI directly from raw price data using vectorbt's built-in indicator functions (e.g. `vbt.MA.run()`, `vbt.RSI.run()`). It does NOT call `compute_features()` from `features.py`, which applies `shift(1)` to all indicators to prevent look-ahead bias.

The vectorbt sweep is therefore testing a version of the strategy that "sees" the current bar's indicator value when deciding whether to enter a trade on the same bar. The live system uses `shift(1)`, meaning it only sees the previous bar's indicator value.

The optimized parameters from `vbt_sweep.py` may not transfer correctly to the live strategy because they were found under different (look-ahead) conditions.

## Code Location
`backtesting/vbt_sweep.py` — indicator computation section

## Severity Note
The docs acknowledge this issue with a comment but mark it as "handled" — it is not actually handled. The comment is incorrect.

## Fix Required
Apply `shift(1)` to all indicator Series computed in `vbt_sweep.py` before using them as entry/exit signals, OR refactor vbt_sweep to call `compute_features()` directly and use the shifted feature columns.

## Impact
**High** — backtest performance will be overstated. Parameters tuned under look-ahead conditions will underperform in live trading.

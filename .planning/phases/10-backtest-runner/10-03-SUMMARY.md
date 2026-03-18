---
phase: 10-backtest-runner
plan: 03
subsystem: backtest
tags: [python, quantstats, stats-report, backtest-complete]

requires:
  - phase: 10-02
    provides: load_model(), run_backtest()

provides:
  - compute_stats_report(returns, closed_trades) -> dict
  - print_stats_report(stats, args) -> None
  - Complete scripts/backtest.py — runnable end-to-end backtest runner

affects: [11-xgboost-training]

tech-stack:
  added: [quantstats]
  patterns: [4H-annualization-PERIODS_4H-2190, trade-based-win-rate, quantstats-stats-api]

key-files:
  created: []
  modified: [scripts/backtest.py]

key-decisions:
  - "PERIODS_4H = 2190 (365.25 * 24 / 4) — correct annualization for 4H crypto bars (NOT 252)"
  - "Trade win rate computed manually from closed_trades (NOT qs.stats.win_rate which is bar-based)"
  - "Guard returns Series with returns[returns.index.notna()] — quantstats requirement to remove NaT entries"
  - "Always pass periods=PERIODS_4H to qs.stats.sharpe(), qs.stats.sortino(), qs.stats.cagr(), qs.stats.volatility()"

duration: 4min
completed: 2026-03-18
---

# Phase 10 Plan 03: Stats Report Summary

**Complete, runnable end-to-end backtest runner with comprehensive quantstats-based metrics report**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-18T14:00:00Z
- **Completed:** 2026-03-18T14:04:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `import quantstats as qs` to scripts/backtest.py
- Defined `PERIODS_4H = 2190` constant for correct 4H crypto bar annualization (365.25 * 24 / 4)
- Implemented `compute_stats_report(returns, closed_trades) -> dict`:
  - Computes total return, CAGR, Sharpe, Sortino, max drawdown, volatility via quantstats with periods=PERIODS_4H
  - Calculates trade win rate manually from closed_trades (NOT qs.stats.win_rate which is bar-based)
  - Tracks n_trades, avg_trade_pnl, best_trade, worst_trade, n_bars
  - Guards returns Series with `returns[returns.index.notna()]` to remove NaT entries
- Implemented `print_stats_report(stats, args) -> None`:
  - Formatted table output with 55-character separator
  - Displays all metrics: period bars, total return, CAGR, Sharpe, Sortino, max drawdown, volatility
  - Displays trade statistics: # trades, win rate, avg PnL, best/worst trade
  - Displays parameters: threshold, fee (bps)
- Wired complete `main()` pipeline:
  - Step 1: Load and prepare features with progress indicator
  - Step 2: Load pre-trained XGBoost model
  - Step 3: Run bar-by-bar simulation
  - Step 4: Compute and print stats
  - Step 5: Optional CSV output via --output flag
- All 8 command-line arguments functional and tested

## Task Commits

1. **Task 1: Add compute_stats_report()** - `24f9f61` (feat)
2. **Task 2: Wire main() + print_stats_report()** - Included in first commit (no separate change after)

## Files Created/Modified

- `scripts/backtest.py` — added compute_stats_report(), print_stats_report(), wired main()

## Decisions Made

- **PERIODS_4H = 2190**: Exact 4H annualization (365.25 × 24 / 4), not the default 252 which is for daily data. Critical for correct Sharpe and Sortino ratios on intraday crypto data.
- **Manual trade win rate calculation**: Computed as `sum(1 for t in closed_trades if t["pnl_pct"] > 0) / len(closed_trades)`. The quantstats.stats.win_rate() function computes bar-based win rate (% of bars with positive returns), not trade-based win rate.
- **Guard returns with notna()**: Before passing to quantstats functions, filter out NaT index entries. This is a quantstats requirement and prevents edge case errors.
- **Always pass periods parameter**: Never rely on default periods=252 for quantstats functions. Always explicitly pass periods=PERIODS_4H for consistency.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- **Encoding issue with arrow character (→)**: Windows cmd.exe uses cp1252 encoding, which doesn't support the Unicode arrow character. Resolved by replacing with ASCII "to" in progress output.

## Next Step

Phase 10 complete (3/3 plans). All core backtest functionality implemented and tested. Ready for Phase 11 (XGBoost Model Training).

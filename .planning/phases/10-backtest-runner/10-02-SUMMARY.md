---
phase: 10-backtest-runner
plan: 02
subsystem: backtest
tags: [python, xgboost, simulation, bar-by-bar, position-tracking]

requires:
  - phase: 10-01
    provides: scripts/backtest.py (CLI + prepare_features)

provides:
  - load_model(path) -> XGBClassifier
  - run_backtest(feat_df, model, threshold, fee_bps) -> (returns_series, closed_trades)

affects: [10-03]

tech-stack:
  added: []
  patterns: [bar-by-bar-simulation, three-way-signal-state-machine, feature_names_in_-column-ordering]

key-files:
  created: []
  modified: [scripts/backtest.py]

key-decisions:
  - "Use model.feature_names_in_ for column ordering — avoids silent column mismatch at inference time"
  - "Explicit three-way if/elif/else for BUY/SELL/HOLD — HOLD must never trigger exit"
  - "Pass pd.DataFrame (not numpy) to predict_proba() — preserves XGBoost column name validation"

patterns-established:
  - "fee applied symmetrically: entry_price * (1+fee_rate), exit_price * (1-fee_rate)"
  - "returns_series built with pd.DatetimeIndex for quantstats compatibility"

issues-created: []

duration: 5min
completed: 2026-03-18
---

# Phase 10 Plan 02: Backtest Engine Summary

**Bar-by-bar XGBoost simulation engine with 3-way signal state machine, fee-adjusted PnL tracking, and DatetimeIndex returns Series for quantstats**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-18T05:30:00Z
- **Completed:** 2026-03-18T05:35:37Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- `load_model()`: loads XGBClassifier .pkl, validates `predict_proba` and `feature_names_in_` presence
- `run_backtest()`: iterates bar-by-bar, generates BUY/SELL/HOLD signals via `predict_proba`, tracks FLAT/LONG position state, records closed trades with fee-adjusted entry/exit prices
- `main()` updated to call all three stages: `prepare_features` → `load_model` → `run_backtest` → summary print

## Task Commits

1. **Task 1+2: load_model + run_backtest + main()** - `906cf27` (feat)

## Files Created/Modified

- `scripts/backtest.py` — added `load_model()`, `run_backtest()`, updated `main()`

## Decisions Made

- Used `model.feature_names_in_` (not hardcoded column list) to guarantee column ordering matches training — silent mismatch would produce garbage predictions without error
- Explicit three-way `if/elif/else` for BUY/SELL/HOLD: `signal == "BUY"`, `elif signal == "SELL"`, `else` (HOLD) — HOLD is never checked in state machine conditions, only the assignment line
- Returns Series index built as `pd.DatetimeIndex(timestamps)` — required by quantstats in Phase 10-03

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## Next Step

Ready for 10-03-PLAN.md (stats report: Sharpe, Sortino, max drawdown, win rate, equity curve)

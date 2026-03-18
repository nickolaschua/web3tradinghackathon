---
phase: 10-backtest-runner
plan: 01
subsystem: backtest
tags: [python, xgboost, quantstats, feature-pipeline, backtest]

requires:
  - phase: 09-01
    provides: data/BTCUSDT_4h.parquet, ETHUSDT_4h.parquet, SOLUSDT_4h.parquet
  - phase: 04-03
    provides: compute_features, compute_cross_asset_features

provides:
  - scripts/backtest.py (partial — CLI args + prepare_features())
  - requirements.txt (updated with xgboost, quantstats)

affects: [10-02, 10-03, 11-xgboost-training]

tech-stack:
  added: [xgboost>=2.0, quantstats>=0.0.62]
  patterns: [feature-pipeline-reuse-in-backtest, UTC-date-filter-on-parquet]

key-files:
  created: [scripts/backtest.py]
  modified: [requirements.txt]

key-decisions: []
---

# Phase 10 Plan 01: Feature Prep Summary

**Installed backtest dependencies and built feature preparation pipeline — foundation for model inference.**

## Accomplishments

1. **Task 1: Dependencies (commit fcd0b4a)**
   - Added `xgboost>=2.0` and `quantstats>=0.0.62` to requirements.txt
   - Installed packages; verified imports with `python -c "import xgboost; import quantstats"` → "xgb 3.0.4 qs OK"

2. **Task 2: Backtest script (commit 6a9f096)**
   - Created `scripts/backtest.py` with argparse CLI (8 args: --model, --btc, --eth, --sol, --start, --end, --threshold, --fee-bps, --long-only, --output)
   - Implemented `prepare_features(btc_path, eth_path, sol_path, start=None, end=None)` function
   - Feature pipeline order preserved exactly: `compute_features() → compute_cross_asset_features() → dropna()`
   - Date filtering works: tested with --start "2024-01-01" --end "2024-12-31" (2191 bars returned from 9169 total)
   - Script tested with real Parquet files; produces feature matrix (9169 bars × 17 columns) with UTC DatetimeIndex

## Files Created/Modified

- `scripts/backtest.py` — CLI skeleton + prepare_features() function (149 lines)
- `requirements.txt` — added xgboost>=2.0, quantstats>=0.0.62

## Decisions Made

- Used `sys.path.insert()` in scripts/backtest.py to ensure bot package imports work when script is run standalone
- Used ASCII-safe output (× → x, → → to) to prevent UnicodeEncodeError on Windows

## Issues Encountered

- Initial import error when running script as standalone file (bot module not found); resolved by adding project root to sys.path
- Windows UnicodeEncodeError with special characters in output; fixed by using ASCII-safe formatting

## Next Step

Ready for 10-02-PLAN.md (model training pipeline)

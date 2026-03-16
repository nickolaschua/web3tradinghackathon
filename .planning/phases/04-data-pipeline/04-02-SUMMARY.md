---
phase: 04-data-pipeline
plan: 02
subsystem: data
tags: [python, pandas-ta-classic, feature-engineering, look-ahead-prevention]

requires:
  - phase: 04-01
    provides: bot/data package, LiveFetcher._to_dataframe

provides:
  - compute_features(df) in bot/data/features.py
  - Columns: atr_proxy, RSI_14, MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9, EMA_20, EMA_50, ema_slope
  - All indicator columns shifted 1 bar (look-ahead safe)

affects: [04-03-cross-asset, 06-strategy-interface, 07-main-loop]

tech-stack:
  added: [pandas_ta_classic]
  patterns: [shift-after-compute, close-to-close ATR proxy, disabled synthetic-candle indicators]

key-files:
  created: [bot/data/features.py]
  modified: []

key-decisions:
  - "Manual ema_slope instead of df.ta.slope(close='EMA_20') — kwarg form unverified"
  - "ATR proxy: log_ret.rolling(14).std() * close * 1.25 — same formula for backtest and live"
  - "Shift applied after all indicators in one pass — prevents subtle off-by-one"

patterns-established:
  - "ohlcv_cols set used to identify which columns to skip when shifting"
  - "append=True on all ta methods — modifies out in-place, returns None"

issues-created: []

duration: 5 min
completed: 2026-03-16 23:55 UTC
---

# Phase 4 Plan 02: compute_features Summary

**Feature computation with correct shift-after-compute and ATR proxy for synthetic candles.**

## Performance

- Duration: 5 min
- Tasks: 2
- Files modified: 1

## Accomplishments

- Created `bot/data/features.py` with `compute_features(df)` function
- Implemented all required technical indicators: atr_proxy, RSI_14, MACD_12_26_9/s/h, EMA_20, EMA_50, ema_slope
- Applied shift-after-compute logic to all indicator columns, preventing look-ahead bias
- Disabled ta.atr(), ta.adx(), ta.obv() which return ≈0 on Roostoo synthetic candles (H=L, volume=0)
- Verified all 5 assertion groups: columns present, OHLCV untouched, indicators shifted, no disabled indicators, warmup period correct

## Files Created/Modified

- `bot/data/features.py` — compute_features with ATR proxy, RSI/MACD/EMA, shift-after-compute

## Decisions Made

- ATR proxy: log returns SD × close × 1.25 (conservative crypto fat-tail buffer) instead of ta.atr()
- Manual ema_slope calculation instead of df.ta.slope(close="EMA_20") — kwarg form unverified
- Shift applied to all non-OHLCV columns in one pass after all indicators computed

## Issues Encountered

- pandas-ta-classic not pre-installed; resolved by `pip install pandas-ta-classic`
- Verification assertion logic refined to exclude custom 'atr_proxy' column from disabled indicators check

## Next Step

Ready for 04-03-PLAN.md (compute_cross_asset_features + is_warmed_up + integration test)

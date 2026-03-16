---
phase: 04-data-pipeline
plan: 01
subsystem: data
tags: [python, pandas, deque, live-fetcher, data-pipeline]

requires:
  - phase: 01-project-scaffolding
    provides: bot package structure (bot/__init__.py already exists)

provides:
  - LiveFetcher class with seed_dfs constructor
  - _seed_from_history (flat Parquet access, lowercase guard)
  - poll_ticker (synthetic candle H=L=O=C)
  - get_latest_price, get_candle_boundaries, _to_dataframe

affects: [04-02-compute-features, 04-03-cross-asset, 07-main-loop]

tech-stack:
  added: [pandas-ta-classic, collections.deque]
  patterns: [deque ring buffer, flat Parquet column access, lowercase column guard]

key-files:
  created: [bot/data/__init__.py, bot/data/live_fetcher.py]
  modified: []

key-decisions:
  - "seed_dfs keys use Roostoo format BTC/USD not Binance BTCUSDT — normalised at boundary"
  - "Always lowercase Parquet columns on load — Binance CSVs are capitalized, prevents KeyError startup failure"
  - "Synthetic candles H=L=O=C=LastPrice, vol=0 — intentional, not a bug"

patterns-established:
  - "deque(maxlen=500) per pair for O(1) streaming candle append"
  - "_to_dataframe converts buffer to UTC DatetimeIndex DataFrame"

issues-created: []

duration: 5 min
completed: 2026-03-16T23:42:00Z
commits:
  - a535e73

---

# Phase 4 Plan 01: LiveFetcher Core Summary

**LiveFetcher with deque ring buffers: seed from Parquet with lowercase column guard, poll_ticker synthetic candles (H=L=O=C=LastPrice, vol=0), get_latest_price and get_candle_boundaries accessors.**

## Performance

- Duration: 5 min
- Tasks: 2
- Files created: 2

## Accomplishments

- Created `bot/data` package with empty `__init__.py`
- Implemented LiveFetcher class with full deque-based ring buffer system
- Seed-from-history properly handles Binance Parquet with capital column names (enforces lowercase)
- poll_ticker appends synthetic candles with H=L=O=C=LastPrice pattern, vol=0
- get_latest_price and get_candle_boundaries provide required accessor interface for main.py
- _to_dataframe converts buffers to UTC-indexed DataFrames
- All verification assertions pass without error

## Files Created/Modified

- `bot/data/__init__.py` — Package init (empty)
- `bot/data/live_fetcher.py` — LiveFetcher (90 lines): __init__, _seed_from_history, poll_ticker, get_latest_price, get_candle_boundaries, _to_dataframe, __repr__

## Decisions Made

- Single commit strategy: all methods (Task 1 + Task 2) shipped in a535e73 since the plan specified complete implementation in one go
- Lowercase guard implemented at _seed_from_history entry point to catch Binance CSV capitalization issues at load time
- Synthetic candles use int(time.time()) for timestamp in poll_ticker to ensure current Unix seconds

## Issues Encountered

None. Implementation fully verified.

## Next Step

Ready for 04-02-PLAN.md (compute_features)

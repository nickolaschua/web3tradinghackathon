---
phase: 04-data-pipeline
plan: 03
subsystem: data
tags: [python, cross-asset, feature-engineering, integration-test, warmup]

requires:
  - phase: 04-01
    provides: LiveFetcher, _to_dataframe, get_latest_price, get_candle_boundaries
  - phase: 04-02
    provides: compute_features in bot/data/features.py

provides:
  - compute_cross_asset_features(btc_df, other_dfs) in bot/data/features.py
  - LiveFetcher.is_warmed_up(pair) — True when buffer >= 35 bars
  - LiveFetcher.get_feature_matrix(pair) — full clean feature matrix, correct ordering
  - Integration smoke test: seed → feature matrix pipeline verified end-to-end

affects: [06-strategy-interface, 07-main-loop]

tech-stack:
  added: []
  patterns: [cross-asset-before-dropna, lag-feature injection via reindex, warmup guard]

key-files:
  created: []
  modified: [bot/data/features.py, bot/data/live_fetcher.py]

key-decisions:
  - "Cross-asset features injected via .reindex(btc_df.index) — handles timestamp misalignment gracefully"
  - "get_feature_matrix returns empty DataFrame before warmup — caller must check is_warmed_up()"
  - "lag columns NOT double-shifted — lag is encoded by name (lag1 = yesterday), no extra shift(1)"

patterns-established:
  - "3-step pipeline: compute_features → compute_cross_asset_features → dropna (ORDER IS CRITICAL)"
  - "prefix extraction: pair.split('/')[0].lower() for column naming"

issues-created: []

duration: ~15 min
completed: 2026-03-16
---

# Phase 4 Plan 03: Cross-Asset Features + Integration Summary

**Cross-asset features (ETH/SOL log-return lags) fully integrated into the pipeline with correct ordering: compute_features → compute_cross_asset_features → dropna**

## Performance

- Duration: ~15 min
- Tasks: 3
- Files modified: 2

## Accomplishments

- Added `compute_cross_asset_features(btc_df, other_dfs)` to `bot/data/features.py` with proper lag-1/lag-2 log-return columns named `{symbol}_return_lag{n}`
- Added `is_warmed_up(pair)` to LiveFetcher — guards against trading before 35-bar MACD warmup
- Added `get_feature_matrix(pair)` to LiveFetcher — enforces critical 3-step ordering (compute_features → cross_asset → dropna)
- Validated full pipeline end-to-end: 60-bar synthetic 3-pair fixture → 10-row clean feature matrix with all 17 expected columns (5 OHLCV + 6 indicators + 4 cross-asset + shift wrapper + MACD variants)

## Files Created/Modified

- `bot/data/features.py` — Added `compute_cross_asset_features()` and `__all__` export list
- `bot/data/live_fetcher.py` — Added import of features module + `is_warmed_up()` and `get_feature_matrix()` methods

## Decisions Made

1. **Critical ordering enforcement**: `get_feature_matrix()` implements the exact 3-step sequence from plan (compute_features → cross_asset → dropna). If dropna precedes cross-asset injection, ETH/SOL columns don't exist yet and pandas silently removes all rows.

2. **Lag encoding by name, not shift**: Cross-asset lag columns are named `{symbol}_return_lag1` and `{symbol}_return_lag2` to represent yesterday's and two-days-ago returns. No additional `shift(1)` is applied — the lag is semantic.

3. **Warmup threshold**: 35 bars matches MACD(12,26,9) stabilization confirmed in RESEARCH.md. `is_warmed_up()` returns False for buffers < 35 bars.

4. **Reindex alignment**: Cross-asset columns are aligned to BTC's index via `.reindex(btc_df.index)`, which fills NaN where timestamps don't match — safe for non-overlapping trading hours.

## Issues Encountered

None. All three tasks passed verification on first attempt. Integration test confirmed non-empty feature matrix with all expected columns and zero NaN rows.

## Next Step

Phase 4 Data Pipeline complete (3/3 plans done). Ready for Phase 6 (Strategy Interface) or Phase 7 (Main Loop Orchestration) — both depend on the `bot/data` API established here.

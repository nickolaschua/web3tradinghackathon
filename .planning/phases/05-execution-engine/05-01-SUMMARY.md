---
phase: 05-execution-engine
plan: 01
subsystem: execution
tags: [python, pandas, numpy, ema, regime-detection]

requires:
  - phase: 01-project-scaffolding
    provides: bot package structure and namespaces
provides:
  - RegimeDetector class with 4H→daily resample, EMA(20)/EMA(50) crossover, hysteresis
  - RegimeState enum (BULL_TREND/SIDEWAYS/BEAR_TREND) with size_multiplier property
  - dump_state/load_state for crash-safe persistence
affects: [05-02-risk-manager, 05-03-order-manager, 07-main-loop]

tech-stack:
  added: []
  patterns: [bot.execution.regime import pattern, 4H-to-daily resample before EMA]

key-files:
  created: [bot/execution/regime.py, bot/execution/__init__.py]
  modified: []

key-decisions:
  - "Resample 4H→daily BEFORE computing EMA(20)/EMA(50) — EMAs calibrated for daily; raw 4H causes 6x faster regime flips"
  - "CONFIRMATION_BARS=2 hysteresis prevents thrashing at crossover boundaries"
  - "Default SIDEWAYS on cold start (conservative — 0.5x size, never 0.0 to avoid missed entries)"
  - "0.1% dead zone in EMA spread to suppress noise at crossover"

patterns-established:
  - "Warmup guard: check is_warmed_up() before any signal computation, return current state if not ready"
  - "dump_state/load_state pattern: serialize enum by .name, deserialize via RegimeState[name]"

issues-created: []

duration: 5min
completed: 2026-03-16
---

# Phase 5 Plan 01: RegimeDetector Summary

**RegimeDetector with 4H→daily EMA(20/50) crossover, 2-bar hysteresis, and crash-safe dump/load state**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-16T15:20:02Z
- **Completed:** 2026-03-16T15:25:11Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- RegimeState enum (BULL_TREND=1.0, SIDEWAYS=0.5, BEAR_TREND=0.0) with `size_multiplier` property consumed by RiskManager
- RegimeDetector correctly resamples 4H DataFrame to daily OHLCV before computing EMA(20)/EMA(50), preventing 6x inflated regime flip frequency
- Hysteresis with `CONFIRMATION_BARS=2` ensures regime only changes after 2 consecutive consistent crossover signals, eliminating thrashing
- dump_state/load_state round-trips `_current_regime`, `_pending_regime`, `_pending_count` for crash recovery

## Task Commits

1. **Task 1: RegimeState enum and skeleton** — `49586ac` (feat)
2. **Task 2: EMA crossover, hysteresis, dump/load** — `ee2afa9` (feat)

## Files Created/Modified

- `bot/execution/regime.py` — RegimeState enum + RegimeDetector: resample, EMA crossover, hysteresis update(), dump/load state
- `bot/execution/__init__.py` — Package init (empty)
- `bot/__init__.py` — Package init (empty, created by subagent as blocking fix)

## Decisions Made

- Resample 4H to daily before EMA — EMA span calibrated for daily data; raw 4H would require span ≈ 120/300 to replicate, and would flip 6x faster at boundaries
- 0.1% dead zone (`abs(e20-e50)/e50 < 0.001`) classifies near-crossover as SIDEWAYS to suppress noise
- Conservative SIDEWAYS default: 0.5x size means bot trades but at half position — avoids both "stuck flat" and "overconfident cold-start" failure modes

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- `bot/execution/regime.py` importable via `from bot.execution.regime import RegimeDetector, RegimeState`
- All 4 verification checks pass
- Ready for 05-02-PLAN.md (RiskManager)

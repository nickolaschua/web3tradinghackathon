---
phase: 06-strategy-interface
plan: "06-01"
subsystem: strategy
provides: [TradingSignal, SignalDirection, BaseStrategy]
affects: ["06-02", "07-main-loop-orchestration"]
key-files:
  created:
    - bot/strategy/base.py
    - bot/strategy/__init__.py
key-decisions:
  - "TradingSignal.pair is required positional field with no default (Issue 10 fix)"
  - "SignalDirection enum: BUY/SELL/HOLD"
  - "BaseStrategy ABC: generate_signal(pair, features) -> TradingSignal"
tech-stack:
  added: []
  patterns: ["dataclass with required positional field", "ABC strategy interface"]

# Metrics
duration: 3min
completed: 2026-03-16
---

# Phase 6 Plan 01: Strategy Base Types Summary

**TradingSignal dataclass with required pair field, SignalDirection enum, and BaseStrategy ABC wiring the strategy contract**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-16T15:32:33Z
- **Completed:** 2026-03-16T15:35:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created `SignalDirection` enum (BUY/SELL/HOLD) and `TradingSignal` dataclass with `pair` as a required positional field — `TradingSignal()` raises `TypeError` at construction time, preventing silent empty-pair order failures
- `__post_init__` validates all fields: non-empty pair, valid direction, size 0–1, confidence 0–1
- `BaseStrategy` ABC with `generate_signal(pair, features) -> TradingSignal` abstract method
- Exported public API from `bot/strategy/__init__.py`

## Task Commits

1. **Task 1: Create TradingSignal, SignalDirection, BaseStrategy** - `e014bdd` (feat)
2. **Task 2: Export public strategy API** - `f6dd2b2` (feat)

## Files Created/Modified

- `bot/strategy/base.py` — SignalDirection enum, TradingSignal dataclass, BaseStrategy ABC
- `bot/strategy/__init__.py` — Public API exports

## Decisions Made

- TradingSignal.pair has no default — `TradingSignal()` raises TypeError. This is the explicit fix for Issue 10 (empty pair causes silent order failures).
- Did not use `@dataclass(frozen=True)` — stubs may need mutable signals during development.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Step

Ready for 06-02-PLAN.md

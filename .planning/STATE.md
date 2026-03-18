# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-16)

**Core value:** A correctly-wired, crash-safe infrastructure that submits valid orders and never loses state — the user fills in alpha logic on top.
**Current focus:** PROJECT COMPLETE — bot live on EC2, Round 1 keys active, entering warmup period

## Current Position

Phase: 10 of 11 (Backtest Runner) - In progress
Plan: 1 of 3 in Phase 10 (10-01 COMPLETE)
Status: Dependencies (xgboost, quantstats) installed; scripts/backtest.py with CLI args + prepare_features() created. Feature pipeline validated against real Parquet files (9,169 bars × 17 features).
Last activity: 2026-03-18 — Completed 10-01 (xgboost>=2.0, quantstats>=0.0.62 added; scripts/backtest.py created with working prepare_features)

Progress: ██████████████████░░ 96% (20/21 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 18
- Average duration: 6 min
- Total execution time: ~120 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Project Scaffolding | 2/2 | 4 min | 2 min |
| 2. API Client & Rate Limiter | 2/2 | 11 min | 5.5 min |
| 3. Infrastructure Utilities | 1/1 | 5 min | 5 min |
| 4. Data Pipeline | 3/3 | 15 min | 5 min |
| 5. Execution Engine | 3/3 | 31 min | 10 min |
| 6. Strategy Interface | 2/2 | 5 min | 2.5 min |
| 7. Main Loop Orchestration | 2/2 | 32 min | 16 min |
| 8. EC2 Deployment | 2/2 | ~17 min | ~8.5 min |

**Recent Trend:**
- Last 5 plans: 08-02 (live), 08-01 (7 min), 07-02 (16 min), 07-01 (16 min), 05-01 (10 min)
- All phases complete

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

| Phase | Decision | Rationale |
|-------|----------|-----------|
| 04-03 | Cross-asset before dropna (order critical) | If dropna runs first, ETH/SOL columns don't exist yet; pandas silently removes all rows |
| 04-03 | is_warmed_up threshold: 35 bars | MACD(12,26,9) requires 35 bars to stabilize; below this, indicators unreliable |
| 04-03 | Lag encoding by name, not shift | Column names (lag1, lag2) are semantic; no additional shift(1) applied |
| 05-01 | Resample 4H→daily before EMA(20/50) | EMAs calibrated for daily data; raw 4H causes 6x faster regime flips |
| 05-01 | CONFIRMATION_BARS=2 hysteresis | Prevents thrashing at crossover boundaries |
| 07-02 | Epoch-based candle detection (int(t)//14400) | Monotonic; never triggers twice per 4H block; immune to clock drift |

### Deferred Issues

None.

### Blockers/Concerns

None. Project complete.

## Session Continuity

Last session: 2026-03-18
Stopped at: Phase 10 Plan 1 complete — backtest infrastructure (dependencies + feature prep) ready
Resume file: .planning/HANDOFF.md
Next: Phase 10 Plan 2 (10-02-PLAN.md) — model inference and position tracking logic

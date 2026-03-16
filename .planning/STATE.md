# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-16)

**Core value:** A correctly-wired, crash-safe infrastructure that submits valid orders and never loses state — the user fills in alpha logic on top.
**Current focus:** Phase 2 — API Client & Rate Limiter (1/2 complete)

## Current Position

Phase: 2 of 8 (API Client & Rate Limiter)
Plan: 1 of 2 in Phase 2 (02-01 COMPLETE)
Status: 02-01 complete (HMAC signing + all 6 endpoints); ready for 02-02
Last activity: 2026-03-16 — Completed 02-01-PLAN.md (RoostooClient all 6 endpoints)

Progress: █████████████████░ 87% (14/16 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 13
- Average duration: 6 min
- Total execution time: 95 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Project Scaffolding | 2/2 | 4 min | 2 min |
| 4. Data Pipeline | 3/3 | 15 min | 5 min |
| 5. Execution Engine | 3/3 | 31 min | 10 min |
| 6. Strategy Interface | 2/2 | 5 min | 2.5 min |
| 7. Main Loop Orchestration | 2/2 | 32 min | 16 min |
| 2. API Client & Rate Limiter | 1/2 | 8 min | 8 min |

**Recent Trend:**
- Last 5 plans: 06-02 (2 min), 01-01 (2 min), 04-02 (5 min), 04-03 (15 min), 02-01 (8 min)
- Trend: Wiring plans (reconciliation, main loop, cross-asset) take 8-12 min; scaffold plans 2-3 min

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

Phase 3 (infrastructure) pending — bot cannot actually run until TelegramAlerter and StateManager are implemented. Phase 2 is 1/2 complete (RoostooClient done, rate limiter pending).

## Session Continuity

Last session: 2026-03-16
Stopped at: Completed 02-01-PLAN.md (RoostooClient all 6 endpoints implemented)
Resume file: None
Next: Phase 2, Plan 02-02 — rate limiter (30/min sliding window) + exponential backoff (2s/4s/8s)

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-16)

**Core value:** A correctly-wired, crash-safe infrastructure that submits valid orders and never loses state — the user fills in alpha logic on top.
**Current focus:** Phase 7 Plan 02 — Main loop (7-step order, boundary-aligned sleep)

## Current Position

Phase: 7 of 8 (Main Loop Orchestration)
Plan: 1 of 2 in Phase 7 (COMPLETE)
Status: 07-01 complete; ready for 07-02
Last activity: 2026-03-16 — Completed 07-01-PLAN.md

Progress: ████████████░░░ 56% (9/16 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 6 min
- Total execution time: 51 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Project Scaffolding | 2/2 | 4 min | 2 min |
| 4. Data Pipeline | 1/3 | 5 min | 5 min |
| 5. Execution Engine | 3/3 | 31 min | 10 min |
| 6. Strategy Interface | 2/2 | 5 min | 2.5 min |
| 7. Main Loop Orchestration | 1/2 | 8 min | 8 min |

**Recent Trend:**
- Last 5 plans: 06-01 (3 min), 06-02 (2 min), 01-01 (2 min), 01-02 (2 min), 07-01 (8 min)
- Trend: Structural/scaffold plans are fast; wiring plans with reconciliation logic take ~8 min

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

| Phase | Decision | Rationale |
|-------|----------|-----------|
| 05-01 | Resample 4H→daily before EMA(20/50) | EMAs calibrated for daily data; raw 4H causes 6x faster regime flips |
| 05-01 | CONFIRMATION_BARS=2 hysteresis | Prevents thrashing at crossover boundaries |
| 05-01 | Default SIDEWAYS on cold start | Conservative: trades at 0.5x, never frozen at 0.0 |
| 07-01 | Interface stubs for Phase 2/3 deps | main.py imports cleanly before RoostooClient/TelegramAlerter/StateManager are fully implemented |
| 07-01 | Shutdown handler registered before reconciliation | Catches crashes during startup — state flushed even if reconciliation fails |
| 07-01 | WARN not ABORT on reconciliation discrepancy | Bot continues with live state taking precedence; operator notified via Telegram |

### Deferred Issues

None yet.

### Blockers/Concerns

Phase 7 Plan 02 depends on LiveFetcher (Phase 4 — not yet implemented). The main loop step (4) "if new 4H candle: features + signal + size + submit" requires `LiveFetcher.get_candle_boundaries()` and `compute_features()`. Plan 07-02 must either import LiveFetcher as-is (stub exists from 04-01) or add a guard.

## Session Continuity

Last session: 2026-03-16T16:00:00Z
Stopped at: Completed 07-01-PLAN.md (startup reconciliation + shutdown handler)
Resume file: None
Next: 07-02-PLAN.md (main loop — 7-step order, boundary-aligned sleep)

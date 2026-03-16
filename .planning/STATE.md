# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-16)

**Core value:** A correctly-wired, crash-safe infrastructure that submits valid orders and never loses state — the user fills in alpha logic on top.
**Current focus:** Phase 8 — EC2 Deployment

## Current Position

Phase: 8 of 8 (EC2 Deployment)
Plan: 0 of 2 in Phase 8 (NOT STARTED)
Status: Phase 7 complete (07-01 + 07-02); ready for Phase 8
Last activity: 2026-03-16 — Completed 07-02-PLAN.md (7-step main loop)

Progress: ██████████████░ 69% (11/16 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: 6 min
- Total execution time: 68 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Project Scaffolding | 2/2 | 4 min | 2 min |
| 4. Data Pipeline | 2/3 | 10 min | 5 min |
| 5. Execution Engine | 3/3 | 31 min | 10 min |
| 6. Strategy Interface | 2/2 | 5 min | 2.5 min |
| 7. Main Loop Orchestration | 2/2 | 20 min | 10 min |

**Recent Trend:**
- Last 5 plans: 06-01 (3 min), 06-02 (2 min), 01-01 (2 min), 04-02 (5 min), 07-02 (12 min)
- Trend: Wiring plans (reconciliation, main loop) take 8-12 min; scaffold plans 2-3 min

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
| 07-02 | Epoch-based candle detection (int(t)//14400) | Monotonic; never triggers twice per 4H block; immune to clock drift |
| 07-02 | features_cache: updated step 4, read step 3 | ATR from last computed bar available for stop checks; 2% price fallback on cold start |
| 07-02 | seed_dfs={} default | Binance Parquet seeding (Phase 4.03) not yet done; LiveFetcher warms from live ticks |

### Deferred Issues

None.

### Blockers/Concerns

Phase 4 Plan 03 (cross-asset features) still not done — main loop skips cross-asset columns silently. Phase 2 (API client) and Phase 3 (infrastructure) also pending — bot cannot actually run until those are implemented.

## Session Continuity

Last session: 2026-03-16T23:59:00Z
Stopped at: Completed 07-02-PLAN.md (7-step main loop with _run_one_cycle, _load_seed_data, wired main())
Resume file: None
Next: Phase 8 — EC2 deployment (08-01: EC2 setup, 08-02: deploy and verify)

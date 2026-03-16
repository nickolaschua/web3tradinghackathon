# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-16)

**Core value:** A correctly-wired, crash-safe infrastructure that submits valid orders and never loses state — the user fills in alpha logic on top.
**Current focus:** Phase 2 — API Client & Rate Limiter (2/2 complete)

## Current Position

Phase: 8 of 8 (EC2 Deployment) - IN PROGRESS
Plan: 1 of 2 in Phase 8 (08-01 COMPLETE)
Status: Phase 8 Plan 01 complete (deployment artifacts: systemd unit, bootstrap script, deploy script); ready for 08-02 EC2 provisioning
Last activity: 2026-03-17 — Completed 08-01-PLAN.md (systemd service, bootstrap.sh, deploy.sh)

Progress: ██████████████████░░ 95% (17/18 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 16
- Average duration: 6 min
- Total execution time: 103 min

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

**Recent Trend:**
- Last 5 plans: 07-02 (16 min), 05-01 (10 min), 04-03 (15 min), 07-01 (16 min), 08-01 (7 min)
- Trend: Wiring plans (main loop, cross-asset) take 10-16 min; deployment artifacts (systemd + scripts) faster at ~7 min

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

Phase 3 (infrastructure) pending — bot cannot actually run until TelegramAlerter and StateManager are implemented. Phase 2 now complete.

## Session Continuity

Last session: 2026-03-17
Stopped at: Completed 08-01-PLAN.md (systemd service unit, bootstrap.sh, deploy.sh)
Resume file: None
Next: Phase 8 Plan 02 — EC2 provisioning and deployment execution (launch instance, smoke test with testing keys)

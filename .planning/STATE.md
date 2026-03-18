# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-16)

**Core value:** A correctly-wired, crash-safe infrastructure that submits valid orders and never loses state — the user fills in alpha logic on top.
**Current focus:** PROJECT COMPLETE — bot live on EC2, Round 1 keys active, entering warmup period

## Current Position

Phase: 11 of 11 (XGBoost Model Training) - COMPLETE
Plan: 2 of 2 in Phase 11 (11-02 COMPLETE)
Status: Final model training + save complete. Trained XGBClassifier on 2020-2023 train+val data (4,330 bars), evaluated on 2024 held-out test (4,833 bars). Test AP=0.392, F1=0.170. Model saved to models/xgb_btc_4h.pkl with pickle, verified round-trip OK. Ready for backtest.py and live trading.
Last activity: 2026-03-18 — Completed 11-02 (final model training and save)

Progress: ████████████████████ 100% (21/21 plans complete)

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
Stopped at: Phase 11 Plan 1 complete — label engineering and walk-forward CV validation complete
Resume file: .planning/HANDOFF.md
Next: Phase 11 Plan 2 (11-02-PLAN.md) — final model training + save to models/xgb_btc_4h.pkl

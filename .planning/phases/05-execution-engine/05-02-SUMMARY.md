---
phase: 05-execution-engine
plan: 02
subsystem: execution
provides: [RiskManager, RiskDecision, SizingResult, StopCheckResult]
affects: [05-03, 07-main-loop]
tech-stack:
  added: []
  patterns: [tiered-circuit-breaker pattern, dump/load state pattern]
key-files: [bot/execution/risk.py]
key-decisions: [tiered-cb-thresholds, atr-trailing-stop, hard-stop-8pct]
---

# Phase 5 Plan 02: RiskManager Summary

**Complete ATR-based stop-loss system with tiered circuit breaker and crash-safe state persistence.**

## Accomplishments

- **RiskManager class** with 10 public methods + full config support via constructor
- **ATR trailing stops** that move only upward, never downward; hard 8% stop loss as floor
- **Tiered circuit breaker** with 4 size tiers: 1.0x / 0.5x / 0.25x / 0.0x at <10% / 10-20% / 20-30% / ≥30% drawdown
- **Position sizing gate system** blocking new positions by: CB active, BEAR regime, max positions, insufficient balance, zero combined multiplier
- **Crash-safe state persistence** via dump_state() / load_state() for trailing_stops, entry_prices, portfolio_hwm, circuit_breaker_active
- **Stop check cycle** independent of signal generation (called every 60s poll)

## Files Created/Modified

- `bot/execution/risk.py` - RiskManager with tiered CB, ATR stops, sizing, dump/load state

## Decisions Made

- **tiered-cb-thresholds**: 4-tier system (0%/25%/50%/100% size) rather than binary on/off allows recovery trading during drawdown phases
- **atr-trailing-stop**: ATR × 2.0 multiplier with hard 8% floor ensures stops tighten as volatility rises; unidirectional (up only) prevents stop whipsaws
- **hard-stop-8pct**: 8% hard stop from entry price provides catastrophic-loss safeguard while allowing normal volatility

## Issues Encountered

None. Implementation followed spec exactly.

## Verification Results

- [PASS] All imports (RiskManager, RiskDecision, SizingResult, StopCheckResult)
- [PASS] Tiered CB returns correct multipliers: 1.0x / 0.5x / 0.25x / 0.0x at specified drawdown thresholds
- [PASS] check_stops() correctly triggers hard stop at 8% loss
- [PASS] ATR trailing stop moves upward only, never down
- [PASS] size_new_position() blocks by: circuit breaker, zero regime, max positions
- [PASS] dump_state() / load_state() round-trips all 4 state fields
- [PASS] Circuit breaker activates at 30%, deactivates on HWM recovery

## Next Step

Ready for 05-03-PLAN.md (OrderManager with fill_price None check, position resync, state persistence)

# Phase 5: Execution Engine - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<vision>
## How This Should Work

The Execution Engine is the hard gate between a strategy signal and a live order. It has three components working in sequence:

1. **RegimeDetector** — reads BTC 4H data, resamples to daily, computes EMA(20)/EMA(50) crossover to classify the market as BULL_TREND, SIDEWAYS, or BEAR_TREND. Uses hysteresis (2 confirmation bars) to prevent thrashing at regime boundaries. Outputs a `RegimeState` with a size multiplier (1.0x / 0.5x / 0.0x) that the rest of the system respects.

2. **RiskManager** — acts as a hard gate on every new position and every existing position. Checks ATR trailing stops and hard percentage stops every 60s independently of signal generation. Enforces a tiered circuit breaker (30%/20%/10% drawdown thresholds mapping to 0%/25%/50%/100% size allowed). Must persist its full state (trailing_stops, entry_prices, portfolio_hwm, circuit_breaker_active) so a crash/restart loses nothing.

3. **OrderManager** — handles pre-flight validation (precision rounding, MiniOrder check, balance check), submits orders via RoostooClient, parses Success field explicitly, tracks order lifecycle, reconciles local state against exchange every 5 minutes, and restores position state on startup. fill_price None check must use explicit `is None` (not `or`). The `_resync_from_exchange()` must write back to `self._positions`.

These three work together: RegimeDetector sets the multiplier → RiskManager sizes and gates → OrderManager executes and tracks.

</vision>

<essential>
## What Must Be Nailed

- **Crash recovery**: `dump_state()` / `load_state()` on both RiskManager and OrderManager must be complete and correct — the circuit breaker and trailing stops must survive a process restart
- **Stop checks run every cycle**: Stop checking is NOT tied to signal generation — it runs every 60s poll independently
- **Tiered circuit breaker**: Three thresholds (30%/20%/10% drawdown) → four size tiers (0%/25%/50%/100%) — not a binary on/off
- **fill_price None check**: Must use explicit `if fill_price is None` — using `or` will treat 0.0 as falsy and corrupt the fill price
- **RegimeDetector resamples 4H→daily**: Do not run EMA on raw 4H data — resample first; requires 300+ 4H bars warmup
- **`_resync_from_exchange()` writes back to `self._positions`**: Without this, the OMS doesn't recover from discrepancies

</essential>

<boundaries>
## What's Out of Scope

- Strategy logic (momentum/mean_reversion) — that's Phase 6
- Main loop wiring — that's Phase 7
- The StateManager atomic write implementation — that's Phase 3 (persistence/state.py)
- Data pipeline / feature computation — that's Phase 4
- API client / rate limiter — that's Phase 2
- Any UI, reporting, or dashboards
- LIMIT order logic beyond the existing MARKET order default (cancel_order is a stub only)

</boundaries>

<specifics>
## Specific Ideas

- **RegimeDetector**: `_pending_regime` + `_pending_count` hysteresis pattern with `CONFIRMATION_BARS = 2`; default to SIDEWAYS on cold start (conservative)
- **RiskManager tiered CB**:
  - drawdown < 10% → full size (1.0×)
  - drawdown 10–20% → half size (0.5×)
  - drawdown 20–30% → quarter size (0.25×)
  - drawdown ≥ 30% → no new positions (0.0×)
- **OrderManager.cancel_order(order_id)**: stub only — call `/v3/cancel_order`, log result, return bool. No complex lifecycle tracking needed.
- **Pre-flight validation**: precision rounding via `exchange_info.round_quantity()` and `round_price()` before every submission
- **Reconciliation**: every 5 min auto, plus `force=True` on startup
- **`get_all_positions()`**: returns `dict[str, Position]` — needed by healthcheck and state persistence

</specifics>

<notes>
## Additional Context

Agent 3 codes against the interfaces of Agent 1 (bot.api.client, bot.api.rate_limiter, bot.api.exchange_info, bot.monitoring.telegram) without needing those files to exist yet. All interfaces are fully spec'd in the docs.

Key issue from issue_15: `RiskManager.dump_state()` and `RiskManager.load_state()` were missing from the original spec — these must be added. Similarly, `OrderManager.get_all_positions()`, `dump_state()`, and `load_state()` must all be present.

The tiered circuit breaker in the ROADMAP differs from the binary circuit breaker in `docs/07_layer6_risk_management.md`. The ROADMAP version (tiered: 30%/20%/10% → 0%/25%/50%/100%) takes precedence as the more detailed and recent spec.

Competition deadline: Round 1 starts Mar 21, 8:00 PM. Infrastructure must be deployed and tested before then.

</notes>

---

*Phase: 05-execution-engine*
*Context gathered: 2026-03-16*

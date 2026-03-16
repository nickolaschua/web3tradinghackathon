# Summary: 07-02 — Main Loop Orchestration

**Completed:** 2026-03-16
**Duration:** ~12 min
**Commit:** 2ec5ed8

## What Was Built

Replaced the `# TODO(07-02): main loop goes here` placeholder with a fully-wired
`while True:` main loop in `main.py`, implementing all 7 steps in the mandatory
order specified in PROJECT.md Issue 08.

### New Functions Added to main.py

**`_run_one_cycle(**kwargs)`** — 7-step loop body:
1. **Step 1 — Poll ticker**: `client.get_ticker(pair)` for all `feature_pairs`; appends synthetic candles via `live_fetcher.poll_ticker()`. Prices dict built for downstream use.
2. **Step 2 — Balance + CB**: `client.get_balance()` → `risk_manager.check_circuit_breaker(total_usd)`. Logs warning if CB active.
3. **Step 3 — Stop checks**: For each open position, uses ATR from `features_cache` (2% price fallback). Calls `risk_manager.check_stops()` → `order_manager.close_position()` + Telegram alert on trigger.
4. **Step 4 — New 4H candle**: Detects new epoch via `int(time.time()) // _CANDLE_4H_SECONDS > last_signal_epoch`. On new epoch: `live_fetcher._to_dataframe()` → `compute_features().dropna()` → `regime_detector.update()` → `strategy.generate_signal()` → `risk_manager.size_new_position()` → `order_manager.place_order()` (BUY) or `order_manager.close_position()` (SELL).
5. **Step 5 — Write state**: `state_manager.write({risk_manager, order_manager, last_signal_epoch, timestamp})`
6. **Step 6 — Heartbeat**: `logger.info("Heartbeat: positions=%d cb_active=%s 4h_epoch=%d ...")`
7. **Step 7 — Sleep**: `time.sleep(max(0.0, 60.0 - (time.time() % 60.0)))` — boundary-aligned

**`_load_seed_data(config)`** — Tries 3 filename conventions + glob for Binance Parquet files per feature pair. Missing files skipped silently (cold start).

**`main()` updated** — Now instantiates `RegimeDetector`, loads seed data, creates `LiveFetcher(seed_dfs=seed_dfs)`, selects `MomentumStrategy`, restores `loop_state["last_signal_epoch"]` from persisted state, then runs the `while True:` loop with `KeyboardInterrupt` + catch-all exception handlers.

### Constants Added
- `_CANDLE_4H_SECONDS = 4 * 3600` — 4H period in seconds
- `_BINANCE_TO_ROOSTOO` — Binance→Roostoo symbol map (used in `_load_seed_data`)

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Epoch-based candle detection (`int(t) // 14400`) | Monotonic; immune to clock drift; never triggers twice in same 4H block |
| `features_cache` updated in step 4, read in step 3 | Provides ATR from last computed bar; 2% price fallback on cold start |
| `open_positions` passed as `{pair: usd_value}` dict | Matches `size_new_position()` internal `.values()` iteration |
| `strategy.generate_signal(pair, features_df)` passes full DataFrame | Matches `BaseStrategy.generate_signal(pair, features: pd.DataFrame)` signature |
| `seed_dfs={}` default | Binance Parquet seeding (Phase 4.03) not yet implemented; LiveFetcher warms up from live ticks |
| 10s retry on unhandled exception | Prevents crash-loop tight spin; Telegram WARN sent |
| `loop_state["last_signal_epoch"]` restored from `state_manager.read()` | Prevents duplicate signal on restart within same 4H epoch |

## Verification

```
python -m py_compile main.py  →  OK
AST check (7 required functions)  →  PASSED
7-step comments (Step 1–Step 7)  →  PASSED
while True loop + _run_one_cycle call  →  PASSED
KeyboardInterrupt handler + 10s retry  →  PASSED
last_signal_epoch restored from state  →  PASSED
LiveFetcher + RegimeDetector instantiated  →  PASSED
max(0.0, 60.0 - (time.time() % 60.0)) sleep  →  PASSED
current_price * 0.02 ATR fallback  →  PASSED
maybe_reconcile() call  →  PASSED
```

## Phase 7 Status

Both plans complete:
- ✅ 07-01: startup_reconciliation + shutdown handler
- ✅ 07-02: 7-step main loop + _load_seed_data + wired main()

Phase 7 is **COMPLETE**. Ready for Phase 8: EC2 Deployment.

# Gap 08: Main Loop Architecture and Component Wiring

## Why This Is a Gap
`main.py` doesn't exist and is the most complex file to write correctly (Issue 19). Before writing it, I need to understand the exact sequencing and error handling at the orchestration level.

## What I Need to Know

1. **What is the correct order of operations in each 60-second loop cycle?**
   - Current best guess from docs:
     1. Sync time (check timestamp offset)
     2. Check circuit breaker (get live balance)
     3. For each open position: check stops, exit if breached
     4. Check candle boundary — if new 4H bar: compute features, generate signal
     5. If signal is LONG and no position: size and submit BUY
     6. Write state.json
     7. Run healthcheck
     8. Sleep until next 60s boundary
   - Is this the correct sequence? Should stop checks come before or after signal generation?

2. **How should exceptions be handled at the main loop level?**
   - API timeout on `/v3/ticker`: skip this cycle, try again next poll?
   - API timeout on order submission: is the order submitted or not? Need to check pending orders
   - Feature computation failure: skip signal generation but keep monitoring stops?
   - State write failure: log and continue (non-fatal) or halt?

3. **How should the component dictionary (`c = {}`) be structured?**
   - Layer 10 docs reference `c["fetcher"]`, `c["client"]`, `c["oms"]`, etc.
   - What is the initialization order? (`client` must exist before `oms`, `fetcher` before `feature_engine`, etc.)

4. **How should the sleep be implemented?**
   - `time.sleep(60)`: simple but can drift
   - Sleep until next 60-second boundary: `sleep(60 - (time.time() % 60))` — keeps in sync with minute boundaries
   - The docs recommend the boundary-aligned approach for candle detection

## Priority
**Critical** — `main.py` cannot be written without resolving this.

---

## Research Findings (Architecture + Systems Design, 2026-03-12)

### Correct Operation Order per 60-Second Cycle

After reviewing all 13 layers and their dependencies:

```
1. GET /v3/ticker → update LiveFetcher buffer (synthetic candle)
2. GET /v3/balance → check circuit breaker against HWM
3. If CB hard-stop triggered → skip steps 4-6, write state, sleep
4. For each open position → check_stops() using latest price
   → If stop breached: submit SELL order, record_exit(), clear position
5. Check if new 4H candle boundary crossed:
   → If YES: compute_features() → generate_signal()
   → If signal is LONG and no open position: size_new_position() → submit BUY
6. Write state.json (atomic: write tmp, rename)
7. Log heartbeat (for healthcheck scraping)
8. Sleep until next 60s boundary: time.sleep(60 - (time.time() % 60))
```

**Stop checks MUST come before signal generation.** Reason: if the current price
has breached a stop, we should exit first before reconsidering entry.
Processing the opposite order (signal first, then stop) could generate a new
entry signal at the same time we should be exiting.

### Component Initialization Order

```python
def build_components(config: dict) -> dict:
    c = {}
    # Layer 1: API client (no dependencies)
    c["client"] = RoostooClient(
        api_key=config["api_key"],
        api_secret=config["api_secret"],
        base_url=config["base_url"]
    )
    # Layer 2: Live data fetcher (depends on client)
    c["fetcher"] = LiveFetcher(client=c["client"], symbol=config["symbol"])
    # Layer 3: Feature engine (depends on fetcher)
    c["feature_engine"] = FeatureEngine(config=config)
    # Layer 4: Risk manager (depends on config only)
    c["risk"] = RiskManager(config=config)
    # Layer 5: OMS (depends on client + risk)
    c["oms"] = OrderManagementSystem(client=c["client"], risk=c["risk"])
    # Layer 6: State manager (depends on all)
    c["state"] = StateManager(components=c, config=config)
    return c
```

### Exception Handling Matrix

| Event | Action | Rationale |
|-------|--------|-----------|
| `/v3/ticker` timeout | Skip cycle, log WARNING, sleep | Non-fatal; next poll recovers |
| `/v3/balance` timeout | Skip CB check, continue with cached value | Better than halting |
| Stop-check price unavailable | Use last known price with stale flag | Keeps stops active |
| Order submission timeout | Log ERROR, check `/v3/query-order` next cycle | Order may have executed |
| `compute_features()` failure | Skip signal generation, log ERROR | Stops still monitored |
| `state.json` write failure | Log ERROR, continue | Non-fatal; state on restart |
| Unhandled exception | Log CRITICAL, allow systemd restart | systemd RestartSec=10 handles |

### Boundary-Aligned Sleep

```python
import time

def sleep_to_next_minute():
    """Sleep until the next 60-second wall-clock boundary."""
    now = time.time()
    sleep_time = 60.0 - (now % 60.0)
    if sleep_time < 1.0:
        sleep_time += 60.0  # Avoid waking up 0.1s into a minute
    time.sleep(sleep_time)
```

This ensures candle detection via `(ts // 14400) != last_candle_id` works
reliably because the bot polls at predictable minute boundaries.

### 4H Candle Boundary Detection

```python
CANDLE_SECONDS = 4 * 60 * 60  # 14400

def is_new_candle(ts: float, last_candle_ts: float) -> bool:
    return int(ts // CANDLE_SECONDS) != int(last_candle_ts // CANDLE_SECONDS)
```

Track `last_candle_ts` in the main loop state. Update it when a new candle is
detected and features are computed.

### Startup Reconciliation

On startup, before entering the main loop:
```python
def startup_reconciliation(c: dict) -> None:
    """Reconcile saved state with live exchange state."""
    saved = c["state"].load()
    live_balance = c["client"].get_balance()
    live_orders = c["client"].get_open_orders()

    if saved.has_open_position and not live_orders:
        # Position was closed externally — clear local state
        c["risk"].clear_position()
        logger.warning("Startup: position cleared externally, resetting state")

    # Initialize HWM from saved state or live balance
    hwm = max(saved.hwm or 0.0, live_balance["USD"]["total"])
    c["risk"].initialize_hwm(hwm)
```

This addresses Issue 16 (startup reconciliation unimplemented).

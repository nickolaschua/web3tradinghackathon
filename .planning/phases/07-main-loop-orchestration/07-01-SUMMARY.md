---
phase: 07-main-loop-orchestration
plan: "07-01"
subsystem: main-loop
tags: [python, startup, reconciliation, sigterm, sigint, state-persistence]
requires:
  - phase: 05-01
    provides: RegimeDetector
  - phase: 05-02
    provides: RiskManager (dump/load_state)
  - phase: 05-03
    provides: OrderManager (dump/load_state, get_all_positions)
  - phase: 06-02
    provides: MomentumStrategy, MeanReversionStrategy
provides:
  - RoostooClient stub (bot/api/client.py)
  - TelegramAlerter stub (bot/monitoring/telegram.py)
  - StateManager stub (bot/persistence/state_manager.py)
  - main.py with startup_reconciliation(), _register_shutdown_handler(), _setup_logging(), _load_config(), main()
affects: ["07-02-main-loop"]
tech-stack:
  added: []
  patterns: ["atomic state persistence", "never-raise alerting stub", "SIGTERM/SIGINT flush-before-exit"]
key-files:
  created:
    - bot/api/client.py
    - bot/monitoring/telegram.py
    - bot/persistence/state_manager.py
    - main.py
  modified:
    - bot/api/__init__.py
    - bot/monitoring/__init__.py
    - bot/persistence/__init__.py
key-decisions:
  - "Interface stubs define correct method signatures so main.py imports cleanly before Phases 2/3 are implemented"
  - "StateManager stub uses os.replace() (atomic on POSIX) — will be reviewed in Phase 3"
  - "Shutdown handler registered BEFORE startup_reconciliation() to flush state even on crashes during startup"
  - "Discrepancies during reconciliation WARN via Telegram but do NOT abort startup"
issues-created: []

# Metrics
duration: 8min
completed: 2026-03-16
---

# Phase 7 Plan 01: Startup Reconciliation and Shutdown Handler Summary

**Crash-safe startup sequence: state loading, exchange reconciliation, Telegram WARN on discrepancy, and SIGTERM/SIGINT flush-before-exit**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-16T15:50:00Z
- **Completed:** 2026-03-16T15:58:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Created `RoostooClient` stub (`bot/api/client.py`): defines full public interface (`get_balance`, `get_open_orders`, `get_ticker`, `place_order`, `pending_count`, `cancel_order`) — raises `NotImplementedError` until Phase 2 implements it
- Created `TelegramAlerter` stub (`bot/monitoring/telegram.py`): `send()` never raises (wrapped in try/except) — logs at INFO level during development; Phase 3 adds real HTTP calls
- Created `StateManager` stub (`bot/persistence/state_manager.py`): functional atomic write (`os.replace()`) + read with error recovery — fully usable now, Phase 3 may harden further
- Updated `bot/api/__init__.py`, `bot/monitoring/__init__.py`, `bot/persistence/__init__.py` to export their classes
- Created `main.py` with five functions:
  - `_setup_logging()`: rotating file handler (10 MB × 10) + stdout at INFO
  - `_load_config()`: loads `bot/config/config.yaml` via PyYAML
  - `startup_reconciliation()`: loads persisted RiskManager + OrderManager state, fetches live balance + open orders, reconciles positions and portfolio value, sends Telegram WARN on any discrepancy
  - `_register_shutdown_handler()`: registers SIGTERM + SIGINT handlers that call `dump_state()` on both managers, write state atomically, then `sys.exit(0)`
  - `main()`: wires up all components, registers shutdown handler, runs reconciliation, leaves TODO for 07-02 main loop

## Task Commits

1. **Task 1: Interface stubs** — `cf92885` (feat)
2. **Task 2: main.py startup sequence** — `1ceaf27` (feat)

## Files Created/Modified

- `bot/api/client.py` — RoostooClient stub
- `bot/monitoring/telegram.py` — TelegramAlerter stub (never-raise)
- `bot/persistence/state_manager.py` — StateManager stub (atomic write)
- `bot/api/__init__.py` — exports RoostooClient
- `bot/monitoring/__init__.py` — exports TelegramAlerter
- `bot/persistence/__init__.py` — exports StateManager
- `main.py` — startup_reconciliation(), _register_shutdown_handler(), boot sequence

## Decisions Made

- **Stub-first approach**: main.py imports cleanly against stubs; Phase 2/3 replace stubs without touching main.py
- **Shutdown handler before reconciliation**: If reconciliation crashes, SIGTERM still flushes whatever state was loaded
- **WARN not ABORT on discrepancy**: Operator gets notified but bot continues with live state taking precedence

## Issues Encountered

None.

## Next Step

07-02: Main loop — 7-step loop in correct order, boundary-aligned 60s sleep, full integration with LiveFetcher and strategy signals

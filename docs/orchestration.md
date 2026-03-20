# Multi-Agent Build Orchestration

## Overview

The trading bot infrastructure is built using 4 parallel Claude Code agents, coordinated through the GSD framework. Three agents work simultaneously on independent packages; the fourth waits to wire everything together.

All strategy files (`bot/strategy/momentum.py`, `bot/strategy/mean_reversion.py`) are clean stubs — the infrastructure is built, the interfaces are defined, and the user fills in the alpha logic.

---

## Repository Structure (FAQ Q37)

```
trading-bot/
├── bot/
│   ├── __init__.py
│   ├── api/             # Agent 1 — Roostoo API client, rate limiter, exchange info
│   ├── data/            # Agent 2 — downloader, gap detector, live fetcher, features
│   ├── execution/       # Agent 3 — regime detector, risk manager, order manager
│   ├── monitoring/      # Agent 1 — logger, telegram, healthcheck
│   ├── persistence/     # Agent 1 — state manager
│   ├── strategy/        # Agent 4 — base class + stubs
│   └── config/
│       └── config.yaml  # Agent 4 — all tunable parameters
├── tests/               # Agent 4 — empty skeleton, user fills in
├── logs/                # Auto-created at runtime by logger
├── main.py              # Agent 4 — entry point, imports from bot.*
├── Dockerfile           # Agent 4
├── README.md            # Agent 4
├── requirements.txt     # Agent 1
└── .env.example         # Agent 1
```

All imports within the bot use the `bot.*` namespace: e.g. `from bot.api.client import RoostooClient`, `from bot.strategy.base import TradingSignal`. `main.py` sits at the project root and is the entry point invoked by systemd (`python main.py`).

---

## Competition Timeline

| Event | Date/Time |
|---|---|
| Round 1 starts | Mar 21, 8:00 PM |
| First trade deadline | Mar 22, 8:00 PM (must trade within 24h of start) |
| Round 2 starts | Apr 4, 8:00 PM |
| End | System auto-liquidates — no manual action needed |

Infrastructure must be deployed, tested with testing keys, and bot confirmed running **before Mar 21**. Round 1 API keys replace testing keys at competition start. Finalist round keys are issued separately.

---

## Setup (Main Window — Do This First)

Before opening any agent windows:

1. **`/gsd:new-project`** — Creates `PROJECT.md` capturing the full spec. All 4 agents read this for shared context.
2. **`/gsd:create-roadmap`** — Defines 4 milestones, one per agent.

---

## Agent 1 — Foundation Layer

**Owns:** `bot/api/`, `bot/monitoring/`, `bot/persistence/`

**Files to create:**
```
bot/__init__.py
bot/api/__init__.py
bot/api/client.py            # RoostooClient with HMAC SHA256 signing + exponential backoff retry
bot/api/rate_limiter.py      # TradingRateLimiter — trade cooldown (65s) AND global 30/min call budget
bot/api/exchange_info.py     # ExchangeInfoCache, PairInfo dataclass
bot/monitoring/__init__.py
bot/monitoring/logger.py     # setup_logging, JsonFormatter, TradeLogger
bot/monitoring/telegram.py   # TelegramAlerter
bot/monitoring/healthcheck.py # HealthChecker (memory/disk/CPU + heartbeat)
bot/persistence/__init__.py
bot/persistence/state.py     # StateManager with atomic write (tmp -> rename)
requirements.txt
.env.example
```

**Why independent:** Zero imports from any other custom package. Pure stdlib + `requests` + `psutil`.

**Critical constraints from FAQ:**
- **Rate limit is 30 calls/minute for ALL API calls** — not just trades. `get_ticker`, `get_balance`, `query_order` all count. The rate limiter must maintain a sliding window counter for every outbound request, not only trade submissions. The existing 65s trade cooldown is still correct, but is a separate concern layered on top.
- **Exponential backoff is mandatory** (FAQ Q19) — the client must retry failed requests with backoff before raising. Recommended: 3 retries, delays of 1s / 2s / 4s.
- **`.env.example` needs dual key set documentation** — testing keys (`ROOSTOO_API_KEY_TEST`, `ROOSTOO_SECRET_TEST`) and competition round 1 keys (`ROOSTOO_API_KEY`, `ROOSTOO_SECRET`). Finalist keys are a third set issued later.
- **EC2 is Sydney region** (ap-southeast-2) — note this in deployment comments if relevant.

**Docs to read:**
- `docs/01_layer0_external_dependencies.md`
- `docs/02_layer1_api_client.md`
- `docs/10_layer9_monitoring.md`
- `docs/09_layer8_state_persistence.md`
- `docs/faq_and_answers.md` (Q19, Q20, Q21, Q28)

**ECC skills to invoke:**
- `everything-claude-code:python-patterns` — type hints, dataclasses, EAFP error handling
- `everything-claude-code:security-review` — HMAC signing, no hardcoded secrets, API key handling via env vars

**GSD commands:**
```
/gsd:discuss-phase     (provide doc refs + file list above)
/gsd:plan-phase        (creates PLAN.md)
/gsd:execute-plan      (builds the files)
/gsd:verify-work       (self-check before handoff)
```

---

## Agent 2 — Data Pipeline

**Owns:** `bot/data/`

**Files to create:**
```
bot/data/__init__.py
bot/data/downloader.py      # BinanceDataDumper wrapper, saves Parquet to data/parquet/
bot/data/gap_detector.py    # Forward-fills gaps with volume=0
bot/data/live_fetcher.py    # LiveFetcher — poll_and_update, get_dataframe, is_warmed_up
                             # FIX: constructor accepts dict[str, pd.DataFrame] not single df
                             # FIX: add get_latest_price(pair) method
                             # FIX: add get_candle_boundaries() -> dict[str, int] method
bot/data/features.py        # compute_features(), compute_cross_asset_features()
                             # All features shifted by 1 bar (no look-ahead bias)
```

**Why independent:** Only imports `pandas`, `pandas-ta`, `binance_historical_data`. Zero cross-package imports.

**Docs to read:**
- `docs/03_layer2_data_pipeline.md`
- `docs/04_layer3_feature_engineering.md`
- `docs/issues/issue_15_undefined_methods_across_layers.md`

**Key fixes this agent must implement:**
- `LiveFetcher.__init__` in docs takes `seed_df` (single DataFrame) but `main.py` passes `seed_dfs` (dict keyed by pair symbol). Fix: accept `dict[str, pd.DataFrame]`.
- `get_latest_price(pair: str) -> float` — return last close from live buffer for the given pair.
- `get_candle_boundaries() -> dict[str, int]` — return dict of last candle close timestamps per pair, used by main loop to detect new candle formation.

**ECC skills to invoke:**
- `everything-claude-code:python-patterns`
- `agents/python-reviewer.md` — invoke after writing for a self-review pass

**GSD commands:**
```
/gsd:discuss-phase
/gsd:plan-phase
/gsd:execute-plan
/gsd:verify-work
```

---

## Agent 3 — Execution Layer

**Owns:** `bot/execution/`

**Files to create:**
```
bot/execution/__init__.py
bot/execution/regime.py      # RegimeDetector with hysteresis, RegimeState, MarketRegime enum
bot/execution/risk.py        # RiskManager — check_stops, check_circuit_breaker,
                              # size_new_position, record_entry, record_exit, initialize_hwm
                              # FIX: add dump_state() method
                              # FIX: add load_state() method
bot/execution/order_manager.py # OrderManager — submit_buy, submit_sell,
                                # _pre_flight_and_submit, reconcile, get_all_positions,
                                # dump_state, load_state
                                # ADD: cancel_order(order_id) stub — /v3/cancel_order exists
```

**Why independent:** The interfaces it depends on (`bot/api/`, `bot/monitoring/`) are fully spec'd in the docs. It codes against those contracts without needing the live files. No file conflicts with Agents 1 or 2.

**Docs to read:**
- `docs/06_layer5_strategy_engine.md` (for RegimeDetector and RegimeState)
- `docs/07_layer6_risk_management.md`
- `docs/08_layer7_order_management.md`
- `docs/09_layer8_state_persistence.md` (for state schema reference)
- `docs/issues/issue_15_undefined_methods_across_layers.md`

**Key fixes this agent must implement:**
- `RiskManager.dump_state() -> dict` — serialize trailing_stops, entry_prices, portfolio_hwm, circuit_breaker_active.
- `RiskManager.load_state(state: dict)` — restore all of the above from state.json on startup.

**ECC skills to invoke:**
- `everything-claude-code:python-patterns`
- `agents/architect.md` — invoke for the risk -> OMS -> state dependency chain to ensure correct interface design

**GSD commands:**
```
/gsd:discuss-phase
/gsd:plan-phase
/gsd:execute-plan
/gsd:verify-work
```

---

## Agent 4 — Strategy Stubs + Orchestration

**Owns:** `bot/strategy/`, `bot/config/`, `main.py`, `Dockerfile`, `README.md`, `tests/`

**Files to create:**
```
bot/strategy/__init__.py
bot/strategy/base.py          # BaseStrategy ABC, TradingSignal dataclass
                               # FIX: TradingSignal.pair has no default (must be required)
bot/strategy/momentum.py      # STUB — correct interface, empty signal logic for user to fill
bot/strategy/mean_reversion.py # STUB — correct interface, empty signal logic for user to fill
bot/config/config.yaml        # All tunable parameters (pairs, timeframe, risk limits, etc.)
main.py                        # Full orchestration — load_config, build_components,
                               # startup_sequence, select_strategy, run_main_loop
Dockerfile                     # Container definition for EC2 deployment
README.md                      # Setup instructions, architecture overview, strategy explanation
tests/__init__.py              # Empty skeleton — user fills in tests
```

**Sequencing — IMPORTANT:**
- Start `bot/strategy/base.py` and `bot/config/config.yaml` immediately (no cross-deps).
- **Hold on `main.py`** until Agents 1, 2, and 3 have finished. `main.py` imports from all packages and must wire them together correctly.

**Docs to read:**
- `docs/06_layer5_strategy_engine.md`
- `docs/11_layer10_orchestration.md`
- `docs/00_project_overview.md`
- `docs/issues/issue_10_signal_pair_always_empty.md`
- `docs/issues/issue_19_main_py_and_config_not_written.md`

**Key fixes this agent must implement:**
- `TradingSignal.pair` — remove the `""` default, make it a required positional field so strategies are forced to set it.
- `startup_sequence` in `main.py` — properly call `startup_reconciliation()` from Layer 7 docs rather than leaving it as a placeholder comment.
- Strategy stubs must have fully correct method signatures (`generate_signal`, `get_regime_compatibility`) with clear docstrings explaining what the user needs to implement.

**ECC skills to invoke:**
- `everything-claude-code:python-patterns`
- `agents/architect.md` — for the `main.py` wiring of all 6 component groups

**GSD commands:**
```
/gsd:discuss-phase
/gsd:plan-phase
/gsd:execute-plan      (bot/strategy/ + bot/config/config.yaml first; main.py after Agents 1-3 finish)
/gsd:verify-work
```

---

## Integration Step (Back in Main Window)

After all 4 agents report done:

1. **`/gsd:verify-work`** — smoke test: can Python import `main` without errors?
2. **`agents/python-reviewer.md`** — run across the full codebase (`git diff`) for a unified review pass
3. **`everything-claude-code:verification-loop`** — syntax, imports, no hardcoded secrets

---

## Known Issues Summary

All agents must be aware of the following cross-cutting issues. They are documented in detail under `docs/issues/`:

| Issue | Owner | Fix |
|---|---|---|
| `TradingSignal.pair` defaults to `""` | Agent 4 | Make it required (no default) |
| `RiskManager.dump_state()` missing | Agent 3 | Implement from state schema |
| `RiskManager.load_state()` missing | Agent 3 | Implement from state schema |
| `LiveFetcher.get_latest_price()` missing | Agent 2 | Implement from live buffer |
| `LiveFetcher.get_candle_boundaries()` missing | Agent 2 | Implement returning dict per pair |
| `LiveFetcher` takes `seed_df` but `main.py` passes `seed_dfs` dict | Agent 2 | Fix constructor signature |
| `main.py` startup reconciliation is placeholder | Agent 4 | Wire in `startup_reconciliation()` from Layer 7 |
| Rate limiter only tracks trade cooldown, not global 30/min budget | Agent 1 | Add sliding window counter to client for all outbound calls |
| `.env.example` has single key set | Agent 1 | Document testing keys and competition keys separately |

---

## File Ownership Map

No two agents touch the same file. Conflicts are impossible.

```
Agent 1:  bot/api/  bot/monitoring/  bot/persistence/  bot/__init__.py  requirements.txt  .env.example
Agent 2:  bot/data/
Agent 3:  bot/execution/
Agent 4:  bot/strategy/  bot/config/  main.py  Dockerfile  README.md  tests/
```

---

## Data Storage Reference

For any agent that needs to understand where data lives:

| Data | Format | Location |
|---|---|---|
| Historical OHLCV | Parquet | `bot/data/parquet/BTCUSDT_1h.parquet` etc. |
| Live price buffer | In-memory DataFrame | RAM only (LiveFetcher) |
| Bot operational state | JSON | `state.json` (atomic write each cycle) |
| Trade history | JSON Lines | `logs/trades.log` (append-only) |
| System logs | JSON Lines | `logs/bot.log` (rotating 10MB x 10) |

# Roadmap: Roostoo Quant Trading Bot

## Overview

Build an autonomous BTC/USD swing trading bot from scratch: project scaffolding and config first, then the API client with rate limiting, infrastructure utilities (alerting + state persistence), the data pipeline with synthetic-candle-safe features, the execution engine with tiered risk management, strategy stubs for the user's alpha, main loop orchestration with crash-safe startup/shutdown, and finally EC2 deployment with systemd under the competition deadline.

## Domain Expertise

None

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Project Scaffolding** - Package layout, requirements.txt (pandas-ta-classic), config.yaml, .env.example, __init__.py files
- [x] **Phase 2: API Client & Rate Limiter** - RoostooClient HMAC SHA256 signing, exponential backoff (3 retries, 2s/4s/8s), global 30/min sliding-window rate limiter, 65s trade cooldown
- [x] **Phase 3: Infrastructure Utilities** - TelegramAlerter (real HTTP, never-raise, level emojis), StateManager (atomic write with backup/fallback, versioned metadata, JSON serializer)
- [x] **Phase 4: Data Pipeline** - LiveFetcher (seed from Binance Parquet flat columns, live poll), close-to-close ATR proxy, no ADX/OBV, cross-asset features before dropna, shift(1) for look-ahead prevention, 35-bar warm-up
- [x] **Phase 5: Execution Engine** - RegimeDetector (resample 4H->daily, EMA20/50), RiskManager (tiered CB + dump/load state), OrderManager (fill_price None check, resync, dump/load state, cancel_order stub)
- [x] **Phase 6: Strategy Interface** - TradingSignal (pair required, no default), momentum.py stub, mean_reversion.py stub with correct generate_signal signatures
- [x] **Phase 7: Main Loop Orchestration** - main.py with startup reconciliation, 7-step loop order, boundary-aligned sleep, SIGTERM/SIGINT shutdown handler
- [x] **Phase 8: EC2 Deployment** - systemd service, chrony Amazon Time Sync, deploy script, smoke test with testing keys
- [x] **Phase 9: Historical Data Download** - one-time script to download BTC/ETH/SOL 4H Binance candles as Parquet, seeding LiveFetcher on startup
- [x] **Phase 10: Backtest Runner + Feature Prep** - bar-by-bar simulation accepting pre-trained XGBoost model, full feature pipeline, comprehensive stats report
- [ ] **Phase 11: XGBoost Model Training** - label engineering (N=6, τ=0.00015) and walk-forward CV complete (Mean AP=0.415, Mean F1=0.363); final training + save .pkl pending

## Phase Details

### Phase 1: Project Scaffolding
**Goal**: Establish the full package directory structure, install dependencies, and create all config/env skeleton files so Agents 2-4 have a valid namespace to import from day one.
**Depends on**: Nothing (first phase)
**Research**: Unlikely (standard Python packaging, all specs in PROJECT.md)
**Plans**: 2 plans

Plans:
- [x] 01-01: Package skeleton — `bot/` with `api/`, `data/`, `execution/`, `strategy/`, `monitoring/`, `persistence/`, `config/` subdirs + all `__init__.py` files
- [x] 01-02: Config and dependency files — `requirements.txt` (pandas-ta-classic, not pandas-ta), `bot/config/config.yaml`, `.env.example` (three key sets), `.gitignore`

### Phase 2: API Client & Rate Limiter
**Goal**: Deliver a fully-signed RoostooClient with HMAC SHA256 (alphabetical param sort, form-encoded POST body), 3-retry exponential backoff, and a global sliding-window rate limiter (30/min, lock released before sleep, 65s trade cooldown) that covers ALL outbound calls.
**Depends on**: Phase 1
**Research**: Unlikely (HMAC spec and rate-limit behaviour fully specified in PROJECT.md + FAQ Q19)
**Plans**: 2 plans

Plans:
- [x] 02-01: `RoostooClient` — HMAC signing, all endpoints (`/v3/ticker`, `/v3/balance`, `/v3/place_order`, `/v3/pending_count`, `/v3/cancel_order`), `pending_count` Success=false not a WARNING (check TotalPending)
- [x] 02-02: Rate limiter — global 30/min sliding window in `bot/api/rate_limiter.py`, lock released BEFORE `time.sleep()`, 65s trade cooldown layer; exponential backoff (2s/4s/8s) integrated into client

### Phase 3: Infrastructure Utilities
**Goal**: TelegramAlerter that can never propagate an exception, and StateManager with guaranteed atomic writes (write to `.tmp` + `os.rename`) so a crash mid-write never corrupts state.
**Depends on**: Phase 1
**Research**: Unlikely (standard patterns)
**Plans**: 1 plan

Plans:
- [x] 03-01: `bot/monitoring/telegram.py` (TelegramAlerter real HTTP, level emojis, _enabled guard) + `bot/persistence/state_manager.py` (atomic write with backup, versioned metadata, JSON serializer)

### Phase 4: Data Pipeline
**Goal**: LiveFetcher that seeds from Binance Parquet flat columns (not multi-index), polls Roostoo ticker every 60s to build synthetic candles, computes features with the close-to-close ATR proxy (no ATR/ADX/OBV), calls cross-asset features BEFORE dropna, shifts all indicators 1 bar, and reports is_warmed_up() after 35 bars.
**Depends on**: Phase 1
**Research**: Likely (pandas-ta-classic API surface; Binance Parquet flat column names confirmation)
**Research topics**: pandas-ta-classic available indicators and import path; confirm `open/high/low/close/volume` flat columns in Binance 4H Parquet; `log_returns.rolling(14).std() * close * 1.25` ATR proxy validation
**Plans**: 3 plans

Plans:
- [x] 04-01: `LiveFetcher.__init__` with `seed_dfs: dict[str, pd.DataFrame]`, `_seed_from_history()` using flat Parquet column access, live ticker poll, synthetic candle append, `get_latest_price(pair)`, `get_candle_boundaries()`
- [x] 04-02: `compute_features()` — close-to-close ATR proxy, RSI, MACD, EMA slope; OBV disabled; ADX removed; all columns shifted by 1 bar
- [x] 04-03: `compute_cross_asset_features()` called BEFORE `dropna()`; `is_warmed_up()` threshold 35 bars; integration test seeding from fixture Parquet

### Phase 5: Execution Engine
**Goal**: RegimeDetector that resamples 4H->daily before EMA crossover (preserving calibration, needing 300+ 4H bar warmup), RiskManager with tiered circuit breaker (30%/20%/10% thresholds) + full dump/load state for crash recovery, and OrderManager with correct fill_price None check, position resync from exchange, and state persistence.
**Depends on**: Phase 2
**Research**: Unlikely (all thresholds and logic fully specified in PROJECT.md)
**Plans**: 3 plans

Plans:
- [x] 05-01: `RegimeDetector` in `bot/execution/regime.py` — resample 4H DataFrame to daily, EMA(20)/EMA(50) crossover, bullish/bearish/neutral enum, requires 300+ 4H bars warmup
- [x] 05-02: `RiskManager` in `bot/execution/risk.py` — ATR-based stop-loss, trailing stops, tiered circuit breaker (0%/25%/50%/100% size at 30%/20%/10%/<10% drawdown), `dump_state()` / `load_state()` for `trailing_stops`, `entry_prices`, `portfolio_hwm`, `circuit_breaker_active`
- [x] 05-03: `OrderManager` in `bot/execution/order_manager.py` — place/track orders, explicit `is None` fill_price check (not `or`), `_resync_from_exchange()` writes back to `self._positions`, `get_all_positions()`, `dump_state()` / `load_state()`, `cancel_order(order_id)` stub

### Phase 6: Strategy Interface
**Goal**: TradingSignal dataclass with `pair` as required positional field (no default), base strategy interface, and momentum/mean-reversion stubs with correct `generate_signal(pair, features)` signatures and docstrings for the user to fill in alpha logic.
**Depends on**: Phase 4
**Research**: Unlikely (internal interface design only)
**Plans**: 2 plans

Plans:
- [ ] 06-01: `bot/strategy/base.py` — `TradingSignal` dataclass (pair required, no default; direction, size, confidence fields), `BaseStrategy` ABC with `generate_signal(pair, features) -> TradingSignal`
- [ ] 06-02: `bot/strategy/momentum.py` + `bot/strategy/mean_reversion.py` — stubs returning neutral signal with docstrings describing expected alpha logic

### Phase 7: Main Loop Orchestration
**Goal**: Fully-wired main.py using `bot.*` namespace throughout: startup reconciliation (balance + open orders + Telegram WARN on discrepancy), 7-step main loop in correct order, boundary-aligned 60s sleep, SIGTERM/SIGINT handler that calls state_manager.write() before exit.
**Depends on**: Phases 3, 5, 6
**Research**: Unlikely (loop order and reconciliation spec fully defined in PROJECT.md)
**Plans**: 2 plans

Plans:
- [x] 07-01: `startup_reconciliation()` — `client.get_balance()` + `client.get_open_orders()`, reconcile vs state, Telegram WARN on discrepancy; RiskManager + OrderManager `load_state()` on startup; SIGTERM/SIGINT handler registered
- [x] 07-02: Main loop — (1) poll ticker (2) balance + CB check (3) check stops per position (4) if new 4H candle: features + signal + size + submit (5) write state.json (6) heartbeat log (7) `time.sleep(60.0 - (time.time() % 60.0))` + seed loader + wired main()

### Phase 8: EC2 Deployment
**Goal**: Provision t3.medium in ap-southeast-2 (via HackathonBotTemplate), configure systemd service (ExecStart=python main.py, RestartSec=10), sync time with chrony → Amazon Time Sync (169.254.169.123), deploy bot, smoke-test with testing keys, confirm first trade executes before competition window closes.
**Depends on**: Phase 7
**Research**: Likely (EC2 + Python 3.11 systemd service setup; chrony Amazon Time Sync config)
**Research topics**: systemd Python venv service ExecStart pattern; chrony refclock for 169.254.169.123; EC2 ap-southeast-2 t3.medium (HackathonBotTemplate) bootstrap with Python 3.11; testing key smoke-test checklist
**Connection**: Session Manager only (SSH blocked in hackathon environment); login via SSO portal https://d-906625dad1.awsapps.com/start
**Plans**: 2 plans

Plans:
- [x] 08-01: Deployment artifacts — write systemd unit file, bootstrap.sh (AL2023 python3.11), deploy.sh (git pull + restart)
- [x] 08-02: EC2 provisioning & smoke test — launched t3.medium ap-southeast-2 (HackathonBotTemplate), connected via Session Manager, cloned repo + venv (Python 3.9), populated .env with Round 1 keys, bot running in tmux, verified startup_reconciliation ($50k USD balance), state.json written, Round 1 keys active

### Phase 9: Historical Data Download
**Goal**: One-time local script that downloads BTC/USD, ETH/USD, SOL/USD 4H candles from Binance and saves them as Parquet files in the `data/` directory, so the bot can seed its LiveFetcher instantly on startup instead of cold-starting.
**Depends on**: Phase 4 (LiveFetcher expects flat OHLCV Parquet columns)
**Research**: Likely (Binance public API or binance-historical-data package; rate limits; Parquet column naming to match LiveFetcher expectations)
**Plans**: TBD

Plans:
- [ ] 09-01: `scripts/download_data.py` — download BTC/ETH/SOL 4H candles from Binance, save as `data/BTCUSDT_4h.parquet`, `data/ETHUSDT_4h.parquet`, `data/SOLUSDT_4h.parquet` with flat open/high/low/close/volume columns

### Phase 10: Backtest Runner + Feature Prep
**Goal**: A backtest runner (`scripts/backtest.py`) that loads historical Parquet data, runs the full feature pipeline (existing `bot/data/features.py`), accepts a pre-trained XGBoost model (.pkl), simulates bar-by-bar execution, and outputs a comprehensive stats report: total return, Sharpe, Sortino, max drawdown, win rate, avg trade PnL, and any other metrics specified in the hackathon problem statement.
**Depends on**: Phase 9 (needs Parquet data), Phase 4 (reuses feature pipeline)
**Research**: Likely (XGBoost model input/output format; Sortino and other metric formulas; bar-by-bar simulation without lookahead)
**Plans**: TBD

Plans:
- [ ] 10-01: Feature prep pipeline — load Parquet → clean → run `compute_features()` + `compute_cross_asset_features()` → produce aligned feature matrix ready for model input
- [ ] 10-02: Backtest engine — bar-by-bar simulation with XGBoost model, position tracking, fill simulation, PnL tracking
- [ ] 10-03: Stats report — total return, Sharpe, Sortino, max drawdown, win rate, avg trade, equity curve output

### Phase 11: XGBoost Model Training
**Goal**: Train an XGBoost model on earlier historical data (e.g. 2022–2023) and validate on held-out data (e.g. 2024) to confirm it's not overfit. Save the trained model as a `.pkl` file for use in Phase 10 backtest and live trading.
**Depends on**: Phase 10 (needs backtest runner to validate the trained model)
**Research**: Likely (XGBoost feature importance, walk-forward validation, label engineering for BUY/SELL signals)
**Plans**: 2 plans

Plans:
- [x] 11-01: Label engineering + walk-forward CV — create scripts/train_model.py with prepare_features(), prepare_training_data() (forward-return threshold N=6 horizon τ=0.00015), run_walk_forward_cv() (TimeSeriesSplit n_splits=5 gap=24), verify no look-ahead leakage
- [ ] 11-02: Final training + save — train on full dataset excluding test period, save to models/xgb_btc_4h.pkl, verify model interface (pickle, feature_names_in_, predict_proba)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Project Scaffolding | 2/2 | Complete | 2026-03-16 |
| 2. API Client & Rate Limiter | 2/2 | Complete | 2026-03-17 |
| 3. Infrastructure Utilities | 1/1 | Complete | 2026-03-17 |
| 4. Data Pipeline | 3/3 | Complete | 2026-03-16 |
| 5. Execution Engine | 3/3 | Complete | 2026-03-16 |
| 6. Strategy Interface | 2/2 | Complete | 2026-03-16 |
| 7. Main Loop Orchestration | 2/2 | Complete | 2026-03-17 |
| 8. EC2 Deployment | 2/2 | Complete | 2026-03-17 |
| 9. Historical Data Download | 1/1 | Complete | 2026-03-17 |
| 10. Backtest Runner + Feature Prep | 3/3 | Complete | 2026-03-18 |
| 11. XGBoost Model Training | 1/2 | In Progress | 2026-03-18 |

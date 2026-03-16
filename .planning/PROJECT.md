# Roostoo Quant Trading Bot

## What This Is

An autonomous, long-only cryptocurrency swing trading bot built for the Roostoo Quant Trading Hackathon. The bot trades BTC/USD on the Roostoo mock exchange (Binance-priced) using 4H candles, regime-filtered momentum and mean-reversion strategies, ATR-based stop-losses, and circuit breaker risk management. It runs 24/7 on AWS EC2 (Sydney, ap-southeast-2) under systemd and requires zero manual intervention once deployed.

## Core Value

A correctly-wired, crash-safe infrastructure that submits valid orders and never loses state — the user fills in alpha logic on top.

## Requirements

### Validated

(None yet — ship to validate)

### Active

**Agent 1 — Foundation Layer (`bot/api/`, `bot/monitoring/`, `bot/persistence/`)**
- [ ] `RoostooClient` with HMAC SHA256 signing: sort params alphabetically, join as `key=val&...`, sign with secret, send `RST-API-KEY` + `MSG-SIGNATURE` headers; POST body form-encoded (not JSON)
- [ ] Exponential backoff: 3 retries, delays 2s/4s/8s (FAQ Q19)
- [ ] Global 30-call/minute sliding window rate limiter covering ALL outbound requests (not just trades); separate 65s trade cooldown layered on top
- [ ] Rate limiter must release threading lock BEFORE calling `time.sleep()` — holding the lock blocks emergency SELL orders (Issue 02)
- [ ] `pending_count` `Success=false` must NOT log WARNING — it is the normal response for zero pending orders; check `TotalPending` field instead (Issue 01)
- [ ] `TelegramAlerter.send()` must NEVER raise — all calls wrapped in try/except
- [ ] `StateManager` with atomic write (write to `.tmp` then rename to final path)
- [ ] `.env.example` documents three key sets: testing keys (`ROOSTOO_API_KEY_TEST`, `ROOSTOO_SECRET_TEST`), Round 1 competition keys (`ROOSTOO_API_KEY`, `ROOSTOO_SECRET`), and a placeholder for finalist keys

**Agent 2 — Data Pipeline (`bot/data/`)**
- [ ] `LiveFetcher.__init__` accepts `seed_dfs: dict[str, pd.DataFrame]` (not `seed_df: pd.DataFrame`) — the dict is keyed by pair symbol (Issue 15 / orchestration.md)
- [ ] `_seed_from_history()` uses flat Parquet column access (`df["close"]` etc.), NOT multi-index `df["BTC/USD"]` — Binance Parquet files use flat columns (Issue 04, CRITICAL: 8h startup delay otherwise)
- [ ] `get_latest_price(pair: str) -> float` method on `LiveFetcher` (Issue 15)
- [ ] `get_candle_boundaries() -> dict[str, int]` method on `LiveFetcher` returning last candle close timestamps per pair (Issue 15)
- [ ] `compute_features()` uses close-to-close ATR proxy: `log_returns.rolling(14).std() * df["close"] * 1.25` — NOT `pandas_ta ATR` (ATR≈0 on synthetic flat candles from Roostoo; Gap 02)
- [ ] OBV disabled in feature set (volume=0 on synthetic candles; Gap 05)
- [ ] ADX removed from live feature set (H=L on synthetic candles; replaced by EMA slope or RSI regime; Gap 05/06)
- [ ] `compute_cross_asset_features()` called BEFORE `dropna()` — current ordering drops all ETH/SOL rows (Issue 07, HIGH)
- [ ] All indicator columns shifted by 1 bar (`shift(1)`) to prevent look-ahead bias
- [ ] `is_warmed_up()` threshold: minimum 35 bars (MACD(12,26,9) warmup; Gap 12)

**Agent 3 — Execution Layer (`bot/execution/`)**
- [ ] `RegimeDetector` resamples 4H buffer to daily before applying EMA(20)/EMA(50) crossover logic — preserves original calibration; requires 300+ 4H bars warmup (Issue 11, Gap 06)
- [ ] `RiskManager.dump_state() -> dict` — serialize `trailing_stops`, `entry_prices`, `portfolio_hwm`, `circuit_breaker_active` (Issue 12, CRITICAL)
- [ ] `RiskManager.load_state(state: dict)` — restore all above from state dict on startup (Issue 12, CRITICAL)
- [ ] `OrderManager` fill_price logic uses explicit `is None` check, not `or` chain — `fill_price or quantity` silently records quantity as price when `fill_price=0.0` (Issue 13, CRITICAL)
- [ ] `_resync_from_exchange()` writes detected discrepancies back to `self._positions` after fetching live balance (Issue 14, HIGH)
- [ ] `OrderManager.get_all_positions()` method implemented (Issue 15)
- [ ] `OrderManager.dump_state()` and `load_state()` implemented (Issue 15)
- [ ] `cancel_order(order_id)` stub added (API endpoint `/v3/cancel_order` exists per FAQ Q18)
- [ ] Tiered circuit breaker: 30% drawdown → hard stop (0% size); 20-30% → 25% size; 10-20% → 50% size; <10% → 100% size (Gap 07)

**Agent 4 — Strategy + Orchestration (`bot/strategy/`, `bot/config/`, `main.py`)**
- [ ] `TradingSignal.pair` has NO default value — required positional field; `TradingSignal()` without pair must fail at construction (Issue 10, CRITICAL)
- [ ] `bot/config/config.yaml` exists with all tunable parameters: `hard_stop_pct`, `atr_stop_multiplier`, `circuit_breaker_drawdown`, `max_positions: 1`, `tradeable_pairs: ["BTC/USD"]`, `feature_pairs: ["BTC/USD", "ETH/USD", "SOL/USD"]`, `candle_interval: "4h"`, `regime.*`, `trade_cooldown_seconds: 65`
- [ ] `main.py` uses `from bot.api.client import RoostooClient` etc. (`bot.*` namespace throughout — Issue 19 + orchestration.md spec)
- [ ] `startup_reconciliation()` fully wired in `main.py` startup sequence — calls `client.get_balance()`, `client.get_open_orders()`, reconciles positions vs live state, sends Telegram WARN on discrepancy (Issue 16)
- [ ] Shutdown handler (SIGTERM/SIGINT) calls `state_manager.write(...)` before exiting — prevents losing the last ~60s of state (Issue 17, HIGH)
- [ ] Main loop operation order: (1) poll ticker → (2) get balance + check CB → (3) check stops per open position → (4) if new 4H candle: compute features + generate signal + size + submit → (5) write state.json → (6) heartbeat log → (7) sleep to next 60s boundary (Gap 08)
- [ ] Boundary-aligned sleep: `time.sleep(60.0 - (time.time() % 60.0))` to keep candle detection reliable
- [ ] `requirements.txt` uses `pandas-ta-classic` NOT `pandas-ta` (original is unmaintained, broken on Python 3.11 + pandas 2.x; Gap 12)
- [ ] Strategy stubs (`momentum.py`, `mean_reversion.py`) have correct `generate_signal(pair, features)` signatures with docstrings; user fills in alpha logic

### Out of Scope

- **FGI (Fear & Greed Index) integration** — correlation with next-day BTC returns < 0.05; API reliability risk; external dependency; EMA regime already captures similar signal (Gap 11 recommendation)
- **ETH/SOL/BNB trading positions** — Roostoo API only supports `BTC/USD` as a tradeable pair; ETH/SOL/BNB are feature-only inputs (Issue 20)
- **Limit orders** — market orders only for MVP; Roostoo mock fills market orders synchronously with zero slippage (Gap 04)
- **`backtest_fold()` implementation** — pre-competition optimization task for the user; blocked by needing validated data pipeline first (Issue 08)
- **vbt_sweep look-ahead fix** — optimization tooling, not live infrastructure (Issue 09)
- **Multi-pair concurrent positions** — `max_positions: 1` since BTC/USD is the only tradeable pair
- **Binance WebSocket parallel feed** — unnecessary complexity; close-to-close ATR proxy solves the synthetic candle problem without a second data source (Gap 02)
- **Frontend / manual trading** — competition rules forbid manual intervention (FAQ Q26)

## Context

### Competition Setup
- **Exchange**: Roostoo mock exchange (`https://mock-api.roostoo.com`), Binance-priced
- **Round 1**: Starts Mar 21 8PM; first trade must execute before Mar 22 8PM
- **Round 2 (if finalist)**: Starts Apr 4 8PM; separate API keys issued
- **Scoring**: Ranked by ROI + Sharpe + Sortino + Calmar on Roostoo App leaderboard
- **Capital**: $50,000 virtual USD (briefing says $1M — treat API reference as authoritative)
- **Auth**: HMAC SHA256 — sort params alphabetically, `key=val&...` string, sign with secret; POST bodies form-encoded
- **Rate limit**: 30 calls/minute for ALL API calls (not just trades); 65s minimum between `place_order` calls

### Critical Architecture Constraint: No OHLCV from Roostoo
Roostoo provides only `LastPrice` via `/v3/ticker`. Synthetic candles have H=L=O=C=LastPrice and volume=0. This means:
- Standard ATR (requires high-low range) ≈ 0 → entire stop-loss system breaks
- ADX (requires directional movement) ≈ 0 → regime detection breaks
- OBV (requires volume) = static → meaningless
- **Fix**: Replace ATR with `log_returns.rolling(14).std() * close * 1.25`. Use same formula in both backtest and live. Remove ADX and OBV from live feature set.

### Data Architecture
- Historical OHLCV: Binance `BTCUSDT` 4H Parquet files (flat columns: open/high/low/close/volume)
- Live feed: Roostoo `/v3/ticker` polled every 60s → synthetic candle appended to buffer
- State persistence: `state.json` (atomic write each cycle), `logs/trades.log` (append), `logs/bot.log` (rotating 10MB × 10)
- USDT/USD price spread ≈ 0.05% under normal conditions → negligible for indicators (Gap 10)

### Known API Quirks
- `pending_count` returns `{"Success": false, "TotalPending": 0}` for zero pending — NOT an error
- `pending_only` parameter must be string `"TRUE"` / `"FALSE"`, not Python bool
- Quantity is in coin units (BTC), not USD
- Commission: 0.012% TAKER (MARKET orders) — use this for sizing math
- Market orders fill synchronously; `FilledPrice` always populated in response

### EC2 Infrastructure
- Region: Sydney ap-southeast-2 (FAQ Q13)
- Instance: t3.micro On-Demand (NOT Spot — avoids unexpected termination)
- Time sync: chrony → Amazon Time Sync Service (169.254.169.123)
- Process: systemd service, `ExecStart=python main.py`, `RestartSec=10`

### 4-Agent Parallel Build
Agents 1/2/3 are fully independent (no file conflicts). Agent 4 starts `bot/strategy/base.py` + `bot/config/config.yaml` immediately but holds `main.py` until Agents 1/2/3 finish. See `docs/orchestration.md` for full spec.

## Constraints

- **Timeline**: Bot must be deployed, tested with testing keys, and confirmed running before Mar 21. No time for significant architecture changes after competition starts.
- **API**: 30 calls/minute hard limit for ALL requests — rate limiter must cover every outbound call, not just trades
- **Tradeable universe**: BTC/USD only on Roostoo (despite briefing mentioning 66 pairs — API reference is authoritative)
- **Python 3.11**: EC2 instance uses Python 3.11; `pandas-ta-classic` required (original `pandas-ta` breaks on pandas 2.x)
- **Autonomy rule**: All trades must be bot-generated; manual API calls during competition = disqualification (FAQ Q26); strategy updates allowed with commit history
- **No OHLCV**: Close-to-close vol proxy is mandatory everywhere ATR appears (backtest AND live) to ensure calibration consistency
- **Bot namespace**: All `bot/` package imports must use `from bot.api.client import ...` — not bare `from api.client import ...`
- **No hardcoded secrets**: API keys via `.env` only; tested with `everything-claude-code:security-review`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Close-to-close ATR proxy (`log_ret.std() * close * 1.25`) everywhere | Roostoo has no OHLCV; standard ATR ≈ 0 on synthetic candles; must use same formula in backtest and live for calibration consistency | — Pending |
| Resample 4H → daily for regime detection | EMA(20)/EMA(50) are calibrated for daily data; applying to 4H causes 6x faster regime flips; resampling preserves original calibration without re-tuning | — Pending |
| `max_positions: 1` / `tradeable_pairs: ["BTC/USD"]` | Roostoo API only supports BTC/USD; multi-asset architecture adds dead code paths; ETH/SOL/BNB used as features only | — Pending |
| `pandas-ta-classic` over `pandas-ta` | Original `pandas-ta` unmaintained since 2022; broken on Python 3.11 + pandas 2.x; `pandas_ta_classic` is actively maintained fork with identical API | — Pending |
| No FGI integration | Correlation < 0.05 with next-day BTC returns; API reliability risk; existing EMA regime already captures trend sentiment | — Pending |
| MARKET orders only (no LIMIT) | Synchronous fill model simplifies OMS; Roostoo mock exchange has zero slippage; no partial-fill complexity | — Pending |
| Tiered circuit breaker (not binary) | Binary hard-stop at 30% misses recovery rallies; tiered approach (25%/50%/100% size at 20%/10%/0% drawdown) is better for competition scoring | — Pending |
| Hard stop at 30% drawdown for circuit breaker full halt | At typical position sizes (~17%), reaching 30% drawdown requires ~50 consecutive max-loss trades — appropriate as catastrophic-failure safeguard | — Pending |
| `bot.*` namespace for all imports | FAQ Q37 best practices + orchestration.md mandate; Layer 10 docs show old-style imports that Agent 4 must fix | — Pending |
| Hold `main.py` until Agents 1/2/3 finish | `main.py` imports from all packages; writing it before interfaces are final causes integration errors | — Pending |

---
*Last updated: 2026-03-16 after full docs read (all 13 layers + 20 issues + 12 gaps)*

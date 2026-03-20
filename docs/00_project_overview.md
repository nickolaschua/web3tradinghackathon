# Roostoo Quant Hackathon — Project Overview

## What This Is

You are building a fully autonomous cryptocurrency trading bot to compete in the Roostoo Mock Exchange Hackathon. The competition starts you with $50,000 in virtual capital, connects all teams to the same mock exchange that mirrors real crypto prices, and ranks everyone on a public leaderboard by ROI. Your bot runs 24/7 on cloud infrastructure for the entire duration. The team with the highest return at the end wins.

This is not a coding challenge. It is a systems challenge. The team that wins will not have the cleverest algorithm — they will have the most robust, failure-resistant system that makes disciplined, well-validated decisions and survives the full competition without blowing up.

---

## The Core Thesis

**The edge in this competition does not come from finding a secret signal. It comes from not doing what everyone else will do.**

Most teams will:
- Build their bot during the competition and deploy untested code
- Use momentum signals without ever testing them out of sample
- Stay fully invested regardless of market regime
- Have no stops, so one bad position ruins their leaderboard position
- Overtrade trying to catch every move, paying fees constantly

Your advantage is preparation. By building and validating your entire infrastructure before the competition starts, you arrive with a tested, deployed system and spend the competition monitoring rather than building. Every hour another team spends debugging their signing function is an hour you spend watching your bot trade correctly.

---

## What You Are Building

The system has three distinct phases of life:

**Phase 1 — Research (pre-hackathon, local machine)**
A backtesting and research environment that processes years of historical crypto data, tests signal combinations, and validates strategy parameters using walk-forward optimization. Nothing from this phase is deployed to EC2. Its only output is a `config.yaml` with validated strategy parameters and confidence that your strategy has genuine out-of-sample performance.

**Phase 2 — Execution (live, on EC2)**
A production trading bot that polls the Roostoo API every 60 seconds, computes signals from live market data, applies risk management rules, and places orders. This system runs autonomously 24/7. You interact with it only through Telegram alerts and occasional SSH checks.

**Phase 3 — Monitoring (continuous, on your phone)**
Telegram notifications for every trade, every error, every regime change, and a heartbeat every 10 minutes confirming the bot is alive. If it goes silent, something is wrong.

---

## The Architecture in One Sentence Per Layer

- **API Client** — signs every request with HMAC SHA256 and handles all communication with the Roostoo exchange
- **Data Pipeline** — downloads historical OHLCV data for backtesting and maintains a live rolling buffer for signal computation
- **Feature Engineering** — transforms raw price data into the indicators and signals your strategy logic consumes
- **Backtesting Engine** — validates strategies against historical data using walk-forward optimization before any live capital is risked
- **Strategy Engine** — generates BUY/SELL/HOLD decisions from features, with regime detection selecting which strategy is active
- **Risk Management** — enforces stop-losses, position limits, and a portfolio circuit breaker, acting as a hard gate before any order is submitted
- **Order Management** — handles the pre-flight checks and lifecycle tracking of every order from submission to fill or cancellation
- **State & Persistence** — writes the bot's complete state to disk after every cycle so a crash or restart loses nothing
- **Monitoring** — structured logs to disk, Telegram alerts to your phone, and heartbeat checks confirming the process is alive
- **Orchestration** — the main loop that sequences all of the above every 60 seconds, with a catch-all exception handler ensuring nothing kills the bot

---

## Why Each Piece Exists

Every component in this system exists to address a specific failure mode identified from first principles. Nothing is over-engineering for its own sake.

| Component | Failure it prevents |
|---|---|
| HMAC signing + time sync | Authentication rejection, timestamp drift rejection |
| Precision cache from exchangeInfo | Silent order rejection due to wrong decimal places |
| Rate limiter | Rate limit cascade and temporary lockout |
| Historical Parquet data | Slow, corrupt, or gap-filled backtests |
| Gap detector | NaN propagation from missing candles corrupting indicators |
| Live fetcher + candle boundary detection | Spurious repeated signals from evaluating mid-candle |
| Look-ahead bias enforcement (shift + expanding norm) | Strategies that look great in backtest but fail live |
| Walk-forward optimization | Overfitted parameters that don't generalize |
| Regime stress testing | Strategies that only work in one market condition |
| Regime detection (EMA + ADX) | Holding long positions into a bear market and bleeding |
| ATR trailing stop + hard % stop | Single position causing catastrophic portfolio loss |
| Portfolio circuit breaker | Drawdown spiral where bot keeps trading into losses |
| response['Success'] check | Ghost positions from silent order failures |
| Free balance check before entry | Rejected orders due to capital already locked in open orders |
| State persistence + startup reconciliation | Crash recovery losing position context and re-entering incorrectly |
| Telegram heartbeat | Silent outages going unnoticed for hours |
| Catch-all exception handler in main loop | Any unhandled error killing the bot permanently |

---

## What Winning Looks Like

Crypto competitions reward two things: being right when others are wrong, and surviving when others blow up.

The goal is not the highest possible return in ideal conditions — it is the highest return *that you can reliably achieve* given the competition constraints. Those constraints are: 1 trade per minute rate limit (no HFT), long-only (no shorting), $50,000 starting capital, ranked by ROI.

Given these constraints, the optimal strategy profile is:
- Low frequency swing trading on 4-hour candles (2–5 trades per day maximum)
- Momentum-based entries in trending markets, mean reversion in range-bound markets
- Hard rules for going to cash when the market turns bearish (since you cannot short)
- ATR-based trailing stops to ride winners and cut losers mechanically
- Concentrated positions (2–4 pairs) rather than diversified small bets

Evidence from similar competitions shows that the winner typically executes fewer than 50 total trades over a 2-week event, using simple but disciplined signals, while losing teams average 10+ trades per day and give back edge to fees and noise.

---

## The Non-Negotiables

These are the things that, if skipped, will likely determine the outcome of the competition:

1. **Test HMAC signing against the known reference hash before writing any other code.** If signing is broken, nothing works.

2. **Enforce look-ahead bias prevention in every backtest.** `shift(1)` on all signal series. Expanding window normalization only. If your backtest has look-ahead bias, you will deploy a strategy that appears validated but isn't.

3. **Hold out the final 20% of your data and never touch it during optimization.** Evaluate it exactly once at the end of Day 7. If it fails, your parameters are overfit.

4. **Build the cash-as-hedge regime filter on Day 1 of strategy work, not as an afterthought.** Being in cash during a bear move is the most reliable source of relative outperformance in a long-only competition.

5. **Wrap the entire main loop in try/except before deployment.** Any unhandled exception that kills the bot is equivalent to going to cash at zero return for the rest of the competition.

6. **Test crash recovery explicitly.** Kill the process manually, restart it, verify reconciliation succeeds. If you haven't tested it, it will fail when it actually matters.

---

## File Structure Reference

```
trading-bot/
├── main.py                     orchestration loop
├── config.yaml                 strategy params (hot-swappable)
├── .env                        API keys (gitignored, chmod 600)
├── state.json                  runtime state (auto-written every cycle)
│
├── api/
│   ├── client.py               HMAC signing, all endpoints, time sync
│   ├── rate_limiter.py         65s trade cooldown, thread-safe
│   └── exchange_info.py        precision cache, MiniOrder rules
│
├── data/
│   ├── downloader.py           bulk Binance archive download to Parquet
│   ├── gap_detector.py         validate and fill gaps in historical data
│   ├── live_fetcher.py         rolling in-memory buffer, candle-boundary detection
│   └── features.py             all indicator and feature computation (shared)
│
├── backtesting/                PRE-HACKATHON ONLY — not deployed to EC2
│   ├── vbt_sweep.py            vectorbt rapid parameter sweeps
│   ├── bt_validator.py         backtesting.py detailed single-strategy runs
│   ├── walk_forward.py         Optuna + TimeSeriesSplit walk-forward optimization
│   ├── regime_stress.py        performance isolated by labeled market regime
│   └── ic_analysis.py          Spearman IC per feature vs forward returns
│
├── strategy/
│   ├── base.py                 abstract interface: generate_signal() → action
│   ├── momentum.py             primary: multi-timeframe trend following
│   └── mean_reversion.py       backup: Bollinger Band mean reversion
│
├── execution/
│   ├── regime.py               BULL / SIDEWAYS / BEAR classification
│   ├── risk.py                 stops, circuit breaker, position sizing
│   └── order_manager.py        OMS: lifecycle, pre-flight, reconciliation
│
├── persistence/
│   └── state.py                state.json read/write + startup reconciliation
│
└── monitoring/
    ├── logger.py               RotatingFileHandler, JSON logs, trades.log
    ├── telegram.py             alert dispatcher
    └── healthcheck.py          memory, disk, CPU, heartbeat
```

---

## Technology Stack

| Purpose | Library |
|---|---|
| HTTP requests | `requests` |
| Data manipulation | `pandas`, `numpy` |
| Parquet I/O | `pyarrow` |
| Technical indicators | `pandas-ta` |
| Fast backtesting | `vectorbt` |
| Detailed backtesting | `backtesting` (backtesting.py) |
| Bayesian optimization | `optuna` |
| Performance tearsheets | `quantstats` |
| Multi-exchange data | `ccxt` |
| ML signals (optional) | `lightgbm`, `scikit-learn` |
| Secrets management | `python-dotenv` |
| System monitoring | `psutil` |
| Process management | `systemd` (EC2) |
| Clock sync | `chrony` (EC2) |

---

## Competition Constraints and How They Shape the System

| Constraint | Implication |
|---|---|
| 1 trade/minute rate limit | HFT is impossible; swing trading on 4H candles is optimal |
| Long-only | Cash is your only hedge; regime detection is critical |
| Ranked by ROI | Concentration beats diversification; 2–4 pairs is correct |
| 24/7 autonomous operation | State persistence and crash recovery are non-negotiable |
| AWS EC2 provided | Systemd service management, chrony time sync, t3.micro is sufficient |
| Real-time prices on mock exchange | Backtesting on Binance data is a valid proxy (USDT ≈ USD) |
| WhatsApp channel with engineers | Use it immediately if API behaviour differs from documentation |

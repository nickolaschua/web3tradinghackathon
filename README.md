# Web3 Trading Hackathon Bot

## AWS EC2 Deployment Guide

### Connect to Instance
1. AWS Console -> EC2 -> your instance -> **Connect** -> **Session Manager** -> **Connect**

### First Time Setup
```bash
sudo su - ec2-user
cd ~
git clone <your-repo-url>
cd web3tradinghackathon
pip3 install -r requirements.txt
nano .env  # paste your .env contents
python scripts/download_data.py --interval 15m --symbols BTC XRP ETH SOL
```

### Start the Bot
```bash
sudo su - ec2-user
cd web3tradinghackathon
tmux
python3 main.py
```
Detach (leave running): `Ctrl+B` then `D`

### Update & Restart
```bash
sudo su - ec2-user
cd web3tradinghackathon
tmux attach
Ctrl+C          # stop the bot
git pull
python3 main.py
```
Detach: `Ctrl+B` then `D`

### Reattach to Running Bot
```bash
sudo su - ec2-user
tmux attach
```

### Important Notes
- Always run as **ec2-user**, not ssm-user
- Bot lives at: `ec2-user` -> `~/web3tradinghackathon`
- Use `python3`, not `python`
- 15m parquet files must exist in `data/` (not in git, download with script above)
- `cb_active=False` in logs is normal (circuit breaker not triggered)

---

## Live Strategy (v3 -- Focused 2-Coin XGBoost, March 22 2026)

### What Changed (v3)

v2 had 20-coin universe with MR fallbacks that diluted the XGBoost edge. Portfolio allocator gave BTC only ~5% weight instead of 50%, turning a Sharpe 1.2 strategy into Sharpe 0.24.

| Change | v2 | v3 | Why |
|--------|-----|-----|-----|
| Trading universe | 39 coins | **BTC + XRP only** | Concentrate capital on coins with proven XGB edge |
| Portfolio weight | ~5% per coin (HRP/CVaR across 20) | **50% per coin** | No dilution -- each model gets meaningful allocation |
| MR fallback | Active (150 trades, ~50% WR) | **Disabled** | No edge, just commission drag |
| Relaxed MR | Active (micro positions) | **Disabled** | Overkill for activity -- Phase E handles it |
| XRP model | None | **New XGBoost** | High frequency (163 trades), carries activity requirement |
| SOL model | Active (threshold 0.70) | **Removed** | Only 2 trades in 2 years at 0.70 threshold |

### Backtest Results (OOS 2024-2026, $1M capital, 10bps fees)

| Metric | v2 (20-coin diluted) | v3 (BTC+XRP focused) |
|--------|------|------|
| **Return** | +0.79% | **+43.24%** |
| **Sharpe** | 0.238 | **1.130** |
| **Sortino** | 0.099 | **1.391** |
| **Calmar** | 0.166 | **1.494** |
| **CompScore** | 0.21 | **1.344** |
| Max DD | -2.21% | -11.64% |
| Trades | 390 | 199 |

### Signal Stack (per 15M bar)

| Priority | Strategy | Pair | Entry | Exit |
|----------|----------|------|-------|------|
| 1 | **BTC XGBoost** (`xgb_btc_15m_iter5.pkl`) | BTC/USD | P(BUY) >= 0.65 | P(BUY) <= 0.08 or ATR stop |
| 2 | **XRP XGBoost** (`xgb_xrp_15m.pkl`) | XRP/USD | P(BUY) >= 0.65 | P(BUY) <= 0.08 or ATR stop |
| 3 | **Micro-trade fallback** | BTC/ETH/SOL/BNB/XRP | $500 BUY if no trade by 20:00 UTC | ATR stop |

No MR, no pairs ML, no relaxed MR. XGBoost signals only.

### Position Pyramiding

The bot **adds to existing positions** when the model fires repeated BUY signals. With only 2 coins, blocking repeat entries would leave the bot idle most of the time (both positions open, nothing to do until an exit).

How it works:
- Each new BUY signal sizes an independent tranche (2% risk, ATR-based)
- Each tranche gets its own trailing stop
- New tranches are allowed up to the **40% concentration cap per coin**
- With 2 coins, max total exposure is ~80% (40% BTC + 40% XRP), 20% stays cash

Why this is acceptable for competition:
- BTC and XRP are highly liquid -- no execution risk on $400K positions
- Circuit breaker halves sizing at 10% drawdown, halts at 30%
- ATR trailing stops protect each tranche independently
- 10-day competition rewards concentrated bets on best models
- Backtest max drawdown with this approach: -11.6%

Why you would NOT do this with real money:
- 80% in 2 correlated crypto assets is extreme concentration
- A flash crash hits both BTC and XRP simultaneously
- For real portfolios, diversify across 10+ uncorrelated assets

### Trade Frequency

| Coin | Entries (2yr OOS) | Active Days | Frequency |
|------|-------------------|-------------|-----------|
| XRP | 163 | 423/809 (52%) | ~1 every 0.6 days |
| BTC | 36 | 82/809 (10%) | ~1 every 3.5 days |
| **Combined** | **199** | **463/809 (57%)** | **~1 every 1.7 days** |

Recent 10-day windows: 9/10, 11/10, 10/10, 9/10, 10/10 active days. Phase E covers any gaps.

### Risk Management

| Control | Value |
|---------|-------|
| Position sizing | 2% portfolio risk per trade (ATR-based) |
| Portfolio weight | 50% per coin (BTC, XRP) |
| ATR trailing stop | 10x multiplier (15M calibrated, ratchets up only) |
| Hard stop floor | 5% below entry |
| Circuit breaker | 10% DD -> 0.5x / 20% DD -> 0.25x / 30% DD -> halt |
| Kelly gate | Blocks trades with edge <= 0 |
| Max concentration | 40% of portfolio per coin (pyramiding allowed up to cap) |
| Regime scaling | Bull: 1.0x / Sideways: 0.5x / Bear: 0.35x |
| Micro-trade fallback | $500 BUY at 20:00 UTC if no trade that day |

### Regime Detection

EMA(20)/EMA(50) crossover with 0.1% dead zone.

| Regime | Condition | Multiplier |
|--------|-----------|------------|
| Bull | EMA20 > EMA50, spread > 0.1% | 1.00x |
| Sideways | Spread < 0.1% | 0.50x |
| Bear | EMA20 < EMA50 | 0.35x |

### Models

| Model | Pair | Threshold | Test AP | Trades (OOS) | Win Rate |
|-------|------|-----------|---------|--------------|----------|
| `xgb_btc_15m_iter5.pkl` | BTC/USD | 0.65 | 0.405 | 36 | 50% |
| `xgb_xrp_15m.pkl` | XRP/USD | 0.65 | 0.412 | 163 | 39% |

Both trained on pre-2024 data, tested OOS 2024-2026. 19 features: TA indicators + ETH/SOL cross-asset lags + BTC/ETH correlation/beta. Triple-barrier labels (TP=0.5%, SL=0.3%, horizon=16 bars).

### Robustness Notes

- **No look-ahead bias**: All features use shift(1), confirmed via correlation test (0.01)
- **No train/test leakage**: CV gap = 64 bars > 16-bar label horizon
- **Return concentration**: Top 5 days account for 60% of total return -- fragile to timing
- **Q4 2025 is negative (-5.8%)**: Model edge may be decaying in recent conditions
- **Selection bias**: XRP was chosen from 11-coin screen based on OOS performance
- **Realistic 10-day expectation**: -2% to +3% depending on market conditions

### Competition Context

| Field | Value |
|-------|-------|
| Competition | Roostoo Round 1 |
| Dates | March 21-31, 2026 (10 days) |
| Scoring | 0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar |
| Commission | 0.1% per side (10 bps) |
| Starting capital | $1,000,000 |
| Active days required | 8 out of 10 |

### File Reference

| Component | File |
|-----------|------|
| Main entry point | `main.py` |
| XGBoost strategy | `bot/strategy/xgboost_strategy.py` |
| Risk manager | `bot/execution/risk.py` |
| Regime detector | `bot/execution/regime.py` |
| Order manager | `bot/execution/order_manager.py` |
| Portfolio allocator | `bot/execution/portfolio.py` |
| Config | `bot/config/config.yaml` |
| Telegram alerts | `bot/monitoring/telegram.py` |
| Live data | `bot/data/live_fetcher.py` |
| Features | `bot/data/features.py` |
| Data download | `scripts/download_data.py` |
| BTC backtest | `scripts/backtest_15m.py` |
| Allocation comparison | `scripts/backtest_allocation.py` |

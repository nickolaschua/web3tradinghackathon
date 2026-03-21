# Web3 Trading Hackathon Bot

## AWS EC2 Deployment Guide

### Connect to Instance
1. AWS Console → EC2 → your instance → **Connect** → **Session Manager** → **Connect**

### First Time Setup
```bash
sudo su - ec2-user
cd ~
git clone <your-repo-url>
cd web3tradinghackathon
pip3 install -r requirements.txt
nano .env  # paste your .env contents
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
- Bot lives at: `ec2-user` → `~/web3tradinghackathon`
- Use `python3`, not `python` (python command not found)
- There is an old bot setup under `ssm-user/bot` — do not use it
- `cb_active=False` in logs is normal (circuit breaker not triggered)

---

## Live Strategy

### Signal Stack (per 15M bar, per pair)

| Priority | Strategy | Pairs | Entry | Exit |
|----------|----------|-------|-------|------|
| 1 | **BTC XGBoost** (`xgb_btc_15m_iter5.pkl`) | BTC/USD only | P(BUY) ≥ 0.65 | P(BUY) ≤ 0.10 or ATR stop |
| 2 | **SOL XGBoost** (`xgb_sol_15m.pkl`) | SOL/USD only | P(BUY) ≥ 0.75 | P(BUY) ≤ 0.10 or ATR stop |
| 3 | **Mean Reversion** (fallback) | All 39 pairs | RSI < 30 + EMA_20 > EMA_50 + BB undershoot | RSI > 55 or BB reversion |

For BTC/USD and SOL/USD, XGBoost runs first. MR fires only if XGBoost returns HOLD.
For all other 37 pairs, MR is the sole signal source.

### Risk Management

- **Position sizing**: 2% portfolio risk per trade (ATR-based)
- **Primary exit**: ATR trailing stop at 10x multiplier (15M calibrated)
- **Hard stop floor**: 5% below entry
- **Circuit breaker**: 10% DD → half size | 20% DD → quarter size | 30% DD → halt
- **Kelly gate**: blocks trades with edge ≤ 0
- **Max positions**: 5 concurrent

### OOS Backtest Results (2024–2026)

| Stack | Sharpe | Sortino | Calmar | Return | MaxDD |
|-------|--------|---------|--------|--------|-------|
| BTC XGBoost + MR fallback | 1.434 | 2.037 | ~2.25 | +25.8% | -4.8% |
| BTC + SOL XGBoost (portfolio) | **1.814** | **2.725** | — | +29.6% | -4.1% |

### Models

| Model | Features | Triple-barrier labels | CV Sharpe |
|-------|----------|-----------------------|-----------|
| `xgb_btc_15m_iter5.pkl` | 19 (TA + ETH/SOL cross-asset) | TP=0.8%, SL=0.3%, horizon=16 bars | 1.558 (exit=0.10) |
| `xgb_sol_15m.pkl` | 19 (TA + BTC/ETH cross-asset + SOL/BTC beta) | TP=0.8%, SL=0.3%, horizon=16 bars | — |

### Key Design Decisions

- **Exit threshold = 0.10**: XGBoost SELL only fires when P(BUY) ≤ 10% (very bearish). Backtest sweep showed exit=0.10 outperforms the naive complement (exit=1−entry=0.30): Sharpe 1.558 vs 1.387. ATR trailing stop handles ~100% of actual exits.
- **SOL threshold = 0.75**: Higher bar than BTC (0.65) because SOL fires ~4 trades/2yr at 0.75 — very selective, only enters on strong conviction.
- **MR regime gate**: Mean reversion only enters when EMA_20 > EMA_50 per coin. Prevents catching falling knives in downtrends.

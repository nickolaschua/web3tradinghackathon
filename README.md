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
| 1 | **BTC XGBoost** (`xgb_btc_15m_iter5.pkl`) | BTC/USD only | P(BUY) >= 0.65 | P(BUY) <= 0.10 or ATR stop |
| 2 | **SOL XGBoost** (`xgb_sol_15m.pkl`) | SOL/USD only | P(BUY) >= 0.75 | P(BUY) <= 0.10 or ATR stop |
| 3 | **Mean Reversion** (original, high precision) | All 20 feature pairs | RSI < 30 + bb_pos < 0.15 + MACD > 0 + EMA_20 > EMA_50 | RSI > 55 or bb_pos > 0.6 |
| 4 | **Relaxed MR** (activity layer) | All 20 feature pairs | RSI < 35 + bb_pos < 0.25 (soft regime gate) | RSI > 50 or bb_pos > 0.55 |

For BTC/USD and SOL/USD, XGBoost runs first. MR and relaxed MR fire only if XGBoost returns HOLD.
For all other 18 pairs, MR cascade is the sole signal source.

### Why 4 Layers?

The competition requires **8 active trading days out of 10** to qualify. The BTC+SOL XGBoost models produce high Sharpe (1.814 portfolio) but only trade on ~12.7% of days — far too infrequent. A unified 20-coin XGBoost was attempted but failed (CV Mean AP = 0.272, barely above random) because pooling heterogeneous coins destroys signal.

The hybrid approach preserves the proven XGBoost edge while using relaxed MR as an activity layer:
- XGBoost provides alpha (BTC: +1.27% avg PnL per trade)
- Relaxed MR provides activity (micro 0.03x positions, 99.3% of days covered)
- Net result: Sharpe 1.280, Sortino 1.848, worst 10-day window = 9/10 active days

### Feature Pairs (20 coins)

BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC, UNI, NEAR, SUI, APT, PEPE, ARB, SHIB, FIL, HBAR

All verified: Roostoo-tradeable, 15M parquet data available (2.9-5 years), top 20 by Binance liquidity.

### Risk Management

- **Position sizing**: 2% portfolio risk per trade (ATR-based) for XGBoost; 0.03x portfolio for relaxed MR
- **Primary exit**: ATR trailing stop at 10x multiplier (15M calibrated)
- **Hard stop floor**: 5% below entry
- **Circuit breaker**: 10% DD -> half size | 20% DD -> quarter size | 30% DD -> halt
- **Kelly gate**: blocks trades with edge <= 0
- **Max positions**: 5 concurrent

### OOS Backtest Results (2024-2026, $1M capital)

| Stack | Sharpe | Sortino | Return | MaxDD | Trades | Active Days |
|-------|--------|---------|--------|-------|--------|-------------|
| BTC + SOL XGBoost only | 1.710 | 2.475 | +36.4% | ~-4% | 80 | 12.7% |
| **Hybrid (XGB + relaxed MR)** | **1.280** | **1.848** | **+27.0%** | **-5.5%** | **2342** | **99.3%** |

The Sharpe tradeoff (1.710 -> 1.280) is acceptable because:
1. Without 8/10 active days, the bot **does not qualify** for competition scoring
2. Sortino 1.848 is still strong (highest weight in scoring formula: 0.4x)
3. MR positions are micro-sized (0.03x) — limited downside drag

### Models

| Model | Features | Triple-barrier labels | Backtest Sharpe |
|-------|----------|-----------------------|-----------------|
| `xgb_btc_15m_iter5.pkl` | 19 (TA + ETH/SOL cross-asset + corr/beta) | TP=0.8%, SL=0.3%, horizon=16 bars | 1.558 (solo, exit=0.10) |
| `xgb_sol_15m.pkl` | 19 (TA + BTC/ETH cross-asset + SOL/BTC beta) | TP=0.8%, SL=0.3%, horizon=16 bars | lifts portfolio to 1.814 |

### Key Design Decisions

- **Exit threshold = 0.10**: XGBoost SELL only fires when P(BUY) <= 10% (very bearish). Backtest sweep showed exit=0.10 outperforms the naive complement (exit=0.30): Sharpe 1.558 vs 1.387. ATR trailing stop handles ~100% of actual exits.
- **SOL threshold = 0.75**: Higher bar than BTC (0.65) because SOL fires ~4 trades/2yr at 0.75 — very selective, only enters on strong conviction.
- **Relaxed MR soft regime gate**: Unlike original MR which blocks all entries in downtrends (EMA_20 < EMA_50), relaxed MR halves position size in downtrends instead of blocking entirely. This prevents "dead days" when all coins are bearish simultaneously.
- **Unified model rejected**: A pooled 20-coin XGBoost was trained and tested. ATR-normalized labels equalized BUY rates (20-24%) but the model had no edge (Mean AP = 0.272 vs 0.234 random). Heterogeneous coins cannot share one decision boundary. Per-coin models + activity layer is the correct architecture.

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
python scripts/download_data.py --interval 15m --symbols BTC XRP ADA HBAR LTC ETH SOL
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
- 15m parquet files must exist in `data/` (auto-refreshed on startup from Binance, or download with script above)
- `cb_active=False` in logs is normal (circuit breaker not triggered)

---

## Live Strategy (v4 -- 5-Coin XGBoost, March 24 2026)

### What Changed (v4)

v3 ran BTC + XRP only. BTC has fired 0 signals recently, XRP is losing money (-0.33% avg PnL on recent 6 months). v4 adds 3 validated models to diversify signal sources.

| Change | v3 | v4 | Why |
|--------|-----|-----|-----|
| Trading universe | BTC + XRP | **BTC + XRP + ADA + HBAR + LTC** | 3 new models with validated positive edge |
| Model selection | Picked best OOS | **Train→Select→Validate** 3-window design | Prevents selection bias / data snooping |
| Auto data refresh | Manual download_data.py | **Automatic on startup** | Parquets updated from Binance before each restart |
| Exit threshold | 0.08 | **0.10** | Validated on recent backtest window |
| Signal cascade | Hardcoded BTC→XRP | **Dict lookup per pair** | Clean, extensible, no cross-pair contamination |
| Feature pipeline | BTC-pipeline for all | **Per-coin: alt coins get {coin}_btc_corr** | Correct features for models trained with train_alt_15m.py |

### Model Validation (3-Window Design)

Models were validated using proper out-of-sample methodology:
1. **Train**: Pre-2024 data (all models)
2. **Select**: 2024-01 to 2025-06 — pick coins with positive edge (even after removing best trade)
3. **Validate**: 2025-07 to 2026-03 — test ONLY selected coins on unseen data

18 coins passed selection. Only 7 survived validation. ADA, HBAR, LTC chosen for deployment based on edge robustness and trade volume.

### Validation Results (2025-07 to 2026-03, unseen data)

| Coin | Trades | Win% | Avg PnL | Avg PnL (w/o best trade) |
|------|--------|------|---------|--------------------------|
| **ADA** | 41 | 39.0% | +1.71% | +0.93% |
| **HBAR** | 46 | 39.1% | +1.12% | +0.24% |
| **LTC** | 17 | 58.8% | +1.24% | +0.54% |
| XRP | 152 | 37.5% | -0.05% | -0.21% |
| BTC | 1 | 100% | +3.32% | — |

### Signal Stack (per 15M bar)

| Strategy | Pair | Entry | Exit |
|----------|------|-------|------|
| **BTC XGBoost** (`xgb_btc_15m_iter5.pkl`) | BTC/USD | P(BUY) >= 0.65 | P(BUY) <= 0.10 or ATR stop |
| **XRP XGBoost** (`xgb_xrp_15m.pkl`) | XRP/USD | P(BUY) >= 0.65 | P(BUY) <= 0.10 or ATR stop |
| **ADA XGBoost** (`xgb_ada_15m.pkl`) | ADA/USD | P(BUY) >= 0.70 | P(BUY) <= 0.10 or ATR stop |
| **HBAR XGBoost** (`xgb_hbar_15m.pkl`) | HBAR/USD | P(BUY) >= 0.70 | P(BUY) <= 0.10 or ATR stop |
| **LTC XGBoost** (`xgb_ltc_15m.pkl`) | LTC/USD | P(BUY) >= 0.70 | P(BUY) <= 0.10 or ATR stop |
| **Micro-trade fallback** | BTC/ETH/SOL/BNB/XRP | $500 BUY if no trade by 20:00 UTC | ATR stop |

Each pair has exactly one strategy. Direct dict lookup — no cascade, no cross-pair contamination.

### Trade Frequency

| Coin | Entries (validation period) | Frequency |
|------|----------------------------|-----------|
| XRP | 152 | ~1 every 1.7 days |
| HBAR | 46 | ~1 every 5.7 days |
| ADA | 41 | ~1 every 6.4 days |
| LTC | 17 | ~1 every 15 days |
| BTC | 1 | rare |
| **Combined** | **257** | **~1 every 1 day** |

XRP provides the trade volume for the 8-active-day competition requirement. Phase E micro-trades cover any gaps.

### Risk Management

| Control | Value |
|---------|-------|
| Position sizing | 2% portfolio risk per trade (ATR-based) |
| ATR trailing stop | 10x multiplier (15M calibrated, ratchets up only) |
| Hard stop floor | 5% below entry |
| Circuit breaker | 10% DD -> 0.5x / 20% DD -> 0.25x / 30% DD -> halt |
| Kelly gate | Blocks trades with edge <= 0 |
| Max concentration | 40% of portfolio per coin |
| Max positions | 5 concurrent |
| Micro-trade fallback | $500 BUY at 20:00 UTC if no trade that day |

### Models

| Model | Pair | Threshold | Features | Pipeline |
|-------|------|-----------|----------|----------|
| `xgb_btc_15m_iter5.pkl` | BTC/USD | 0.65 | eth_btc_corr/beta | BTC pipeline |
| `xgb_xrp_15m.pkl` | XRP/USD | 0.65 | eth_btc_corr/beta | BTC pipeline |
| `xgb_ada_15m.pkl` | ADA/USD | 0.70 | ada_btc_corr/beta | alt pipeline |
| `xgb_hbar_15m.pkl` | HBAR/USD | 0.70 | hbar_btc_corr/beta | alt pipeline |
| `xgb_ltc_15m.pkl` | LTC/USD | 0.70 | ltc_btc_corr/beta | alt pipeline |

All trained on pre-2024 data. 19 features each: 13 TA indicators + 4 cross-asset lags (ETH/SOL 4H+1D) + 2 context features. Triple-barrier labels (TP=0.8%, SL=0.3%, horizon=16 bars).

**Two feature pipelines** (critical for live inference):
- **BTC pipeline** (BTC, XRP): `compute_btc_context_features()` -> eth_btc_corr, eth_btc_beta
- **Alt pipeline** (ADA, HBAR, LTC): target-vs-BTC rolling correlation -> {coin}_btc_corr, {coin}_btc_beta

### Robustness Notes

- **No look-ahead bias**: All features use shift(1), confirmed via correlation test (0.01)
- **No train/test leakage**: CV gap = 64 bars > 16-bar label horizon
- **3-window validation**: Train (pre-2024) -> Select (2024-H1) -> Validate (2025-H2)
- **Return concentration**: Top trades drive disproportionate PnL -- normal for momentum
- **Realistic 10-day expectation**: uncertain; edge is real but small per trade

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
| Live data + features | `bot/data/live_fetcher.py` |
| Feature computation | `bot/data/features.py` |
| Data download | `scripts/download_data.py` |
| Train alt models | `scripts/train_alt_15m.py` |
| Portfolio backtest | `scripts/backtest_portfolio.py` |
| Single-model backtest | `scripts/backtest_15m.py` |

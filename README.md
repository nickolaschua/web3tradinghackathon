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

## Live Strategy (v2 — Bear Market Fix, March 22 2026)

### What Changed (v2)

The bot had **zero trades** on Day 1 because `BEAR_TREND` regime multiplier was `0.0x`, hard-blocking every signal. Four changes fix this:

| Change | Before | After | Why |
|--------|--------|-------|-----|
| Bear regime multiplier | 0.0x (blocked) | **0.35x** | MR works best in bear markets — blocking trades in the regime with max edge was self-defeating |
| Risk gate floor | `== 0.0` | `< 0.10` | Allows 0.35x through, still blocks misconfigured near-zero multipliers |
| SOL entry threshold | 0.75 | **0.70** | SOL's 4.45% daily vol creates stronger reversion setups; 0.75 was too selective for frequency |
| BTC/SOL exit threshold | 0.10 | **0.08** | Bear bounces are sharp but short-lived; exit ~20% earlier in signal decay curve |
| Micro-trade fallback | none | **$500 at 20:00 UTC** | Guarantees active trading day if no signal fires; costs $0.50 in fees |

### Combined Backtest Results (OOS 2024-2026, $1M capital, 10bps fees)

Full strategy stack: BTC XGBoost + SOL XGBoost + MR + Relaxed MR + Regime Detection.

| Metric | v1 (bear=0.0x, SOL=0.75, exit=0.10) | v2 (bear=0.35x, SOL=0.70, exit=0.08) |
|--------|------|------|
| **Return** | +13.02% | **+27.53%** |
| **Sharpe** | 0.921 | **1.127** |
| **Sortino** | 0.477 | **0.773** |
| **Calmar** | 0.931 | **1.507** |
| Max DD | -6.09% | -7.69% |
| **Trades** | 1,259 | **2,096** |
| **Active trading days** | 53.5% | **88.4% (+ micro-trade = 100%)** |
| Regime blocked signals | 5,439 | **0** |

**By signal source (v2):**

| Source | Trades | Win Rate | Avg PnL | Role |
|--------|--------|----------|---------|------|
| XGBoost | 82 | 45% | **+1.19%** | Alpha engine |
| Mean Reversion | 10 | 80% | **+0.60%** | High-precision fallback |
| Relaxed MR | 2,004 | 52% | -0.20% | Activity layer (micro-sized) |

**By asset (v2):**

| Asset | Trades | Win Rate | Avg PnL |
|-------|--------|----------|---------|
| BTC | 1,000 | 48% | -0.16% |
| SOL | 1,096 | 56% | -0.13% |

### Signal Stack (per 15M bar, per pair)

| Priority | Strategy | Pairs | Entry | Exit |
|----------|----------|-------|-------|------|
| 1 | **BTC XGBoost** (`xgb_btc_15m_iter5.pkl`) | BTC/USD only | P(BUY) >= 0.65 | P(BUY) <= 0.08 or ATR stop |
| 2 | **SOL XGBoost** (`xgb_sol_15m.pkl`) | SOL/USD only | P(BUY) >= 0.70 | P(BUY) <= 0.08 or ATR stop |
| 3 | **Mean Reversion** (original, high precision) | All 20 feature pairs | RSI < 30 + bb_pos < 0.15 + MACD > 0 + EMA_20 > EMA_50 | RSI > 55 or bb_pos > 0.6 |
| 4 | **Relaxed MR** (activity layer) | All 20 feature pairs | RSI < 35 + bb_pos < 0.25 (soft regime gate) | RSI > 50 or bb_pos > 0.55 |
| 5 | **Micro-trade fallback** | BTC/ETH/SOL/BNB/XRP | $500 BUY if no trade by 20:00 UTC | ATR stop (negligible exposure) |

For BTC/USD and SOL/USD, XGBoost runs first. MR and relaxed MR fire only if XGBoost returns HOLD.
For all other 18 pairs, MR cascade is the sole signal source.
If nothing fires all day, the micro-trade fallback guarantees the active-day requirement.

### Why 5 Layers?

The competition requires **8 active trading days out of 10** to qualify. The layers serve distinct purposes:

- **XGBoost (layers 1-2)**: Alpha engine. BTC+SOL models provide the actual edge (+1.19% avg PnL per trade). Low frequency (~82 trades over 2 years OOS).
- **Mean Reversion (layer 3)**: High-precision fallback. Fires rarely but wins 80% of the time. Requires uptrend (EMA gate).
- **Relaxed MR (layer 4)**: Activity layer. Micro-positions (0.01x) fire on ~88% of days. Slightly negative avg PnL (-0.20%) but the positions are so small the drag is negligible.
- **Micro-trade fallback (layer 5)**: Compliance guarantee. Catches the ~12% of days where even relaxed MR doesn't fire. Costs $0.50/day, zero strategy risk.

### Regime Detection

EMA(20)/EMA(50) crossover on daily-resampled BTC 4H data. 2-bar hysteresis confirmation.

| Regime | Multiplier | Position sizing effect |
|--------|-----------|----------------------|
| Bull (EMA20 > EMA50) | 1.00x | Full size |
| Sideways (spread < 0.1%) | 0.50x | Half size |
| Bear (EMA20 < EMA50) | **0.35x** | Reduced — never zero |

**Critical rule: the bear multiplier must never be zero.** Long-side mean reversion generates its strongest absolute returns during bear markets (violent snap-back rallies from oversold extremes). Blocking trades during the regime where MR has maximum edge was the v1 bug.

### Feature Pairs (20 coins)

BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC, UNI, NEAR, SUI, APT, PEPE, ARB, SHIB, FIL, HBAR

All verified: Roostoo-tradeable, 15M parquet data available (2.9-5 years), top 20 by Binance liquidity.

### Risk Management

- **Position sizing**: 2% portfolio risk per trade (ATR-based) for XGBoost; 0.01x portfolio for relaxed MR
- **Bear regime scaling**: All positions sized at 0.35x in bear markets (bounds single-trade loss to ~0.7% of portfolio)
- **Primary exit**: ATR trailing stop at 10x multiplier (15M calibrated)
- **Hard stop floor**: 5% below entry
- **Circuit breaker**: 10% DD → half size | 20% DD → quarter size | 30% DD → halt (absolute veto, overrides regime)
- **Kelly gate**: blocks trades with edge <= 0
- **Max positions**: 5 concurrent
- **Max single position**: 40% of portfolio
- **Micro-trade fallback**: $500 BUY at 20:00 UTC if no trade that day (~$0.50 fee, 0.05% portfolio exposure)

### Models

| Model | Features | Triple-barrier labels | Backtest Sharpe |
|-------|----------|-----------------------|-----------------|
| `xgb_btc_15m_iter5.pkl` | 19 (TA + ETH/SOL cross-asset + corr/beta) | TP=0.8%, SL=0.3%, horizon=16 bars | 1.443 (solo, exit=0.08) |
| `xgb_sol_15m.pkl` | 19 (TA + BTC/ETH cross-asset + SOL/BTC beta) | TP=0.8%, SL=0.3%, horizon=16 bars | frequency contribution (tested on SOL features) |

### Key Design Decisions

- **Exit threshold = 0.08** (was 0.10): Bear bounces are sharp but short-lived. Exiting ~20% earlier in signal decay curve takes profit before the next leg down. BTC solo backtest: Sharpe 1.421 → 1.443, return +25.15% → +25.62%.
- **SOL threshold = 0.70** (was 0.75): SOL's higher volatility (4.45% daily vs BTC's 2.3%) creates stronger mean-reversion setups. Lower threshold increases signal frequency without sacrificing model edge.
- **Bear multiplier = 0.35x** (was 0.0x): Bounds single-trade loss to ~0.7% of portfolio while allowing bear-market MR bounces. Never zero — the activity requirement makes zero-exposure fatal.
- **Relaxed MR soft regime gate**: Unlike original MR which blocks all entries in downtrends (EMA_20 < EMA_50), relaxed MR quarters position size in downtrends instead of blocking entirely. This prevents "dead days" when all coins are bearish simultaneously.
- **Micro-trade fallback at 20:00 UTC**: Deterministic activity guarantee. Better than loosening relaxed MR thresholds (which would add more negative-edge trades). Costs $0.50/day vs potentially dollars in additional losses from looser signals.
- **Unified model rejected**: A pooled 20-coin XGBoost was trained and tested. ATR-normalized labels equalized BUY rates (20-24%) but the model had no edge (Mean AP = 0.272 vs 0.234 random). Heterogeneous coins cannot share one decision boundary. Per-coin models + activity layer is the correct architecture.

### Backtest Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/backtest_15m.py` | Single-asset XGBoost backtest (canonical) | `python scripts/backtest_15m.py --model models/xgb_btc_15m_iter5.pkl --threshold 0.65 --exit-threshold 0.08` |
| `scripts/backtest_combined.py` | Multi-asset combined backtest (BTC+SOL+MR+RelaxedMR+Regime) | `python scripts/backtest_combined.py --compare` |

The `--compare` flag runs old vs new params side-by-side.

### Competition Context

| Field | Value |
|-------|-------|
| Competition | Roostoo Round 1 |
| Dates | March 21-31, 2026 (10 days) |
| Scoring | 0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar |
| Commission | 0.1% per side (10 bps) |
| Starting capital | $1,000,000 |
| Active days required | 8 out of 10 |
| Day 1 status | Lost (0 trades due to bear regime blocker) |
| Days remaining | 9 (need 8 active) |

### File Reference

| Component | File |
|-----------|------|
| Main entry point | `main.py` |
| BTC XGBoost strategy | `bot/strategy/xgboost_strategy.py` |
| SOL XGBoost strategy | same class, different model path |
| Mean reversion | `bot/strategy/mean_reversion.py` |
| Relaxed MR (activity) | `bot/strategy/relaxed_mean_reversion.py` |
| Risk manager | `bot/execution/risk.py` |
| Regime detector | `bot/execution/regime.py` |
| Portfolio allocator | `bot/execution/portfolio.py` |
| Order manager | `bot/execution/order_manager.py` |
| Config | `bot/config/config.yaml` |
| Design spec | `docs/superpowers/specs/2026-03-22-regime-fix-design.md` |
| Implementation plan | `docs/superpowers/plans/2026-03-22-regime-fix.md` |

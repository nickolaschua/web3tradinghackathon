# Cross-Sectional Momentum & Contrarian Rotation — Post-Mortem

**Date:** 2026-03-22
**Status:** ABANDONED — shipped hybrid XGBoost + relaxed MR instead
**Time spent:** ~8 hours of research, implementation, and backtesting

---

## 1. What We Were Trying to Solve

The competition requires **8 active trading days out of 10**. Our proven BTC+SOL XGBoost models (portfolio Sharpe 1.814, Sortino 2.725) only generate ~1 trade per 10 days at optimal thresholds. In 62% of 10-day windows, the portfolio has **zero return** — it just sits in cash. Lowering thresholds to increase frequency (BTC=0.55) destroys the edge (50/50 win rate, median Sortino ~0).

We needed a strategy that generates **daily trades with positive edge** across 20 coins.

---

## 2. Approach 1: Unified 20-Coin XGBoost Model

### Hypothesis
Train one XGBoost classifier on pooled 15M data from all 20 coins. ATR-normalized triple-barrier labels would equalize BUY rates across coins with different volatilities.

### What We Built
- `scripts/train_unified_20coin.py` — training pipeline with ATR-normalized labels
- New feature functions: `compute_market_context_features()`, `compute_coin_identity_features()` in `bot/data/features.py`
- 21 features: 13 per-coin technicals + 4 market context (BTC/ETH returns) + 4 coin identity (btc_corr_30d, relative_vol, vol_rank, liquidity_tier)

### Results

| TP/SL Multiplier | BUY Rate | CV Mean AP | Random Baseline | Verdict |
|------------------|----------|------------|-----------------|---------|
| 2.5x / 0.8x ATR | 23.4% | 0.272 | 0.234 | Barely above random |
| 4.0x / 0.5x ATR | 10.9% | 0.139 | 0.109 | Barely above random |

### Why It Failed
**Pooling heterogeneous coins destroys signal.** BTC, PEPE, and HBAR have fundamentally different price dynamics. A single model cannot learn one decision boundary that works across all of them. The ATR normalization successfully equalized BUY rates (20-24% across all coins) but the model couldn't distinguish "this setup will profit" from noise.

The prior BTC-only model worked (Sharpe 1.192) because it was purpose-built with BTC-specific features (eth_btc_corr, eth_btc_beta). Those features lose meaning when applied to arbitrary coins.

### Key Learning
Per-coin models preserve edge. Pooled models destroy it. Don't try to force heterogeneous assets into one classifier.

---

## 3. Approach 2: Cross-Sectional Momentum Rotation

### Hypothesis
Rank all 20 coins by recent momentum every 4H. Hold top 4 with inverse-volatility weighting. Buffer zones to control turnover. Generates daily trades by construction (rebalancing = trading).

### Research Foundation
- Dobrynskaya (2023): Crypto momentum persists 1-4 weeks, reverses after ~1 month
- Liu, Tsyvinski & Wu (2022): Crypto three-factor model with ~3% weekly momentum payoffs
- Jia et al. (2026): Nearness-to-30-day-high dominates momentum as return predictor (~130 bps/week)
- Novy-Marx & Velikov (2016): Buffer zones are the most effective cost-mitigation technique

Full research in `docs/research/compass_artifact_*.md` and `docs/research/cross-sectional-momentum-strategy.md`.

### What We Built
- `bot/strategy/momentum_signals.py` — 8 pure signal functions (sharpe_momentum, nearness_to_high, residual_momentum, composite score, regime flag, IC computation)
- `scripts/ic_analysis.py` — Spearman rank IC analysis across 20 coins
- `scripts/backtest_momentum_rotation.py` — full backtest with 10-day window analysis
- Design spec with 4-component composite score, 4-layer crash protection, buffer zones, regime flag

### IC Analysis Results (Formation: 2024-01 to 2025-06)

| Component | IC | t-stat | Status |
|-----------|-----|--------|--------|
| sharpe_48h (momentum) | **-0.0162** | -3.26 | **NEGATIVE — momentum anti-predictive** |
| sharpe_168h (momentum) | **-0.0156** | -3.09 | **NEGATIVE** |
| residual momentum | **-0.0125** | -2.63 | **NEGATIVE** |
| nearness (price anchor) | +0.0033 | +0.58 | Barely positive |

**All three momentum components had negative IC.** At 4H horizons on these 20 coins, momentum is anti-predictive — coins that went up tend to go DOWN next period (short-term reversal). This is consistent with Zaremba et al. (2021) and Fičura (2023) who documented sub-daily reversal in crypto.

### Key Learning
Crypto momentum works at weekly horizons for large caps, but **reverses at sub-daily horizons**. Our 4H rebalancing cadence sits in the reversal zone. The research warned about this ("skip the most recent 1-2 bars") but the effect was stronger than expected — skipping 1 bar wasn't enough to overcome the reversal.

---

## 4. Approach 3: Contrarian Rotation (Flipped Momentum)

### Hypothesis
If momentum has negative IC, the **contrarian signal** (buy recent losers, sell recent winners) should have positive IC. Flip the ranking and buy oversold coins that bounce.

### IC Analysis Results (Contrarian)

| Component | IC | t-stat | Status |
|-----------|-----|--------|--------|
| contra_48h (buy losers) | **+0.0162** | **+3.26** | Positive, significant |
| contra_168h | **+0.0156** | +3.09 | Positive, significant |
| contra_residual | **+0.0125** | +2.63 | Positive |
| nearness | +0.0033 | +0.58 | Positive |

All 4 components positive. IC-derived weights:
```
contra_48h: 0.335 | contra_168h: 0.323 | contra_residual: 0.259 | nearness: 0.083
```

### IC Stability Across Periods

| Period | contra_48h IC | t-stat | Market |
|--------|--------------|--------|--------|
| Formation 2024 H1 | +0.039 | +1.82 | Bull |
| Formation 2024 H2 | +0.019 | +0.89 | Mixed |
| Formation 2025 H1 | +0.039 | +1.85 | Bull |
| **Holdout 2025 H2** | **+0.023** | +1.21 | **Bear (-40% BTC)** |
| **Holdout 2026 Q1** | **+0.005** | +0.18 | **Bear** |

**Signal was positive even in the holdout** — the contrarian effect is real. But it weakened significantly in the most recent bear market period.

### Backtest Results (Holdout: 2025-06 to 2026-03)

**Baseline (no fixes):**
```
Sharpe: -0.299 | Sortino: -0.421 | Return: -3.0% | MaxDD: -12.2% | Trades: 410
```

Negative returns despite positive IC. Why?

### Root Cause Analysis

**1. Bear market overwhelms contrarian edge.** BTC dropped 39.8% during the holdout. The strategy is long-only — it must hold crypto. Even buying the "best" oversold coins, they're all declining in a -40% market. The IC says the ranking is correct (oversold coins DO bounce more than others), but the absolute return of ALL coins is negative.

**2. Commission drag: 5.1% total.** 410 trades × 5 bps × avg position size = $51K on $1M. When gross alpha from IC=0.016 is maybe +1-2%, commissions eat it entirely.

**3. Crash protection triggers too late.** Circuit breaker goes flat at 12% drawdown — but by then you've already lost 12%. The BTC TSMOM filter (7-day EMA < -5%) doesn't catch slow grinds.

---

## 5. Three Fixes Attempted

### Fix 1: Aggressive Regime Filter
Go flat when BTC 30-day return drops below threshold (catch sustained bear markets early).

| Threshold | Sharpe | Sortino | Return |
|-----------|--------|---------|--------|
| btc_30d < -5% | +0.079 | +0.114 | -0.2% |
| btc_30d < -8% | -0.261 | -0.368 | -2.7% |
| btc_30d < -10% | -0.299 | -0.421 | -3.0% |

Only the most aggressive (-5%) helped — it keeps the strategy in cash during most of the bear market. But the result is barely breakeven (-0.2%).

### Fix 2: Fewer Trades (Less Frequent Rebalancing)
Reduce commission drag by rebalancing less often.

| Frequency | Sharpe | Sortino | Return | Trades |
|-----------|--------|---------|--------|--------|
| 4H (baseline) | -0.299 | -0.421 | -3.0% | 410 |
| 8H | -0.520 | -0.714 | -5.0% | 228 |
| 12H | -0.377 | -0.531 | -4.0% | 169 |
| 24H | -0.457 | -0.637 | -4.3% | 111 |

**Made things worse.** Slower rebalancing misses the contrarian bounce — the whole edge is in catching quick oversold reversals. Holding longer in a bear market = more losses.

### Fix 3: Cash Buffer (Max Exposure Cap)
Never invest more than X% of portfolio, keep the rest as cash.

| Max Exposure | Sharpe | Sortino | Return | Trades |
|-------------|--------|---------|--------|--------|
| 100% (baseline) | -0.299 | -0.421 | -3.0% | 410 |
| **80%** | **+0.482** | **+0.711** | **+6.2%** | 2277 |
| 70% | +0.289 | +0.420 | +2.9% | 2216 |
| 60% | +0.177 | +0.257 | +1.2% | 2382 |
| 50% | +0.159 | +0.228 | +1.0% | 2480 |

**The cash buffer is the dominant fix.** 80% max exposure flips the strategy from -3% to +6.2%. The 20% cash reserve cushions drawdowns in the bear market. More trades happen because the portfolio frequently adjusts positions as exposure limits interact with rebalancing.

### Best Combinations

| Config | Sortino | Sharpe | Return | MaxDD |
|--------|---------|--------|--------|-------|
| **btc-10% + 8H + 70% exposure** | **+0.721** | **+0.497** | **+6.2%** | -12.3% |
| max_exp=0.8 alone | +0.711 | +0.482 | +6.2% | -12.1% |
| btc-10% + 8H + 80% | +0.595 | +0.416 | +5.3% | -14.0% |
| btc-5% + 8H + 70% + tight DD | +0.585 | +0.410 | +3.9% | **-8.2%** |

---

## 6. Why We Didn't Ship the Contrarian Strategy

Despite the best combo showing Sortino +0.721 and +6.2% return during a -40% BTC bear market, we chose not to ship it:

1. **Formation period was also negative** (-2.8% with best combo). The strategy doesn't convincingly work even in-sample. The positive holdout results may be an artifact of the cash buffer protecting against the bear market rather than the contrarian signal generating alpha.

2. **Active days only 46%** — still doesn't hit 8/10 without the relaxed MR layer. Would need to add MR on top, adding complexity.

3. **The proven XGBoost models are strictly better** when they fire. Sharpe 1.814 vs 0.497. The problem was never signal quality — it was signal frequency.

4. **Competition time pressure.** Round 1 started Mar 21. Every hour debugging a marginal strategy is an hour not trading with proven models.

---

## 7. What We Shipped Instead

**Hybrid approach:** BTC+SOL XGBoost (proven alpha) + relaxed MR activity layer (8/10 active days).

```
Signal cascade:
  BTC XGBoost (t=0.65, exit=0.10)     → proven Sharpe 1.814 in portfolio
    → SOL XGBoost (t=0.75, exit=0.10) → adds diversification
      → Original MR (RSI<30)          → high precision, rare
        → Relaxed MR (RSI<35, 0.01x)  → activity coverage
```

Backtest: Sharpe 1.567, Sortino 2.266, 99% active days, worst 10-day window = 9/10.

The XGBoost models carry the alpha. The relaxed MR ensures we hit 8/10 active days with near-zero drag (0.01x position sizes). In any given 10-day competition window, we're betting that BTC/SOL will make at least 1-2 tradeable moves.

---

## 8. Files Produced During This Effort

### Kept (useful for future iterations)
| File | Purpose |
|------|---------|
| `bot/strategy/momentum_signals.py` | Pure signal functions — reusable for any future rotation strategy |
| `scripts/ic_analysis.py` | IC analysis tool — reusable for any signal validation |
| `scripts/backtest_momentum_rotation.py` | Contrarian backtest framework |
| `scripts/sweep_fixes.py` | Fix sweep framework |
| `docs/research/compass_artifact_*.md` | Academic research on crypto momentum |
| `docs/research/cross-sectional-momentum-strategy.md` | Pitfall analysis |
| `docs/superpowers/specs/2026-03-22-momentum-rotation-design.md` | Design spec (fully reviewed) |

### Kept from unified model effort
| File | Purpose |
|------|---------|
| `scripts/train_unified_20coin.py` | Pooled XGBoost training — demonstrated ATR-normalized labels work |
| `scripts/backtest_unified_20coin.py` | Multi-coin portfolio backtest framework |
| `bot/data/features.py` (additions) | `compute_market_context_features()`, `compute_coin_identity_features()` |

---

## 9. Lessons for Future Iterations

1. **Per-coin models beat pooled models.** The BTC model works because it learned BTC-specific patterns. Forcing 20 coins into one model averages out the signal.

2. **Crypto momentum reverses at short horizons.** At 4H, the dominant effect is mean-reversion, not momentum. Weekly+ horizons show momentum. Any future rotation strategy should use weekly lookback with daily rebalancing, not 48H lookback with 4H rebalancing.

3. **Cash buffers are the strongest single fix** for long-only strategies in bear markets. Never invest 100% of capital in crypto — a 20% cash reserve dramatically improves drawdown profiles.

4. **IC > 0 doesn't guarantee positive returns.** IC of +0.016 translates to maybe +1-2% gross alpha per year. After 5% commission drag, it's negative. Need IC > 0.03 for a viable strategy after costs.

5. **The competition scoring formula rewards conservatism.** 70% of the score penalizes downside. A simple conservative allocation with tight risk management likely outperforms a complex strategy with higher returns but larger drawdowns.

6. **Don't fight the market.** No long-only strategy can make money when BTC drops 40%. The best you can do is detect the bear market early (regime filter) and go to cash. The contrarian "buy oversold coins" intuition is partially correct (those coins DO outperform relative to others) but absolutely incorrect (they still lose money in aggregate).

7. **Signal source tracking matters.** When running multiple strategies in parallel, tag each trade with its source ([xgb_btc], [relaxed_mr], etc.) so you can diagnose what's working in production.

# Cross-Sectional Momentum Rotation — Design Spec

**Date:** 2026-03-22
**Goal:** Replace the XGBoost+MR hybrid with a cross-sectional momentum rotation strategy that generates consistent daily trades across 20 coins while optimizing for the competition scoring formula (0.4×Sortino + 0.3×Sharpe + 0.3×Calmar).
**Core insight:** This is a denominator-minimization competition. 70% of the score penalizes downside. Avoiding the single worst day matters more than catching the single best day.

**Research sources:**
- `docs/research/compass_artifact_*.md` — Academic citations, signal construction, turnover control
- `docs/research/cross-sectional-momentum-strategy.md` — Pitfall analysis, competition considerations

---

## 1. Signal Construction (4-component composite)

Every rebalancing period, compute a composite score for each coin:

```
final_score = w1 × sharpe_mom_48H
            + w2 × nearness_ratio
            + w3 × sharpe_mom_168H
            + w4 × residual_sharpe_mom_48H
```

**Weights (w1-w4) are determined by IC analysis, not guessed.** Before locking weights, run information coefficient (IC) analysis on each component across the 20-coin universe on 2024-01 to 2025-06 data. Weight by `IC_i / sum(IC_j)`. If one component dominates IC, it gets dominant weight. Starting guess (0.30/0.25/0.25/0.20) is a placeholder until IC analysis is complete.

The IC analysis is the **first backtest to run** — it determines whether all four components contribute or whether some should be dropped.

### 1a. Sharpe momentum (48H and 168H)

```python
log_ret = log(close[t - skip] / close[t - lookback])   # skip last 1-2 bars
vol = std(log_returns, lookback_window)
sharpe_mom = log_ret / (vol + 1e-10)
```

- **48H lookback** = 12 bars at 4H cadence (skip last 1 bar)
- **168H lookback** = 42 bars at 4H cadence (skip last 1 bar)
- Skip avoids sub-daily mean-reversion contamination

### 1b. Nearness-to-recent-high (price anchor)

```python
nearness = current_price / rolling_max(close, 180 bars)   # ~30-day high at 4H
```

Coins near their high (nearness → 1.0) rank highest. Strongest single crypto predictor per Jia et al. 2026 (~130 bps/week).

### 1c. Residual momentum (BTC beta stripped)

```python
# Rolling 42-bar OLS: coin_4h_return = alpha + beta * btc_4h_return + epsilon
# Take the residual of the CURRENT bar only (no look-ahead)
# Cumulate residuals over last 48 bars (12 × 4H = 48H)
# Divide by rolling residual volatility
residual_mom = sum(epsilon[-48:]) / (std(epsilon[-48:]) + 1e-10)
```

**Precision notes:**
- Regression window = 42 bars (7 days) for beta estimation
- Residual accumulation window = 12 bars (48H) for the momentum signal
- Uses rolling expanding-window OLS (not in-sample) to prevent look-ahead
- Floor `std(epsilon)` at 1e-10 to handle coins tracking BTC near-perfectly (e.g. high-beta large caps)
- All inputs are 4H returns, not 15M (4H = native rebalancing cadence)

Isolates coin-specific momentum from market-wide BTC moves. Roughly doubles risk-adjusted alpha vs raw momentum (Blitz et al. 2011).

### 1d. Score smoothing

Apply EMA smoothing to raw scores before ranking:
```python
smoothed_score = 0.4 * raw_score + 0.6 * previous_smoothed_score
```

Prevents ranking instability from causing unnecessary trades. Cuts turnover 30-40%.

---

## 2. Portfolio Construction

### 2a. Holdings: 4 coins

Top 4 by smoothed composite score. Optimal for Calmar (one coin dropping 10% = 2.5% portfolio hit vs 3.3% with 3 coins).

### 2b. Weighting: inverse volatility

```python
weight_i = (1 / vol_i) / sum(1 / vol_j for j in top_4)
```

Allocates less to volatile coins. Directly reduces portfolio variance → improves all three scoring ratios.

### 2c. Buffer zones (turnover control)

- **Buy zone:** Rank 1-4 → enter position
- **Hold zone:** Rank 5-8 → maintain, do not sell
- **Sell zone:** Below rank 8 → exit, replace with highest-ranked non-held coin
- **Minimum swap threshold:** Replacement's smoothed score must exceed current holding's score by ≥ 1 std dev of the score distribution

Expected: ~10-20 actual trades over 10 days (vs ~60 without buffer zones).

---

## 3. Rebalancing Mechanics

### 3a. Frequency: every 4H

Rebalance at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.
Generates 6 evaluation points per day = guaranteed daily activity.

### 3b. Order execution: limit orders with fallback

**Primary:** Place limit orders at current best bid/ask. Fee = 0.05% (half of market order fee).

**Fallback:** If limit order unfilled after 30 seconds, cancel and submit market order at 0.10%. This handles fast-moving coins where a limit at the bid won't fill during a momentum rally.

**Rationale:** Over 20 trades, limit orders save ~1 percentage point vs market-only. But unfilled limits risk missing the signal entirely, which is worse than paying 5 extra bps.

### 3c. Rebalancing procedure

```
1. Compute composite scores for all 20 coins
2. Apply EMA smoothing
3. Apply filters (unlock screen, volume filter)
4. Rank by smoothed score
5. Compare to current holdings:
   a. If a held coin drops below rank 8 → mark for SELL
   b. If an unheld coin enters rank 1-4 AND its score exceeds
      the weakest holding's score by ≥ 1 std → mark for BUY
6. Execute SELLs first (free up capital)
7. Execute BUYs (allocate freed capital)
8. Adjust remaining position sizes to match target weights
   (only if adjustment > $5K or 0.5% of portfolio)
```

---

## 4. Crash Protection (4-layer graduated system)

### 4a. Layer 1: Volatility scaling (always active)

```python
target_daily_vol = 0.025  # ~40% annualized
realized_vol = std(portfolio_returns, last 42 bars)  # 7-day rolling
vol_scalar = min(target_daily_vol / realized_vol, 1.0)  # cap at 1.0, no leverage
```

Applied to all position sizes multiplicatively. **Freed capital sits as cash.** This reduces both returns and volatility — net effect on scoring ratios depends on regime. In high-vol regimes (where vol_scalar is low), the cash drag is acceptable because avoiding drawdowns improves Calmar/Sortino more than the lost return hurts Sharpe.

### 4b. Layer 2: BTC time-series momentum filter

```python
btc_7d_ema_return = EMA(btc_daily_returns, span=7)
if btc_7d_ema_return < -0.05:  tsmom_scalar = 0.0   # go flat — crash
elif btc_7d_ema_return < 0:    tsmom_scalar = 0.50   # caution — BTC trending down
else:                          tsmom_scalar = 1.0    # full exposure
```

*Note: order matters — check the severe condition first.*

### 4c. Layer 3: Cross-sectional dispersion filter

```python
dispersion = std(4H_returns across all 20 coins)
dispersion_pct = percentile_rank(dispersion, last 180 bars)
if dispersion_pct < 20:  # below 20th percentile = low dispersion
    dispersion_scalar = 0.50
else:
    dispersion_scalar = 1.0
```

Low dispersion = all coins moving together = rankings are noise.

**Calibration note:** The 20th percentile threshold must be validated in absolute terms (bps). Compute the actual dispersion value at P20 on 2024-2026 data and verify it meaningfully separates "all coins correlated" from "normal dispersion." If the P20 threshold falls at e.g. 15 bps, verify that below 15 bps, momentum rankings are indeed noisy (low IC). The percentile approach adapts to regime, but the threshold itself should be sanity-checked against market structure.

### 4d. Layer 4: Drawdown circuit breaker

```python
drawdown = (hwm - current_portfolio) / hwm
if drawdown > 0.12:   dd_scalar = 0.0   # go flat
elif drawdown > 0.08: dd_scalar = 0.50
elif drawdown > 0.05: dd_scalar = 0.75
else:                 dd_scalar = 1.0
```

### 4e. Combined exposure

```python
final_exposure = base_weight * vol_scalar * min(tsmom_scalar, dispersion_scalar) * dd_scalar
```

`min()` on TSMOM and dispersion avoids double-counting. Any severe signal → near-zero exposure.

---

## 5. Competition-Specific Features

### 5a. Late-game adjustment (score-aware, not just rank-aware)

Reduce exposure in the final days based on **current score trajectory**, not just leaderboard rank. A team in 3rd place with a big drawdown on day 6 should behave differently than 3rd place with a clean run.

```python
# Primary trigger: protect existing score components
if current_max_drawdown > 0.04 or competition_day >= 7:
    late_game_scalar = 0.5
else:
    late_game_scalar = 1.0
```

The drawdown trigger (>4%) fires independently of the calendar — if we take a hit on day 3, we protect immediately rather than waiting until day 7. Leaderboard position (visible via Roostoo dashboard) is used as a secondary check: if we're outside top 10 by day 7, aggressive play may be warranted instead of conservative.

### 5b. Asymmetric sizing after losses

```python
rolling_3d_return = portfolio_return_last_3_days
if rolling_3d_return < -0.05:  loss_scalar = 0.40
elif rolling_3d_return < -0.03: loss_scalar = 0.60
else:                           loss_scalar = 1.0
```

Reduces positions after losing streaks. Sortino's quadratic penalty on negative deviations makes this high-leverage.

### 5c. Unlock screen integration

Apply existing `apply_unlock_screen()` to filter coins before ranking. SUI/ENA excluded for Round 1.

---

## 6. Universe & Filters

### 6a. Coin universe: 20 coins

Same 20 coins already validated: BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, LTC, UNI, NEAR, SUI, APT, PEPE, ARB, SHIB, FIL, HBAR.

### 6b. Volume filter

Filter out the bottom 4 coins by **Binance 24H volume** (not Roostoo volume, which is simulated). This leaves ~16 eligible coins — deep enough to avoid identical-score problems in low-dispersion environments, while removing coins where momentum may revert rather than persist (Fičura 2023). Use Binance volume data already available via the data pipeline.

### 6c. Unlock filter

Remove coins with upcoming unlocks (existing unlock_screen.py).

---

## 7. What to Validate in Backtest

| Question | Test | Pass criteria |
|----------|------|---------------|
| Does composite score beat raw momentum? | Compare composite vs raw return ranking | Composite Sortino > raw Sortino |
| Does residual momentum add value? | Compare with/without residual component | With-residual Sharpe > without |
| Does nearness signal help? | Compare with/without nearness component | Additive improvement |
| Do buffer zones reduce turnover enough? | Measure trades with/without buffer | <20 trades over 10-day windows |
| Is commission drag survivable? | Total fees / gross return | Fees < 30% of gross alpha |
| Does crash protection work? | Compare managed vs unmanaged MaxDD | Managed MaxDD < 8% |
| 4H vs 8H rebalancing? | Compare frequencies | Similar Sortino with fewer trades = prefer 8H |
| Limit order fill rate? | N/A in backtest (assume fills) | Test in paper trading |
| **Full strategy vs BTC hold (PRIMARY)** | Run full strategy across ALL overlapping 10-day windows in 2024-2026. Compare distribution of 10-day competition scores vs BTC buy-and-hold | Median competition score > BTC hold; P5 outcome > -5% return |
| What fraction of windows beat BTC? | Count windows where strategy score > BTC score | > 60% of windows |
| Worst-case 10-day window | P5 of 10-day return distribution | > -8% |
| Sensitivity to dispersion threshold | Sweep 10th/15th/20th/25th percentile | Score stable across thresholds |
| Sensitivity to drawdown flat trigger | Sweep 8%/10%/12%/15% | Score stable across thresholds |

---

## 8. Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| Create | `bot/strategy/momentum_rotation.py` | MomentumRotationStrategy class |
| Create | `scripts/backtest_momentum_rotation.py` | Backtest with 10-day window analysis |
| Modify | `main.py` | Wire momentum rotation as primary strategy |
| Modify | `bot/config/config.yaml` | Rebalancing params, crash protection thresholds |
| Keep | `bot/strategy/xgboost_strategy.py` | Retained but demoted (optional overlay) |
| Keep | `bot/strategy/relaxed_mean_reversion.py` | Removed from cascade (momentum handles activity) |

---

## 9. Market Regime Flag (competition-open calibration)

At competition open (day 1), compute a simple regime flag using BTC's 30-day return and 7-day volatility. This determines initial weight emphasis:

```python
btc_30d_ret = log(btc_close[-1] / btc_close[-180])  # 30 days at 4H
btc_7d_vol = std(btc_4h_returns[-42:])               # 7 days at 4H

if btc_30d_ret > 0.05 and btc_7d_vol > median_vol:
    regime = "HIGH_VOL_TREND"     # 48H momentum gets more weight
elif btc_30d_ret > 0 and btc_7d_vol < median_vol:
    regime = "LOW_VOL_TREND"      # nearness signal dominates
elif btc_30d_ret < -0.05:
    regime = "BEARISH"            # max crash protection, minimal exposure
else:
    regime = "SIDEWAYS"           # balanced weights
```

This adds one degree of freedom and is interpretable. The competition is only 10 days — getting the opening-day regime call right matters a lot. The regime flag adjusts IC-derived weights by ±20% (not a full override).

---

## 10. Implementation Order

1. **Momentum signal functions** — compute composite scores, residual momentum, nearness
2. **Backtest script** — simulate 10-day windows across OOS period, sweep parameters
3. **Validate** — check all gates in Section 7
4. **Strategy class** — MomentumRotationStrategy with buffer zones, crash protection
5. **Wire into main.py** — replace signal cascade
6. **Paper trade** — verify limit order fills on Roostoo before going live

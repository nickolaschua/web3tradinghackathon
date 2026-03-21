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
final_score = 0.30 × sharpe_mom_48H
            + 0.25 × nearness_ratio
            + 0.25 × sharpe_mom_168H
            + 0.20 × residual_sharpe_mom_48H
```

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
# Rolling regression: coin_return = alpha + beta * btc_return + epsilon
# Use 168-bar (7-day) window
residual = cumulative_epsilon / std(epsilon)
```

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

Applied to all position sizes multiplicatively.

### 4b. Layer 2: BTC time-series momentum filter

```python
btc_7d_ema_return = EMA(btc_daily_returns, span=7)
if btc_7d_ema_return < 0:     tsmom_scalar = 0.50
elif btc_7d_ema_return < -0.05: tsmom_scalar = 0.0  # go flat
else:                           tsmom_scalar = 1.0
```

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

### 5a. Late-game adjustment (leaderboard-aware)

If ahead on the leaderboard by day 7-8, reduce exposure to 50-60% to lock in favorable risk-adjusted ratios. The marginal return from final days is not worth the drawdown risk.

```python
if competition_day >= 7 and leaderboard_position <= 5:
    late_game_scalar = 0.5
else:
    late_game_scalar = 1.0
```

Leaderboard is visible mid-competition — can be checked via Roostoo dashboard.

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

Before ranking, filter to top two-thirds by 24H trading volume (~13 coins). Removes illiquid coins where momentum may revert rather than persist (Fičura 2023).

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

## 9. Implementation Order

1. **Momentum signal functions** — compute composite scores, residual momentum, nearness
2. **Backtest script** — simulate 10-day windows across OOS period, sweep parameters
3. **Validate** — check all gates in Section 7
4. **Strategy class** — MomentumRotationStrategy with buffer zones, crash protection
5. **Wire into main.py** — replace signal cascade
6. **Paper trade** — verify limit order fills on Roostoo before going live

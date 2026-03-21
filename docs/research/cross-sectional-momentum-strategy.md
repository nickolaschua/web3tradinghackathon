# Cross-Sectional Momentum Strategy — Research & Critical Analysis

**Date:** 2026-03-22
**Status:** SUPERSEDED by `compass_artifact_*.md` for signal design. This file retained for pitfall analysis.
**Purpose:** Initial research document. See companion file for academic-backed upgrades (residual momentum, price anchoring, buffer zones, graduated crash protection).

---

## 1. What Is Cross-Sectional Momentum?

Cross-sectional momentum ranks assets by recent performance and takes long positions in relative winners. Unlike time-series momentum ("is BTC going up?"), cross-sectional momentum asks "which coins are outperforming relative to others?"

**Core mechanic:**
1. At each rebalancing period, compute a momentum score for each coin
2. Rank coins by score
3. Go long the top N coins with equal weight (or risk-weighted)
4. Close positions in coins that drop out of the top N
5. Repeat every rebalancing period

**Why it generates daily activity:** Rebalancing inherently creates trades. Even if the same coins stay in the top N, partial rebalancing (adjusting weights) counts as activity. If the ranking shifts, entries + exits on multiple coins occur simultaneously.

---

## 2. Strategy Parameters & Design Choices

### 2a. Momentum Lookback Window

How far back to measure performance when ranking.

| Window | Crypto behavior | Risk |
|--------|----------------|------|
| 1H (4 bars at 15M) | Noise-dominated, whipsaw-prone | Too short — ranking changes every bar |
| 4H (16 bars) | Captures intraday trends | Moderate — may catch short-term reversals |
| **12H (48 bars)** | Balances trend capture vs noise | **Good starting point** |
| 24H (96 bars) | Standard daily momentum | Well-documented in literature |
| 48H (192 bars) | Slower momentum, fewer rebalances needed | May miss rapid crypto regime changes |
| 7D (672 bars) | Weekly momentum | Too slow for 10-day competition |

**Recommendation:** Test 12H, 24H, and 48H. Literature suggests 1-day to 1-week lookback works best in crypto.

**Critical issue — lookback vs holding period:** Academic research shows momentum profits are highest when lookback period ≈ holding period. If you rank on 24H momentum but hold for only 4H, you may be entering too late or exiting before the trend completes.

### 2b. Rebalancing Frequency

How often to re-rank and adjust positions.

| Frequency | Trades/day | Commission drag | Tracking quality |
|-----------|-----------|-----------------|-----------------|
| Every 15M (every bar) | ~96 | Catastrophic (9.6% daily at 10bps) | Perfect tracking |
| Every 1H (4 bars) | ~24 | Very high (~2.4% daily) | Good tracking |
| **Every 4H (16 bars)** | **~6** | **Manageable (~0.6% daily)** | **Reasonable** |
| Every 8H (32 bars) | ~3 | Low (~0.3% daily) | May lag regime shifts |
| Every 24H (96 bars) | ~1 | Minimal | May miss intraday moves |

**Recommendation:** 4H rebalancing. Generates 6 rebalances/day = guaranteed daily activity. Commission drag of ~0.6%/day is significant over 10 days (6%) but manageable if momentum alpha > drag.

**Critical issue — commission is the enemy:** At 0.1% taker fee (10 bps), each round-trip costs 20 bps. With 5 positions rebalanced every 4H (some held, some swapped), assume ~3 position changes per rebalance = 6 round-trips/day = 1.2% daily commission. Over 10 days that's **12% drag**. The momentum alpha MUST exceed this.

### 2c. Number of Holdings (Top N)

How many coins to hold at any time.

| N | Diversification | Alpha concentration | Turnover |
|---|----------------|-------------------|----------|
| 1 | None — all-in on winner | Maximum — but one bad pick kills you | Low (only changes when #1 changes) |
| 3 | Moderate | Good balance | Moderate |
| **5** | **Good** | **Diluted but stable** | **Moderate-high** |
| 10 | Over-diversified (half the universe) | Very diluted — approaches market return | High |

**Recommendation:** 3-5 coins. Literature suggests top quintile (top 20% = 4 coins out of 20) is the standard academic choice.

**Critical issue — concentration risk in crypto:** Crypto correlations spike to 0.9+ during market-wide selloffs. Holding 5 "top momentum" coins doesn't protect you if the entire market dumps. Your max drawdown is essentially the market drawdown.

### 2d. Momentum Score Construction

What metric to use for ranking.

| Score | Formula | Pros | Cons |
|-------|---------|------|------|
| Raw return | `close[t] / close[t-L] - 1` | Simple, transparent | Penalizes volatile coins unfairly |
| **Risk-adjusted return** | `return / volatility` | Sharpe-like, favors consistent winners | Better behaved than raw return |
| Log return | `log(close[t] / close[t-L])` | Symmetric, additive | Similar to raw but better for large moves |
| Weighted multi-window | `0.5 * ret_12h + 0.3 * ret_24h + 0.2 * ret_48h` | Captures multiple timescales | More parameters to overfit |
| Residual momentum | `return - beta * BTC_return` | Alpha-only momentum (removes market) | Complex, needs rolling beta |

**Recommendation:** Start with risk-adjusted return (return / rolling volatility). This is the standard "Sharpe momentum" used in factor investing.

**Critical issue — momentum crash:** Momentum strategies are vulnerable to sudden reversals ("momentum crashes"). The top performers can gap down violently when a trend reverses. This is the #1 killer of momentum strategies. Mitigation: volatility scaling (reduce position sizes when market vol is high).

### 2e. Position Sizing

How to allocate capital across the top N coins.

| Method | Description | Risk profile |
|--------|-------------|-------------|
| Equal weight | 1/N each | Simple, balanced |
| **Inverse volatility** | More $ in less volatile coins | Lower portfolio variance |
| Momentum-weighted | More $ in higher-ranked coins | Concentrates in strongest trends |
| Equal risk contribution | Each coin contributes equal risk | Best risk-adjusted returns but complex |

**Recommendation:** Inverse volatility weighting. Each coin gets weight proportional to `1/vol_i`. This naturally reduces exposure to volatile coins (which have wider swings and higher commission drag).

### 2f. Entry/Exit Execution

| Approach | Description | Slippage risk |
|----------|-------------|--------------|
| **Market order at rebalance** | Buy/sell immediately at 15M close | Minimal on liquid coins |
| Limit order at mid | Place limit, wait for fill | May not fill — miss the signal |
| TWAP over next hour | Spread execution | Reduces impact but delays entry |

**Recommendation:** Market order at rebalance for competition (simplicity + certainty of fill). The 10 bps taker fee is a known cost.

---

## 3. Known Pitfalls & Failure Modes

### 3a. Commission Drag Exceeds Alpha

**The math:** If rebalancing 4H with 5 positions, assume 50% turnover per rebalance (2.5 positions change). That's 5 single-leg trades × 10 bps = 50 bps per rebalance. At 6 rebalances/day × 10 days = 60 rebalances. If each costs 50 bps on the turned-over portion: total drag depends on turnover rate.

**Realistic estimate:** If 30% of capital turns over each rebalance (some positions stay), that's 30% × 20 bps round-trip × 60 rebalances = **3.6% total commission over 10 days**. Momentum alpha needs to beat this.

**Academic reference:** Crypto momentum studies (e.g., Liu, Tsyvinski & Wu 2019) find 1-week momentum generates ~10-15% monthly returns in top quintile. But these are gross returns before transaction costs and use daily data, not intraday.

**Mitigation:**
- Use wider rebalancing bands (only trade if ranking changes by ≥2 positions)
- Longer holding period (8H or 12H instead of 4H)
- Fewer positions (3 instead of 5)
- Set a minimum turnover threshold (don't trade if position change < 5% of portfolio)

### 3b. Momentum Reversal / Mean Reversion at Short Horizons

Momentum works at medium horizons (1 day to 3 months) but **reverses at very short horizons** (< 1 hour) and very long horizons (> 12 months). At 15M resolution with 4H rebalancing, we're at the boundary.

**Risk:** If the 4H lookback captures mean-reversion rather than momentum, the strategy buys high and sells low.

**Mitigation:**
- Skip the most recent 1-4 bars when computing momentum (avoid "last-hour reversal" effect)
- Use return from bar [-48] to bar [-4] instead of bar [-48] to bar [0]
- Test both momentum and contrarian variants in backtest

### 3c. Crypto Correlation Regime

During bull markets, correlations are moderate (0.5-0.7) and cross-sectional dispersion is high — momentum works well because coins diverge.

During crashes, correlations spike to 0.9+ and everything drops together. Cross-sectional momentum FAILS here because there's no dispersion to exploit — ranking becomes noise.

**Mitigation:**
- Monitor rolling cross-sectional dispersion (std of returns across 20 coins)
- When dispersion is low (< threshold), reduce position sizes or go to cash
- When BTC drops > 3% in 24H, halt rebalancing and go flat

### 3d. Survivorship Bias in Backtesting

We're backtesting on the top 20 coins **as of March 2026**. In 2021-2023, some of these coins (SUI, PEPE, ARB) didn't exist or weren't liquid. The backtest will only cover their live period, but the "top 20" selection itself is forward-looking.

**Mitigation:**
- Accept this bias — we're deploying to these specific coins anyway
- Focus OOS analysis on 2024-2026 where all 20 coins have data
- Don't over-interpret pre-2023 results for newer coins

### 3e. Token Unlock Overhang

Cross-sectional momentum will naturally buy coins with strong recent performance. If a coin rallied pre-unlock and then dumps on the unlock date, momentum will have loaded into it right before the dump.

**Mitigation:** Apply existing unlock screen to filter out coins with upcoming unlocks from the momentum ranking. Already implemented for SUI/ENA.

### 3f. Market Impact on Roostoo

With $1M virtual capital and 5 positions = $200K per coin. On Roostoo (simulated exchange), market impact is presumably zero. But in reality, some mid-cap coins (PEPE, ARB, HBAR) might not have $200K of 15M liquidity. This doesn't matter for competition scoring but limits real-world applicability.

### 3g. Overfitting Rebalancing Parameters

With lookback, rebalancing frequency, N holdings, scoring method, position sizing, and filters — there are dozens of parameter combinations. Sweeping all of them on the same OOS data is guaranteed to overfit.

**Mitigation:**
- Use the first half of OOS (2024-01 to 2025-01) for parameter selection
- Validate on the second half (2025-01 to 2026-03)
- Keep the strategy as simple as possible (fewer params = less overfitting)

---

## 4. Competition-Specific Considerations

### 4a. 10-Day Window Scoring

The competition scores on 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar over 10 days.

- **Sortino** (0.4 weight): Penalizes downside vol. Momentum with inverse-vol sizing naturally reduces downside vol.
- **Sharpe** (0.3 weight): Penalizes total vol. More positions = more diversification = lower vol = better Sharpe if returns are positive.
- **Calmar** (0.3 weight): Return / max drawdown. Momentum crashes are the biggest Calmar killer. Need crash protection.

**Optimal for scoring:** Many small positive trades (high Sharpe), minimal losing streaks (high Sortino), and no large drawdowns (high Calmar). This favors:
- More positions (5 > 3 > 1) for diversification
- Inverse-vol sizing for crash protection
- Conservative rebalancing (don't overtrade)

### 4b. Activity Requirement

8 active trading days out of 10. With 4H rebalancing (6/day), every day has trades. This is solved by construction — no need for an MR activity layer.

### 4c. Combining with Existing XGBoost

**Option 1: Pure momentum (replace XGBoost entirely)**
Simpler, fewer moving parts, guaranteed activity. Risk: if momentum doesn't work in this 10-day window, no fallback.

**Option 2: Momentum as base + XGBoost as overlay**
Use momentum for base allocation (guaranteed activity). When XGBoost fires BUY on BTC/SOL, overweight that coin in the portfolio. When XGBoost is silent, momentum runs alone.

**Option 3: Momentum for non-XGBoost coins, XGBoost for BTC/SOL**
BTC and SOL positions managed by XGBoost (proven alpha). Other 18 coins managed by momentum rotation. Hybrid approach.

**Recommendation:** Option 1 for simplicity. If momentum works, it works for all coins including BTC/SOL. Layering XGBoost on top adds complexity without clear benefit (XGBoost fires ~1 time in 10 days anyway).

---

## 5. Proposed Implementation Outline

```
Every 4H boundary (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC):

1. RANK: For each of the 20 coins:
   - Compute momentum_score = log_return_24h / rolling_vol_24h
   - Adjust for lookback skip: use return from bar[-96] to bar[-4] (skip last hour)

2. FILTER:
   - Remove coins in unlock exclusion list
   - Remove coins where rolling_vol_24h > 2× median_vol (blow-up risk)

3. SELECT: Pick top N coins (N=3-5) by momentum_score

4. SIZE: Allocate capital to each coin:
   - weight_i = (1 / vol_i) / sum(1 / vol_j for j in top_N)
   - target_usd_i = total_portfolio × weight_i
   - Apply max single position cap (e.g., 40%)

5. REBALANCE:
   - Compare target vs current positions
   - Only trade if position change > min_trade_threshold (e.g., $5K or 0.5% of portfolio)
   - Close positions in coins that dropped out of top N
   - Open/adjust positions in new/continuing top N coins
   - Use market orders (taker fee = 10 bps)

6. RISK CHECK:
   - If BTC 24H return < -5%: go to cash (crash protection)
   - Circuit breaker: same as existing (10%/20%/30% drawdown tiers)
```

---

## 6. Expected Performance Characteristics

**Upside scenario (trending market):**
- Coins diverge, top performers keep outperforming
- Momentum captures 60-70% of the best-performing coins' moves
- Estimated 10-day return: +3% to +8%
- Estimated Sharpe: 1.5-3.0 (annualized over 10-day window)

**Base scenario (sideways market):**
- Moderate dispersion, some winners and losers
- Momentum alpha ~= commission drag
- Estimated 10-day return: -1% to +2%
- Estimated Sharpe: -0.5 to 1.0

**Downside scenario (crash / correlation spike):**
- All coins drop together, ranking is noise
- Momentum holds losing positions, rebalancing locks in losses
- Estimated 10-day return: -5% to -15%
- Estimated Sharpe: -2.0 to -5.0
- **This is the catastrophic scenario** — crash protection MUST trigger

---

## 7. Key Questions to Resolve Before Implementation

1. **What momentum lookback works best in recent crypto data (2024-2026)?** Need to backtest 12H, 24H, 48H.
2. **Is the commission drag survivable?** At 10 bps taker, can momentum alpha cover ~3-6% total drag over 10 days?
3. **Does volatility-scaled momentum outperform raw momentum?** Literature says yes but must verify on our data.
4. **What's the optimal N (number of holdings)?** 3, 4, or 5 positions?
5. **How effective is the crash protection filter?** BTC < -5% 24H as a circuit breaker.
6. **Should we skip the most recent 1H when computing lookback?** To avoid short-term reversal effects.
7. **Rebalancing bands — do they reduce turnover enough to matter?** Only trade when rank changes by ≥2 positions.

---

## 8. Comparison to XGBoost Approach

| Dimension | XGBoost (current) | Cross-Sectional Momentum |
|-----------|-------------------|--------------------------|
| Signal source | ML model, 19 features | Price momentum ranking |
| Frequency | ~1 trade / 10 days | ~6 rebalances / day |
| Activity guarantee | No (62% of 10-day windows have 0 trades) | Yes (trades every rebalance by construction) |
| Alpha per trade | High (+1.27% avg) | Low (small systematic edge) |
| Commission sensitivity | Low (few trades) | High (many trades) |
| Drawdown risk | Low (in cash most of the time) | Higher (always invested) |
| Regime dependency | Works in any regime (predicts per-bar) | Needs dispersion (fails in high-correlation crashes) |
| Implementation complexity | High (ML pipeline, feature engineering) | Low (rank, allocate, rebalance) |
| Overfitting risk | High (19 features, triple-barrier labels) | Low (few parameters, well-documented factor) |

---

## 9. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Commission drag exceeds alpha | HIGH | MEDIUM | Wider rebalance bands, longer holding periods |
| Momentum crash / reversal | HIGH | LOW (10 days) | BTC crash filter, volatility scaling |
| All coins correlated (no dispersion) | MEDIUM | MEDIUM | Dispersion filter, go to cash when corr > 0.85 |
| Overfitting lookback/N parameters | MEDIUM | HIGH | Keep simple, validate on holdout period |
| Token unlock trap | LOW | LOW | Existing unlock screen filters |
| Roostoo API rate limit (20 coins × 6 rebalances) | MEDIUM | LOW | Batch polling, use existing rotation logic |

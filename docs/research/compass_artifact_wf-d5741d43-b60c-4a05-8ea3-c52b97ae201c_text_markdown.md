# Crypto cross-sectional momentum: a quant playbook for the Roostoo hackathon

**Your Sharpe-momentum signal is directionally correct but leaves significant alpha on the table.** Removing BTC beta to isolate idiosyncratic momentum, blending a price-anchor proximity signal (ratio-to-recent-high), and layering a graduated composite crash filter can roughly double your risk-adjusted performance based on recent academic evidence. Critically, **70% of the competition scoring formula punishes downside**, making aggressive risk management more valuable than aggressive return-chasing — a structural edge most student teams will miss. The research below covers signal construction, turnover management, crash protection, scoring optimization, and key pitfalls, with specific parameter recommendations calibrated to your 4H rebalancing cadence, long-only constraint, and 10-day horizon.

Important context from the competition info session: the Roostoo exchange is long-only, no leverage, no slippage, **0.1% market orders / 0.05% limit orders**, ~30–60 API calls/min, and real-time Binance price feeds across 66 cryptos. Scoring is `0.4×Sortino + 0.3×Sharpe + 0.3×Calmar`. The no-slippage environment is an exploitable feature — you can rebalance more aggressively than would work in production.

---

## 1. The momentum signal itself needs three upgrades

### Residual momentum: strip out BTC beta

Blitz, Huij & Martens (2011) showed that ranking equities on residual returns (after regressing out Fama-French factors) roughly **doubled risk-adjusted alpha** versus total-return momentum. No published paper applies this directly to crypto, but the case is even stronger here: a single principal component dominated by BTC explains the majority of altcoin return variation. Liu, Tsyvinski & Wu (2022, *Journal of Finance*) formalized a crypto three-factor model — market (CMKT), size (CSMB), momentum (CMOM) — with **~3% weekly momentum payoffs** at 1–4 week horizons. Fieberg et al.'s CTREND factor (2025, *JFQA*) generated a **weekly alpha of 2.62% (t = 4.22)** after controlling for CMKT, confirming that idiosyncratic signal survives market-beta removal.

**Implementation**: Run a rolling regression of each altcoin's 4H return on BTC's 4H return over a **168-bar window (~7 days)**. Use the cumulative residual ε divided by residual volatility σ_ε as your momentum score. This strips out the "BTC is pumping so everything pumps" noise and isolates genuine coin-specific momentum.

### Price-to-recent-high anchoring: the strongest single crypto signal

George & Hwang's (2004) nearness-to-52-week-high predictor has now been tested in crypto. Jia, Simkins, Yan, Zhang & Zhao (2026, *Journal of Banking & Finance*) found that coins near their recent high outperform coins far from it by **~130 basis points per week (value-weighted)**. The cANCHOR factor is not subsumed by standard momentum at any formation period tested, and a four-factor model adding cANCHOR to Liu et al.'s C-3 model is superior. An earlier version (Jia et al. 2022, SSRN #4170936) documented that even **30-day high proximity "fully dominates" momentum, downside risk, and idiosyncratic volatility** as return predictors. The signal works because crypto lacks fundamental anchors (no DCF), so traders use price history as a reference point — anchoring bias is amplified.

**Implementation**: Compute `nearness = current_price / max(price, last 120–180 bars)` — roughly a 20–30 day high at 4H cadence. This maps to the 30-day high variant documented by Jia et al. Coins with nearness close to 1.0 (near their highs) rank highest.

### Multi-horizon composite: weight the 48H signal heaviest

Dobrynskaya (2023, *Journal of Alternative Investments*) documented crypto's "faster metabolism": momentum persists for **1–4 weeks** and reverses after roughly one month, versus 3–12 months in equities. Han, Kang & Ryu (2023, SSRN) pin the optimal lookback/holding pair at **28-day lookback, 5-day holding (Sharpe 1.51)**. Drogen, Hoffstein & Otte (2023, SSRN #4322637) found a 30-day formation / 7-day holding long-only momentum strategy "absolutely crushes" the benchmark.

For your 4H cadence over 10 days, blend three horizons with decay weighting toward the medium term:

| Horizon | Lookback | Weight | Rationale |
|---|---|---|---|
| Short | 12H (3 bars) | 0.15 | Captures fast alpha but noisy |
| Medium | 48H (12 bars) | 0.50 | Sweet spot per Dobrynskaya; strongest documented risk-adjusted returns |
| Long | 168H (42 bars) | 0.35 | 1-week signal, matches Han et al. optimal; robust anchor |

Skip the most recent **1–2 bars (4–8H)** — this avoids the sub-daily mean-reversion zone documented by Zaremba et al. (2021) and Fičura (2023). The skip is non-negotiable: at sub-4H horizons, liquid crypto exhibits momentum, but illiquid coins revert, contaminating rankings.

### Recommended composite score

```
final_score = 0.30 × sharpe_mom_48H  +  0.25 × nearness_ratio  +  0.25 × sharpe_mom_168H  +  0.20 × residual_sharpe_mom_48H
```

This blends return-based momentum (vol-adjusted), price-anchoring, and idiosyncratic momentum. Each component contributes orthogonal information.

---

## 2. Turnover control: look frequently, trade infrequently

### Buffer zones beat reduced rebalancing frequency

Novy-Marx & Velikov (2016, *Review of Financial Studies*) demonstrated that the **buy/hold spread is the single most effective cost-mitigation technique** for momentum. A two-thirds reduction in trading frequency yields only a one-third reduction in costs — far less efficient than simply widening the hysteresis band. Arnott, Li & Linnainmaa (2024, *Financial Analysts Journal*) showed that "priority-best" rebalancing (replacing the weakest holding first with the strongest available replacement) retains the most alpha per unit of turnover, and that momentum alpha "peters out after ~8 months" without rebalancing — in crypto's faster metabolism, the decay is even quicker.

For a **top-4 portfolio from ~30 liquid coins** (filter out bottom-third by volume):

- **Buy zone**: Score ranks top 4 → enter position
- **Hold zone**: Ranks 5–8 → maintain, do not sell
- **Sell zone**: Below rank 8 → exit and replace with highest-ranked non-held coin

This 4-position buffer (roughly 1.5–2× the portfolio size) is supported by the Novy-Marx & Velikov framework. The score-improvement threshold for a swap should be at least **0.2% absolute** — approximately 2× the round-trip limit-order cost of 0.10%.

### Full replacement dominates partial rebalancing in this sim

Garleanu & Pedersen (2013, *Journal of Finance*) showed optimal portfolios trade partially toward an "aim" portfolio, weighting current holdings against the ideal. However, in a **zero-slippage simulated exchange with 0.05% limit-order fees**, the main benefit of partial rebalancing (avoiding market impact) vanishes. Over 10 days with ~60 rebalancing checks, expect **10–20 actual trades** with the buffer zone, costing ~1.0–1.5% total. Gross momentum alpha of **2–5% over 10 days** (extrapolating 1.5–3.5% weekly from Grobys et al. 2025) means trading costs consume only 20–30% of alpha — comfortably profitable.

**Always use limit orders.** The 50% fee reduction (0.05% vs 0.10%) compounds: over 20 trades, limit orders save ~1 percentage point versus market orders. In a competition decided by risk-adjusted scores, every basis point matters.

---

## 3. Crash protection: a four-layer graduated system

A single BTC return threshold is brittle. Academic evidence supports combining multiple signals with graduated position scaling rather than binary on/off logic. Keller & Keuning (2016) showed that composite crash protection (momentum + volatility + breadth) increased Sharpe from 0.63 to **1.76** versus momentum alone.

### Layer 1: Volatility scaling (always active)

Moreira & Muir (2017, *Journal of Finance*) demonstrated that scaling positions by inverse realized variance produces large alphas because **changes in volatility are NOT offset by proportional changes in expected returns**. In crypto specifically, Grobys et al. (2025, *Financial Markets and Portfolio Management*) found volatility-managed momentum increased weekly returns by **>200%** and reduced kurtosis from 121.81 to 68.22. A separate 2025 paper in *Finance Research Letters* found crypto risk-managed momentum Sharpe rose from **1.12 to 1.42** — with the improvement coming primarily from enhanced returns, not just reduced downside (unique to crypto versus equities).

```python
target_daily_vol = 0.025  # ~2.5% daily ≈ 40% annualized
realized_vol = std(portfolio_returns[-42:])  # 7-day rolling
vol_scalar = min(target_daily_vol / realized_vol, 1.0)  # cap at 1.0 — no leverage
```

### Layer 2: BTC time-series momentum filter

Liu & Tsyvinski (2021, *Review of Financial Studies*) found TSMOM in crypto is strongest at **1–4 week horizons**. Huang, Sangiorgi & Urquhart (2024, SSRN) documented volume-weighted TSMOM generating **0.94% per day with annualized Sharpe of 2.17**. Han, Kang & Ryu (2023) confirmed time-series momentum evidence in crypto is strong and that **the momentum effect is concentrated among winners**.

Use a 7-day EMA-smoothed BTC return as the filter. If BTC's smoothed 7-day return turns negative, scale exposure to 50%. Below –5%, go flat. This is a refinement of your current –5% BTC filter — the graduated response avoids the binary whipsaw problem.

### Layer 3: Cross-sectional dispersion filter

When all coins move together (high correlation / low dispersion), cross-sectional momentum has no edge — rankings become noise. Borri (2019, *Journal of Empirical Finance*) found crypto tail-risk connectedness surges above **95%** during crises. Compute rolling cross-sectional standard deviation of 4H returns across your universe. When dispersion falls below its **20th percentile**, reduce exposure by 50%.

### Layer 4: Drawdown circuit breaker (continuous)

This is the single most important layer for Calmar optimization. A 15% drawdown requires 17.6% to recover — essentially impossible in the final days of a 10-day competition. Sadaqat & Butt (2023) showed that stop-loss rules improved crypto momentum returns from **–8.02% to +9.13% monthly** with positive skewness.

```
Drawdown > 5%:  reduce positions to 75%
Drawdown > 8%:  reduce to 50%
Drawdown > 12%: go flat
```

### Composite position sizing

```python
final_exposure = base_weight * vol_scalar * min(tsmom_scalar, dispersion_scalar) * dd_scalar
```

The `min()` on TSMOM and dispersion avoids double-counting overlapping signals. The multiplicative structure means any single severe signal can take you near-zero exposure. **Use OR-logic for triggering, not AND-logic** — in a 10-day window, waiting for confirmation from multiple slow signals means the crash has already happened.

---

## 4. The scoring formula rewards conservative positioning

### 70% of your score punishes downside

Decomposing the composite: Sortino (40%) penalizes downside deviation only; Calmar (30%) penalizes the single worst peak-to-trough drawdown; Sharpe (30%) penalizes total volatility. **Upside volatility hurts only 30% of your score, but downside volatility hurts 100%.** This asymmetry is the key strategic insight. A team generating +5% return with 3% max drawdown will outscore a team generating +15% return with 12% max drawdown on this metric. Big up-days are almost free; a single bad day is catastrophic across all three denominators.

With only ~10 daily observations in the scoring window, each daily return has outsized impact. One –5% day dominates the Sortino denominator (squared negative deviation), destroys the Calmar denominator (becomes the max drawdown), and inflates Sharpe's denominator. **Avoiding even one bad day matters more than capturing several good days.**

### Hold 4 coins, not 3 or 5

ReSolve Asset Management ran 16,116 simulations across 79 momentum universes and found "almost no difference in results between portfolios that hold top 2, 3, or 4 assets" for raw returns, but **3-holding portfolios have materially worse drawdown profiles**. One coin dropping 10% in a 3-coin portfolio creates a 3.3% portfolio drawdown; in a 4-coin portfolio, only 2.5%. Given that 70% of the scoring is downside-focused, the marginal diversification from the 4th holding is high-value. Five holdings dilutes the momentum signal by adding a weaker candidate from a small universe.

Use **inverse-volatility weighting** across the 4 selected coins: `weight_i = (1/σ_i) / Σ(1/σ_j)`. This naturally allocates less to the most volatile coin in your momentum portfolio, directly reducing portfolio variance and improving all three scoring ratios.

### Asymmetric sizing amplifies Sortino

After a losing streak, reducing position sizes reduces downside deviation precisely when it's being measured. Track the rolling 3-day portfolio return. If it falls below –3%, reduce next-period positions to 60% of base; below –5%, reduce to 40%. After a winning day, gradually scale back to 100% over 2 periods. The Sortino denominator uses √(mean of squared negative deviations), so the quadratic penalty on large down-days makes even small reductions during losing streaks highly impactful. Literature on anti-martingale approaches suggests **15–30% Sortino improvement** from this technique.

### The game theory: risk-adjusted play is dominant

Most student teams will optimize for absolute return — concentrating in 1–2 high-beta altcoins, building complex ML models that overfit in 10 days, or running aggressive trend-following on BTC alone. Your edge is structural: **you're optimizing the scoring function, not raw returns**. Even if a team achieves +20% return but with a –15% drawdown and 5% daily vol, your +5% return with –4% max drawdown and 1.5% daily vol will produce a higher composite score. Playing the risk-adjusted game wins whether opponents do or don't play it — a Nash-dominant strategy.

**Late-game adjustment**: If you're ahead by day 7–8, reduce exposure to 50–60% to lock in favorable risk-adjusted ratios. The marginal return from the final 2–3 days is not worth the risk of a drawdown event destroying your Calmar.

---

## 5. Five pitfalls ranked by competition severity

### Mean reversion at short horizons is the #1 danger

This is the most practically critical pitfall. Zaremba et al. (2021, *International Review of Financial Analysis*) documented that cryptocurrencies with low prior-day returns **outperform** high-return coins the next day — daily reversal, not momentum. However, Fičura (2023) showed this is **entirely driven by illiquid coins**. Large, liquid coins exhibit **weekly momentum** (t-stat = 2.33) while small coins show weekly reversal (t-stat = –7.31). Wen, Bouri, Xu & Zhao (2022, *North American Journal of Economics and Finance*) confirmed both intraday momentum and reversal coexist in crypto, with the crossover around 30 minutes to 1 hour.

The implication: **never use a lookback shorter than ~24H for momentum scoring in crypto**, and for the primary signal use 5–7 days. Your 12H lookback option is risky — it sits near the momentum/reversal crossover for smaller coins. Skipping 1–2 bars (4–8H) is essential. Filter your universe to the **top two-thirds by 24H trading volume** before computing scores.

### Ranking instability causes invisible transaction cost leakage

With 20–66 coins and scores differing by tiny margins, small noise causes frequent rank reversals at portfolio boundaries. MSCI research shows rank-based (equal-weight within quintile) approaches create "cliff effects" where marginal rank changes trigger full position replacement. Score-tilt weighting — allocating proportionally to the momentum score rather than equally to the top N — **consistently improves factor exposure** and reduces unnecessary turnover.

**Concrete fix**: Apply EMA smoothing to raw scores before ranking (`smoothed = 0.4 × raw + 0.6 × previous_smoothed`). Set a minimum replacement threshold of **1 standard deviation of the score distribution** — only swap a holding if the replacement's smoothed score exceeds it by this margin. This single change can cut turnover by 30–40% with minimal signal loss.

### Post-2022 momentum is fragile but alive

Grobys et al. (2025), using data through December 2023, found crypto momentum profits are **1.74% per week** but only marginally significant, subject to power-law-distributed crash risk. Fieberg et al. (2023, *Quantitative Finance*) confirm persistent factor momentum across subperiods including post-FTX. The consensus: momentum works in crypto post-2022, especially for large-cap coins and at weekly horizons, but is **more fragile** than pre-2021. Volatility management is no longer optional — it's required to keep the strategy viable.

### Crowding is a low concern in this environment

Lou & Polk (2022, *Review of Financial Studies*) showed that when "comomentum" is high (everyone trading the same stocks), momentum reverts strongly. In the Roostoo sim, however, prices come from Binance — competition participants cannot push prices. The real crowding risk is **correlated drawdowns**: if all teams hold the same coins and momentum reverses, everyone crashes together, eliminating differentiation. Mitigation: use your residual momentum and anchoring signals to differentiate your selections from teams using raw-return momentum.

### Liquidity illusion: exploit it, don't fear it

In a zero-slippage sim, you can rebalance more aggressively and trade smaller coins without penalty. Paybis (2025) documents that real-market altcoin slippage can reach 0.5–2% outside the top 100 — costs that don't exist here. **Exploit the sim by including mid-cap coins in your universe** that would be prohibitively expensive to trade in production. However, document slippage assumptions if you plan to deploy the strategy later.

---

## Conclusion: the integrated strategy architecture

The optimal configuration for this specific competition combines a **four-component momentum signal** (vol-adjusted 48H + price-anchor nearness + vol-adjusted 168H + residual momentum), holding **4 coins** with inverse-volatility weighting, protected by a **four-layer graduated crash system** (vol-scaling → BTC TSMOM → dispersion filter → drawdown circuit breaker). The buffer-zone rebalancing approach (buy top 4, sell only below rank 8) with EMA-smoothed scores controls turnover to ~10–20 trades over 10 days.

The highest-leverage insight is that this is not a return-maximization competition — it is a **denominator-minimization** competition with a return floor. The composite scoring formula's 70% weight on downside metrics means the team that avoids the single worst day wins, not the team that catches the single best day. Target +4–8% return with <5% max drawdown. Use limit orders everywhere. Go conservative in the final 2–3 days if ahead. The academic literature is unambiguous: **volatility-managed, buffer-zone momentum with crash protection dominates unmanaged momentum in crypto by every risk-adjusted measure**.

| Parameter | Recommended value | Source |
|---|---|---|
| Primary lookback | 48H (12 bars), skip 1–2 bars | Dobrynskaya 2023, Han et al. 2023 |
| Secondary lookback | 168H (42 bars) | Drogen et al. 2023, Liu et al. 2022 |
| Nearness anchor | 120–180 bars (20–30 day high) | Jia et al. 2026, 2022 |
| BTC beta window | 168 bars (7 days) | Liu et al. 2022 |
| Holdings | 4 coins | ReSolve 16K-simulation study |
| Buy/sell buffer | Top 4 / below rank 8 | Novy-Marx & Velikov 2016 |
| Score smoothing | EMA α = 0.4 | MSCI score-tilt research |
| Vol target | ~40% annualized | Grobys et al. 2025 |
| Drawdown hard stop | 12% → go flat | Sadaqat & Butt 2023 |
| Order type | Limit only (0.05%) | Competition fee structure |
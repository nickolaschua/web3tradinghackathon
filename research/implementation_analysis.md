# Implementation Analysis: What to Build vs. What to Skip

**Summary:** Both research documents agree on the core hierarchy — TSMOM is the most robust alpha,
funding rate carry is a dying strategy, and on-chain metrics are useful only as regime filters.
Given the existing codebase (XGBoost models trained, momentum and mean-reversion strategies already
built, live data pipeline in place), the decision space is narrow: a handful of low-cost additions
can meaningfully improve the existing system, while most of the more exotic strategies would consume
time better spent hardening what's already there.

---

## What's already built (and therefore "free")

Before deciding what to add, it's worth noting what the codebase already covers:

- **XGBoost models** — both 15M and 4H models exist (`models/xgb_btc_*.pkl`), trained and saved.
- **Momentum strategy** — `bot/strategy/momentum.py`, the core TSMOM signal.
- **Mean reversion strategy** — `bot/strategy/mean_reversion.py`, the complementary signal.
- **Regime detection** — `bot/execution/regime.py`, already classifying market conditions.
- **Risk management** — `bot/execution/risk.py` with ATR stops, 2% risk/trade, circuit breaker,
  Kelly gate (as per the canonical backtest memory).
- **Feature engineering** — `bot/data/features.py`, the OHLCV feature pipeline.
- **Live data fetcher** — `bot/data/live_fetcher.py`, Binance API integration.
- **Backtesting** — `scripts/backtest_15m.py` is the canonical backtest with full live-bot
  risk management stack.

The two research documents are mostly confirming what's already been built rather than introducing
entirely new directions. The question is which *additions* are worth the engineering time.

---

## Realistic to implement

### 1. BTC lead-lag features in altcoin models — HIGH PRIORITY

**What it is:** Feed BTC's prior 1-bar and 6-bar returns as input features when predicting altcoin
returns. Also add rolling 30-day BTC-altcoin correlation and rolling beta.

**Why it's realistic:** Pure OHLCV. Already have the live fetcher pulling BTC data. Adding 4-5
features to `bot/data/features.py` is a single-session task. Both documents agree this is the
dominant market factor and it's free.

**What the research says:** Guo et al. (2024) show BTC's lagged return predicts altcoin returns via
information diffusion. Demir et al. show BTC declines affect altcoins asymmetrically — stronger
negative spillover than positive. The model can learn this asymmetry if BTC return is in the feature
set.

**Caveat:** If the model trades both BTC and altcoins simultaneously, BTC features in altcoin models
create correlated risk. This is already present structurally — it's better to make it explicit and
let XGBoost handle it than to pretend the correlation doesn't exist.

---

### 2. Cross-sectional momentum rank features — MEDIUM PRIORITY

**What it is:** At each 4H bar, compute every coin's 7-day, 14-day, and 28-day return and express
each coin's return as a percentile rank within the universe. Add `ret_7d_rank`, `ret_14d_rank`,
`ret_28d_rank` as features.

**Why it's realistic:** Still pure OHLCV. Requires looping over the coin universe at each bar to
compute ranks, but the live fetcher already pulls data for multiple pairs. Implementation is
straightforward numpy/pandas operations in `features.py`.

**What the research says:** The CTREND factor (Fieberg et al., 2025) outperforms raw momentum and
requires no ML infrastructure beyond the XGBoost model already trained. Cross-sectional rank is one
of its inputs. The document explicitly warns about look-ahead bias in rank computation — ensure
ranks are computed only on completed bars.

**Caveat:** With a small coin universe (the hackathon's 39 coins), quintile sorts are noisy (~8
coins per bucket). Using ranks as continuous features fed into XGBoost avoids this problem entirely.
Do not try to build a standalone quintile-sort portfolio from these ranks — just use them as model
inputs.

---

### 3. Funding rate as supplementary sentiment features — LOW PRIORITY, HIGH CONFIDENCE

**What it is:** Fetch Binance's 8-hour funding rate via `GET /fapi/v1/fundingRate` and add 2-3
features: `funding_rate_ma_24h`, `funding_rate_change_24h`, `funding_rate_cross_zscore` (where a
coin's funding rate stands relative to the universe).

**Why it's realistic:** The API endpoint is free, already have the Binance client in
`bot/api/client.py`, and adding features to the pipeline is well-understood. The timestamp
alignment gotcha (funding settles at 00:00, 08:00, 16:00 UTC) is real but manageable — just use
the most recent settled rate, not the accruing rate.

**What the research says:** Both documents agree that funding rate carry as a *standalone strategy*
is nearly dead post-ETF (Schmeling et al. document 36-97% compression). But as a *contrarian
sentiment feature*, it retains value because it captures behavioral dynamics (crowded longs →
reversal pressure) rather than an arbitrage. Presto Research shows predictive R² essentially zero
at 7-day forward horizons — which is why this is supplementary, not primary.

**Caveat:** Funding rate is highly autocorrelated (AR(1) near 1). The level itself is
uninformative; the change and the cross-sectional z-score are what matter. If only adding two
features, `funding_rate_change_24h` and `funding_rate_cross_zscore` are the two to pick.

---

### 4. Token unlock as a pre-model negative screen — LOW PRIORITY, EASY WIN

**What it is:** Before each rebalance, check Tokenomist.ai (free API or manual check) for any
coins with >1% supply unlocks scheduled in the next 7 days. Zero-weight or heavily underweight
those coins regardless of model signal.

**Why it's realistic:** This is a one-time lookup per competition week, not a real-time feed.
Can be implemented as a hardcoded exclusion list for the competition window if no API integration
time is available. Even manual inspection of Tokenomist.ai before the competition starts would
capture the largest upcoming events.

**What the research says:** Keyrock's 16,000-event analysis shows average -0.3% pre- and
post-unlock, with team/investor unlocks averaging -25% crashes. Importantly, price decline begins
~30 days before the event due to pre-positioning. In a 1-week competition, a large team unlock
scheduled for day 4 is a landmine that the model won't learn to avoid from OHLCV alone.

**Caveat:** Ecosystem/development unlocks are actually slightly positive on average (+1.18%), so
a blanket "avoid all unlocks" rule is suboptimal. Filter specifically for team and investor vesting
unlocks above 1% of supply.

---

### 5. Improving the regime filter for mean reversion — MEDIUM PRIORITY

**What it is:** `bot/execution/regime.py` already classifies market conditions. The research
strongly suggests mean reversion should only trigger inside regime-appropriate conditions —
specifically, only take reversal longs in coins that are in an uptrend (price above 50-bar SMA,
or positive TSMOM), to avoid catching falling knives.

**Why it's realistic:** The regime module already exists. This is a logic change to
`bot/strategy/mean_reversion.py` to gate entries on the regime classifier output. The mean
reversion document section explicitly flags this: "reversal fails catastrophically in strong
trending markets."

**What the research says:** Dobrynskaya (2023) documents that crypto momentum persists 2-4 weeks
before transitioning to reversal — so a mean reversion bet that fights a 2-week trend is not
mean-reversion at all, it's picking a fight with the strongest documented alpha in crypto. The
mandatory trend filter is what separates a functional mean-reversion strategy from a money-losing
one.

---

## Not realistic to implement

### Delta-neutral funding rate carry

**Why not:** The strategy requires holding a spot position and an opposing perpetual futures
position, monitoring margin continuously, and managing both legs across settlement cycles. The
hackathon is almost certainly spot-only or has a simplified execution model (Roostoo API based on
the docs). More importantly, Schmeling et al. document that BTC/ETH basis carry turned *negative*
in early 2025. Implementing a strategy whose alpha is confirmed-dead is not a good use of time.

---

### Full cross-sectional long-short quintile portfolios

**Why not:** Both documents agree: the short leg destroys value in almost all configurations. Five
of 21 tested portfolios were liquidated. Short selling in a hackathon environment adds execution
complexity (borrowing, margin, liquidation risk) for negative expected value. The long-only top
quintile is fine as a feature, but building a long-short portfolio atop a 39-coin universe is
fragile — roughly 8 coins per quintile is too few for stable statistical behavior.

---

### On-chain metrics at 4H granularity (MVRV, STH-SOPR, exchange netflows)

**Why not:** Hourly on-chain data requires CryptoQuant or Glassnode at $100-800/month. Daily MVRV
is free and the documents agree it's a useful *regime filter* at the cycle level (MVRV > 3.7 =
distribution zone, < 1.0 = accumulation). But in a 1-week competition, we're operating inside a
single macro regime phase — MVRV isn't going to swing between zones mid-week. Its value is for
deciding whether to be 100% invested or 50% invested in the competition at all, not for
bar-by-bar decision making.

The exception noted in the first document — Chi et al. (2024) showing USDT net inflows predict
BTC returns at 4H intervals — is interesting but requires paid intraday flow data and a separate
data pipeline. Not realistic under time pressure.

---

### Cointegration-based pairs trading

**Why not:** Palazzi (2025) reports a Sharpe of 3.97 for BTC-ETH pairs, which sounds compelling.
But implementing this correctly requires:
1. Rolling Engle-Granger or Johansen cointegration tests (computationally intensive)
2. Continuous monitoring of spread z-score with two simultaneous positions
3. Knowing when cointegration has broken (pairs break down without warning)
4. Managing two legs simultaneously through the execution system

The BTC-neutral residual mean reversion variant (Plotnik, 2025) is simpler — rolling OLS regression
to strip BTC component, trade the residual — and achieves Sharpe ~2.3. This is tempting but still
represents a meaningful engineering effort on top of what's already built. The risk/reward of
adding a second independent strategy mid-competition (debugging, backtesting, integrating) versus
hardening the existing TSMOM pipeline is unfavorable.

---

### CTREND factor (elastic net aggregation of 28 technical signals)

**Why not:** Fieberg et al. (2025) document a Sharpe of ~1.8 using elastic net to combine 28
technical indicators into a single factor. This is the most ML-intensive alpha in the research.
The problem is that the XGBoost model already *is* a non-linear, high-capacity aggregation of
similar features. Adding 28 hand-crafted indicators and then running elastic net to compress them
back to one signal, before feeding that into XGBoost, is essentially reinventing a step XGBoost
already does internally. The value-add is low relative to the implementation cost.

---

### Triple Barrier labeling

**Why not for now:** The research document argues compellingly that triple barrier labels (take
profit / stop loss / time barrier) produce better-calibrated XGBoost models than fixed-horizon
return labels. This is correct in theory. But the models are already trained. Switching to triple
barrier labeling means redefining the prediction target and retraining from scratch — a full
pipeline rebuild during the competition window. The canonical backtest already uses ATR-based stops
at 10x (from memory), which is conceptually aligned with triple barrier logic. The existing labels
are probably good enough.

---

### Combinatorial Purged Cross-Validation (CPCV)

**Why not:** Lopez de Prado's CPCV is the gold standard for preventing false discovery in
time-series ML. The research document explicitly recommends it over walk-forward validation.
But implementing CPCV correctly is a multi-day software engineering task. The models are already
trained and validated. The appropriate question now is whether the live performance tracks the
backtest — not redesigning the validation infrastructure.

---

### Intraday seasonality (21-23 UTC)

**Why not:** Vojtko & Jarkovska (2023) find abnormal returns in specific UTC hours. The
competition runs on 4H candles — capturing a 2-hour intraday window requires sub-4H data and
precision timing that the current bot may not support. More importantly, the paper scores 2/5 on
persistence in the composite ranking. This is the most fragile signal in the review.

---

## The honest prioritization

Given where the codebase is, the highest-leverage actions in order are:

1. **Harden what's built.** Verify the backtest's fee assumptions (20bps round-trip), confirm
   walk-forward validation prevents look-ahead, check the regime classifier is actually
   gating mean-reversion correctly. A working system with fewer bells is better than a broken
   system with more bells.

2. **Add BTC lead-lag features to altcoin models.** Pure OHLCV, single session, high consensus
   from both research documents that this is the dominant market factor.

3. **Add cross-sectional rank features.** Requires computing ranks across the coin universe at
   each bar. Moderate complexity, high value — the CTREND evidence is the strongest multi-signal
   result in the literature.

4. **Add funding rate sentiment features (2-3 features).** Low effort if the Binance client is
   already set up. Not transformative but costs almost nothing to add.

5. **Manual token unlock check.** 15-minute manual lookup before competition start. Zero
   engineering cost, eliminates one class of unexpected losses.

6. **Tighten mean reversion regime gate.** Logic change to an existing module. Prevents the
   "catching falling knives" failure mode that the research specifically flags.

---

## A note on what the papers get wrong for this context

Both documents are written for general practitioner audiences, not specifically for a hackathon
with a fixed 1-week window and a relatively small universe. A few adjustments to their
recommendations:

**The 28-day lookback is optimal for monthly rebalancing; for a 7-day competition, the 7-14 day
lookback is more relevant.** A 28-day trend tells you where the coin was a month ago. In a
1-week competition, you care about where it's going in the next 1-7 days. The shorter lookbacks
(42-bar = 7d, 84-bar = 14d on 4H data) are the right primary signals.

**The Deflated Sharpe Ratio calculation is useful for post-hoc analysis, not real-time
decision-making.** During the competition, you're executing live trades, not running statistical
tests. The relevant question is "is the model producing signals consistent with its backtest
behavior" — which is answered by monitoring live vs. expected signal distribution, not DSR.

**Concentration wins competitions; diversification manages institutional risk.** The academic
strategies are all designed for multi-year, multi-cycle performance. Both documents note this
implicitly — TSMOM's value is partly in going to cash (avoiding drawdowns), and 48% cash holding
produces near-zero returns in a 1-week window. For the hackathon specifically, the regime has to
be read at the start of competition week, and if it's a trending week, full deployment in the top
2-3 momentum signals beats holding 50% cash for drawdown protection.

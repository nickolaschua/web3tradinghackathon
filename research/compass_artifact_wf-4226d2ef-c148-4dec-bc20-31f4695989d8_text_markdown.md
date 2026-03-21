# Crypto trading alphas that survive scrutiny: a 2022–2025 evidence review

**The most implementable, well-documented crypto alphas for a practitioner operating at $50K–$1M capital are time-series momentum on liquid coins, the perpetual futures funding rate carry trade, MVRV-based cycle timing, and cross-crypto lead-lag signals.** Each relies on data available through free or cheap APIs and requires no ML pipelines, NLP models, or sentiment scrapers. However, alpha decay is real and accelerating: funding rate carry has compressed from a Sharpe of **6.45** (2020–2023) to **negative** in early 2025, and cross-sectional crypto anomalies have weakened **9–76%** in the second half of their sample periods. The strategies below are ranked by a composite score of scalability, simplicity, and persistence — the three criteria that matter most for a capital-constrained practitioner building systems without a data science team.

---

## Time-series momentum is the strongest simple alpha in crypto

The single most robust and well-documented OHLCV-based alpha in crypto is **time-series momentum (TSMOM)** — buying assets whose recent returns are positive and avoiding or shorting those with negative returns. Unlike equities where momentum works on 6–12 month lookbacks, crypto momentum is **short-lived: 1–4 weeks optimal**, with rapid decay thereafter.

Han, Kang & Ryu (2024, SSRN #4675565) found that a 28-day lookback / 5-day holding period TSMOM strategy on liquid coins delivered a **Sharpe ratio of 1.51** versus 0.84 for buy-and-hold, after realistic transaction costs of 15 bps per half-turn on Binance. Critically, the alpha is concentrated in the **long leg** — winners keep winning, while shorting losers is dangerous due to violent mean-reversion bounces. Five of 21 cross-sectional portfolios tested were actually *liquidated* during the sample period, confirming that long-only implementation is strongly preferred.

Zarattini, Pagani & Barbon (2025, SFI Research Paper, SSRN #5209907) independently validated this with a **Donchian channel ensemble trend-following** system applied to the top 20 most liquid coins. Their CTA-style approach achieved a **Sharpe above 1.5 net of fees** and **10.8% annualized alpha over Bitcoin**, using only daily OHLCV data with volatility-based position sizing. This is the closest thing to a "managed futures for crypto" strategy, and it survived the 2022 bear market intact.

Huang, Sangiorgi & Urquhart (2024, SSRN #4825389) added a volume-weighting twist: volume-weighted TSMOM generated **0.94% daily returns with an annualized Sharpe of 2.17**, suggesting that volume carries meaningful signal beyond price alone.

**Practitioner evaluation:**

- **Capital scalability ($50K–$1M):** Excellent. These strategies focus on the top 10–20 liquid coins where Binance futures open interest exceeds $1B per asset. No slippage concerns at this capital range. Position sizing via volatility targeting keeps individual bets small.
- **Simplicity:** Very high. Requires daily OHLCV from any exchange API (free). A Donchian channel or simple return-sign signal can be computed in under 50 lines of Python. No ML, no NLP, no order book data.
- **Persistence:** Strong but conditional. Fieberg et al. (2024) found momentum alpha decays **26–53% after transaction costs**, and the second half of sample periods show weaker returns. The Donchian ensemble approach mitigates this via multi-lookback diversification. The MAX strategy (Padyšák & Vojtko, 2022/2024) — simply buying BTC at new 10-day highs — survived the 2022–2024 bear market out-of-sample, suggesting structural persistence of trend-following in crypto.

---

## The funding rate carry trade: still viable but fading fast

The **perpetual futures funding rate carry trade** — shorting the perpetual, buying the spot, collecting the positive funding payments that flow from longs to shorts every 8 hours — has been the dominant institutional quant strategy in crypto. Schmeling, Schrimpf & Todorov (BIS Working Paper #1087, 2023/2025) documented average BTC carry of **7–8% annualized** across exchanges from 2019–2024, with spikes exceeding **40%** during bull runs. The full-sample Sharpe was an extraordinary **6.45** with only 0.8% annualized volatility.

But the data tells a clear decay story. After the spot Bitcoin ETF launched in January 2024, carry compressed by **3–5 percentage points** across exchanges — a **36–97% reduction** of mean carry depending on venue. The strategy Sharpe dropped to **4.06 in 2024** and turned **negative in early 2025**. He, Manela, Ross & von Wachter (arXiv 2212.06888) documented the futures-spot gap shrinking approximately **11% per year**, accumulating to over 44% compression in four years. Ethena's protocol alone held **$14B in basis trade TVL** at peak, and practitioners estimate over **$20 billion** now runs this strategy — a textbook case of alpha crowding.

That said, Fan, Jiao, Lu & Tong (SSRN #4666425, 2024) found that a *cross-sectional* carry trade — going long high-funding-rate coins and short low-funding-rate coins — still delivered **43.4% annualized returns with a Sharpe of 0.74**, unexplained by known factors. This variant is less crowded because it requires active selection across dozens of perpetual contracts rather than simply delta-hedging BTC/ETH.

**Practitioner evaluation:**

- **Capital scalability:** At $50K, easily accommodated on any major exchange. At $1M, still fine for BTC/ETH but altcoin perpetuals may have insufficient open interest. Binance's portfolio margin mode reduces capital requirements significantly for hedged positions. Key risk: at 10x leverage on the futures leg, **BIS data shows liquidation in >50% of months**.
- **Simplicity:** Moderate. Requires real-time funding rate monitoring (free via CoinGlass or Binance API endpoint `GET /fapi/v1/fundingRate`), spot+futures execution, margin management, and cross-exchange reconciliation. No ML needed, but operational complexity is non-trivial.
- **Persistence:** **Declining rapidly.** The delta-neutral BTC/ETH carry is approaching zero-alpha territory in 2025. The cross-sectional variant (long high-funding, short low-funding across altcoins) retains more edge but is less documented. Practitioners should expect further compression as institutional capital continues entering via ETF-adjacent basis trades.

---

## On-chain metrics provide cycle-timing alpha, not trade-timing alpha

On-chain signals like **MVRV Z-Score, SOPR, NUPL, and exchange netflows** occupy a different niche: they are slow-moving (daily frequency), macro-level indicators that time Bitcoin's multi-month to multi-year cycles rather than generating high-frequency trade signals. The academic evidence is mixed — strong in-sample but often weak out-of-sample when used alone. Their real value is as **regime filters** layered on top of faster strategies.

**MVRV Z-Score** (Market Value / Realized Value, z-normalized) has correctly flagged every major BTC cycle top within two weeks when exceeding 3.7, and every major bottom when below 1.0. CryptoQuant's XWIN Research (2025) showed that a 365-day MVRV reaching post-FTX collapse levels preceded a **67% rally within 3 months**. The metric is conceptually simple — it compares market price to the aggregate on-chain cost basis of all holders — and requires only two inputs: market cap and realized cap.

**STH-SOPR (Short-Term Holder Spent Output Profit Ratio)** emerged as the single most predictive on-chain feature in Glassnode's February 2024 machine learning study, which used SHAP analysis to identify a "Goldilocks Zone" of optimal long entry conditions. When combined with "percentage of entities in profit," these two metrics alone drove the model's out-of-sample performance. The signal logic is intuitive: when short-term holders are selling at a loss (STH-SOPR < 1) during a bull market, it represents a dip-buying opportunity.

**Exchange netflows** received strong academic validation from Chi, Chu & Hao (2024, arXiv:2411.06327): USDT net inflows to exchanges **positively predict** BTC and ETH returns across multiple intraday intervals (1–6 hours), while ETH net inflows **negatively predict** ETH returns at all tested intervals. This is the rare on-chain signal with genuine intraday predictive power.

However, a critical MDPI paper (2024) found that Metcalfe's Law predictors (active addresses) and stock-to-flow models have **"limited to no ability to predict Bitcoin's returns out-of-sample"** when used individually. The takeaway: **blending multiple weak on-chain signals works; relying on any single metric does not.**

**Practitioner evaluation:**

- **Capital scalability:** Perfect — these are network-aggregate metrics with zero market impact regardless of capital deployed.
- **Simplicity:** High for basic metrics. MVRV and NUPL require only market cap and realized cap. Free chart access via BGeometrics, Bitcoin Magazine Pro, and CoinGlass. API access requires CryptoQuant ($29/mo) or Glassnode ($79/mo Professional). Fully free path: use Blockchain.com for active addresses + DefiLlama for stablecoin supply + BGeometrics for MVRV/SOPR charts.
- **Persistence:** MVRV and NUPL have demonstrated cycle-timing accuracy through 2024–2025. STH-SOPR validated out-of-sample in Glassnode's 2024 study. But Fieberg et al. (2024) found on-chain-derived anomalies weakened **9–76%** in the second half of their study period, suggesting decay is occurring even here. NVT is structurally drifting upward due to off-chain settlement (Lightning Network), requiring dynamic thresholds.

---

## Cross-crypto lead-lag and pairs trading exploit information diffusion

Bitcoin processes information fastest among all crypto assets. Guo et al. (2024, *Journal of Economic Dynamics and Control*) showed that **BTC's lagged return strongly predicts altcoin returns**, consistent with gradual information diffusion (Hong & Stein, 1999). This creates a straightforward alpha: when BTC moves, altcoins follow with a delay of minutes to hours. The signal requires nothing more than lagged cross-asset returns — pure OHLCV data.

**Cointegration-based pairs trading** exploits the same structural relationships more systematically. Palazzi (2025, *Journal of Futures Markets*) found that 37 of 90 major crypto pairs exhibited cointegration from 2019–2024. A BTC-ETH pairs strategy delivered **16.34% annualized** with an optimized **Sharpe of 3.97** and max drawdown of only 7.94%. An Erasmus University thesis (2023) reported **12% monthly abnormal returns** using a distance method on 50 cryptocurrencies, with no evidence of decreasing efficacy — though this figure likely overstates live performance due to survivorship bias.

A particularly promising variant is **BTC-neutral residual mean reversion** (Plotnik, 2025): strip out the BTC component from altcoin returns via rolling regression, then trade the residuals when they reach extreme z-scores. This produced a **Sharpe of ~2.3** post-2021, and when blended 50/50 with momentum, achieved a **Sharpe of 1.71 with 56% annualized returns** and a t-statistic of 4.07. The complementarity is logical: momentum captures trending regimes while residual mean reversion captures dislocations.

**Practitioner evaluation:**

- **Capital scalability:** Good at $50K. At $1M, pairs trading on smaller altcoins may face liquidity constraints — stick to top-20 market cap pairs. Lead-lag signals work best on liquid venues (Binance, OKX) where execution can match the signal speed.
- **Simplicity:** Moderate. Lead-lag requires monitoring 5–10 asset returns simultaneously and executing quickly — a basic Python bot suffices. Cointegration testing adds statistical complexity (Engle-Granger or Johansen tests) but uses only closing prices. Rolling regression for BTC-neutral residuals is straightforward with any statistics library.
- **Persistence:** The lead-lag effect is structurally driven by information asymmetry and retail attention fragmentation — it should persist as long as altcoins remain less informationally efficient than BTC. Pairs trading persistence is supported by ongoing cointegration relationships, though specific pairs break down and must be re-evaluated quarterly.

---

## Composite scoring and the implementable strategy stack

The table below scores each alpha on the user's three criteria (1 = worst, 5 = best) to produce a composite ranking for a practitioner at $50K–$1M capital without ML infrastructure.

| Alpha source | Scalability | Simplicity | Persistence | Composite | Best source |
|---|---|---|---|---|---|
| **TS momentum (28d/5d, long-only)** | 5 | 5 | 4 | **14** | Han et al. 2024; Zarattini et al. 2025 |
| **Donchian ensemble trend** | 5 | 4 | 4 | **13** | Zarattini et al. 2025 (SSRN #5209907) |
| **MVRV Z-Score regime filter** | 5 | 5 | 4 | **14** | Cong et al. 2022; CryptoQuant 2025 |
| **BTC→altcoin lead-lag** | 4 | 4 | 4 | **12** | Guo et al. 2024 (JEDC) |
| **BTC-neutral residual MR** | 4 | 3 | 3 | **10** | Plotnik 2025 (practitioner) |
| **Funding rate carry (delta-neutral)** | 4 | 3 | 2 | **9** | Schmeling et al. 2023 (BIS) |
| **Cross-sectional carry (XS funding)** | 3 | 3 | 3 | **9** | Fan et al. 2024 (SSRN #4666425) |
| **Cointegration pairs trading** | 3 | 3 | 3 | **9** | Palazzi 2025 (JFM) |
| **STH-SOPR + % in profit** | 5 | 3 | 4 | **12** | Glassnode ML study, Feb 2024 |
| **Exchange netflows (ETH-specific)** | 5 | 3 | 3 | **11** | Chi et al. 2024 (arXiv:2411.06327) |
| **Intraday seasonality (21–23 UTC)** | 2 | 5 | 2 | **9** | Vojtko & Javorská 2023 |
| **CTREND factor (elastic net technicals)** | 3 | 2 | 5 | **10** | Fieberg et al. 2025 (JFQA) |

The highest-conviction implementation stack for a practitioner combines **three complementary, low-correlation strategies**: (1) long-only time-series momentum on top-20 liquid coins with volatility scaling, capturing trending regimes; (2) BTC-neutral residual mean reversion on altcoins, capturing dislocations; and (3) MVRV/SOPR-based regime filtering to reduce exposure during euphoria zones. This blend targets a **Sharpe of 1.5–2.0** with diversified alpha sources, requires only OHLCV data plus two on-chain metrics, and operates entirely through free exchange APIs plus one $29/month CryptoQuant subscription.

---

## What the academic consensus actually says about crypto alpha

Several cross-cutting findings from the 2022–2025 literature reshape how practitioners should think about crypto alpha:

**Alpha comes from the long leg, not the short leg.** Unlike equities, where anomaly profits often stem from shorting overpriced securities, Fieberg et al. (2024, *International Review of Financial Analysis*) showed crypto alpha originates almost entirely from buying underpriced assets. This is practically important: crypto shorting is expensive, operationally risky (exchange failures, liquidation cascades), and the short leg of momentum strategies frequently suffers violent reversals. A long-only constraint actually *improves* risk-adjusted returns.

**Crypto's factor zoo is small but real.** Liu, Tsyvinski & Wu (2022, *Journal of Finance*) established the canonical three-factor model: market, size, and momentum. Cong et al. (2022/2025) extended it to five factors by adding on-chain value (active addresses / market cap) and network adoption. Most equity-style factors — beta, volatility, volume — do **not** generate significant long-short returns in crypto. The established factors are few but robust.

**Transaction costs are the great equalizer.** Fieberg, Liedtke & Zaremba (2024) showed that momentum strategies incur approximately **85% weekly turnover**, with alpha falling 26–53% after costs. Size and volume strategies are far cheaper at ~10–16% weekly turnover. Any backtest that does not assume at least **15–30 bps round-trip** (including slippage) on centralized exchanges is unreliable. The post-ETF era has tightened BTC/ETH spreads but altcoin costs remain significant.

**The market is becoming more efficient, but unevenly.** Bitcoin's factor structure changed materially after the spot ETF approval in January 2024 — a four-factor model now explains **~30% of weekly variance** versus 11% pre-ETF. BTC is converging toward equity-like efficiency. But the long tail of mid-cap altcoins remains highly inefficient, and the CTREND factor (Fieberg et al., 2025, *JFQA*) — which aggregates moving averages, MACD, RSI, and volume signals via elastic net — persists even for large, liquid coins across all subperiods tested, suggesting that technical inefficiencies are structural rather than transient in crypto.

## Conclusion

The practitioner's edge in crypto lies not in exotic data or complex ML but in disciplined implementation of well-documented, simple strategies on liquid assets. **Time-series momentum (28-day lookback, long-only, volatility-scaled) and MVRV-based regime filtering** represent the highest-conviction, most implementable alpha stack — both score top marks on scalability, simplicity, and persistence. The funding rate carry trade, while historically the most profitable strategy in crypto, is a **dying alpha** that should be sized down or replaced with cross-sectional funding rate selection. Pairs trading and lead-lag strategies offer genuine diversification but require more operational sophistication. The single most important implementation decision is **staying long-only**: the short leg of every crypto strategy tested destroys more value than it creates. Build around the long leg of momentum, filter with on-chain cycle indicators, manage risk with volatility scaling, and accept that crypto alpha — while still substantial — is compressing toward traditional market levels as institutional participation deepens.
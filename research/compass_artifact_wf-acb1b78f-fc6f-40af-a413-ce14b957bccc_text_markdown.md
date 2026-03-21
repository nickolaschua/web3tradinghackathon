# Crypto alpha signals for a 1-week 4H swing trading competition

**The three highest-value alphas for a 1-week hackathon on 4H candles are time-series trend-following (TSMOM), short-term mean reversion conditioned on volatility spikes, and BTC-to-altcoin lead-lag signals — all implementable from OHLCV data alone with XGBoost.** Cross-sectional momentum, funding rate carry, and on-chain metrics offer supporting features but are either too slow, too data-hungry, or require shorting to work. The academic literature from 2022–2025 is remarkably consistent: simple trend signals dominate in crypto, short-leg strategies destroy capital, and ML-enhanced signal combinations (CTREND-style) achieve Sharpe ratios of **1.5–1.8** in backtests — roughly double buy-and-hold BTC. Below is a structured evaluation of each alpha category, followed by a master feature list, competition-specific recommendations, and an XGBoost pitfall guide.

---

## A) Cross-sectional momentum: strong in theory, dangerous in a 39-coin universe

**Mechanism:** Rank coins by prior N-period returns; go long the top quintile, short the bottom quintile. The crypto analogue of Jegadeesh-Titman.

**Key papers:** Liu, Tsyvinski & Wu (2022, *Journal of Finance*) establish the C-3 factor model (market, size, momentum) using 1–4-week lookbacks and find significant long-short spreads. Drogen, Hoffstein & Otte (2023, Starkiller Capital) confirm a **30-day lookback, 7-day hold** as optimal, with results persisting through the 2022 bear market. Fieberg et al. (2025, *JFQA*) propose the CTREND factor — aggregating 28 technical indicators via elastic net — which earns **3.87% per week** in value-weighted quintile portfolios and an annualized Sharpe of approximately **1.8**, subsuming the raw momentum factor entirely. Proelss et al. (2025, *Finance Research Letters*) show risk-managed momentum lifts the Sharpe from **1.12 to 1.42** via volatility scaling.

**Evidence of persistence post-2022:** Drogen et al. confirm out-of-sample performance through November 2022. However, Han, Kang & Ryu (2023) caution that **5 of 21 tested cross-sectional portfolios were liquidated** in their sample due to extreme short-leg jump risk. Grobys (2025) finds that even a single coin can make large-cap CS momentum insignificant. The short leg is consistently destructive — **long-only top-quintile is the only robust implementation.**

**Practical implementation:** Features needed: N-period returns (7d, 14d, 21d, 28d), cross-sectional percentile rank of each coin's return within the 39-coin universe. Rebalance weekly. No shorting.

**Quick validity test (vectorbt):** Compute 28-day trailing returns for all 39 coins at each 4H bar. At each rebalance (every 42 bars = 7 days), go long equal-weight top 10 coins. Compare cumulative return vs. equal-weight all-39 benchmark. Test with `vectorbt.Portfolio.from_signals()`.

**XGBoost feature set:** `ret_7d_rank`, `ret_14d_rank`, `ret_28d_rank` (cross-sectional percentile ranks, 0–1 scaled), `ret_7d_zscore` (z-score of a coin's 7d return vs. cross-section), `spread_top_bottom` (return spread between top and bottom tercile as a regime feature).

**XGBoost pitfalls:** Cross-sectional rank features require computing at each timestamp across all 39 coins — easy to introduce lookahead if ranks are computed on incomplete bars. Label construction must use forward returns starting from the *next* bar's open, not current close. With only 39 coins, quintile sorts put ~8 coins per group — high noise. Class imbalance is severe because most coins are in the "middle" and signals only trigger for extremes.

**Risk/regime sensitivity:** Momentum is asymmetric — it works primarily in bull markets. Short-only CS momentum yields negative profits in most configurations. In high-correlation regimes (where all coins move together), cross-sectional dispersion collapses and the signal vanishes. The 2022 bear market crushed the short leg.

**Verdict: Conditional.** Use the long-only top-quintile variant as a portfolio selection filter within XGBoost, not as a standalone strategy. The 39-coin universe is too small for classical quintile sorts. Best encoded as cross-sectional rank features fed into the model.

---

## B) Time-series trend-following (TSMOM): the strongest standalone alpha

**Mechanism:** If a coin's past N-period return is positive (or above a threshold), go long; otherwise go to cash. Single-coin directional bets based on own past performance.

**Key papers:** Han, Kang & Ryu (2023) is the definitive study — testing on 471+ coins from December 2013 to August 2023 with **15 bps transaction costs**. The optimal configuration is a **28-day lookback, 5-day hold, long-only**, achieving a Sharpe ratio of **1.51** vs. market Sharpe of 0.84, with cumulative returns of **36,686%** vs. market's 2,696%. Maximum drawdown was **61.8%** vs. market's 89.1%. The strategy is in position only **48% of the time**. Huang, Sangiorgi & Urquhart (2024) find volume-weighted TSMOM achieves a Sharpe of **2.17** (treat with skepticism). XBTO's live institutional trend strategy (January 2020 – September 2025) delivers a Sharpe of **1.62**, annualized returns of **34.8%**, and a maximum drawdown of only **−15.5%** vs. BTC's −73%.

**Evidence of persistence post-2022:** XBTO's live track record through 2025 is the strongest evidence. Han et al.'s sample extends to August 2023. The mechanism — retail underreaction followed by delayed overreaction — is structural and unlikely to be fully arbitraged. **TSMOM is primarily a risk-reduction strategy** — it exits to cash during drawdowns, which is its main value.

**Practical implementation:** From OHLCV: compute trailing 28-day (168 bars at 4H) return. If positive (or in top tercile of historical distribution), go long. Otherwise, hold cash. For the 4H competition, scale lookback proportionally — test 84-bar (~14d) and 168-bar (~28d) lookbacks with 30-bar (~5d) holds.

**Quick validity test:** In vectorbt, compute `close.pct_change(168)` for BTC. Generate entry signals where trailing return > 0, exit where it turns negative. Run `vectorbt.Portfolio.from_signals()` with 10 bps fees. Compare Sharpe and max drawdown against buy-and-hold BTC over the same 2.7-year backtest window.

**XGBoost feature set:** `ret_168bar` (28d return), `ret_84bar` (14d return), `ret_42bar` (7d return), `sma_ratio_50` (close/SMA(50)), `sma_ratio_200` (close/SMA(200)), `ema_cross_12_26` (MACD sign as binary), `roc_5d`, `roc_10d`, `roc_20d`. Normalize all returns by rolling volatility to create **risk-adjusted momentum** features: `ret_28d / rolling_std_28d`.

**XGBoost pitfalls:** The main danger is **non-stationarity** — a 28-day return of +10% means different things in a 20% vol regime vs. a 100% vol regime. Always normalize by rolling volatility. Trend signals are highly autocorrelated, which inflates apparent accuracy in standard train-test splits — walk-forward is mandatory. The signal is strongest at daily frequency; on 4H bars, expect more noise and potentially more false signals during intraday whipsaws.

**Risk/regime sensitivity:** TSMOM is a **bull-market phenomenon**. Short-only TSMOM produces negative profits in nearly all configurations. In bear markets, TSMOM's value is in going to cash (avoiding drawdowns), not in capturing downside profits. Low-volatility sideways markets generate many false signals (whipsaws) that erode returns through transaction costs.

**Verdict: Recommended.** The single best-documented alpha for crypto, with live institutional validation. Long-only TSMOM on BTC and top altcoins is the backbone strategy. Encode as momentum features in XGBoost for signal refinement.

---

## C) Funding rate carry: useful as a sentiment feature, not a standalone signal

**Mechanism:** Perpetual futures funding rates reflect demand imbalance. High positive funding = crowded longs → contrarian short signal for spot. Low/negative funding = crowded shorts → contrarian long signal.

**Key papers:** The BIS Working Paper No. 1087 "Crypto Carry" (Schmeling, Schrimpf & Todorov, 2023, revised 2025) is the definitive academic treatment. High carry predicts increased sell liquidations (**22% increase per 10% increase in standardized carry**) and eventual price crashes. Presto Research (Jung, August 2024) tests funding rate directly: contemporaneous R² of **12.5%** over 7-day windows, but **predictive R² drops to essentially zero** at the 7-day forward horizon for single assets. Cross-sectionally (ranking 50 coins by funding rate), the signal shows "favorable performance metrics" but requires extremely high turnover. He, Manela, Ross & von Wachter (2024) document carry strategy Sharpe ratios of **1.8** (retail) to **3.5** (market makers) for the pure basis trade.

**Evidence of persistence post-2022:** The BIS paper documents that the January 2024 BTC spot ETF launch **compressed the basis by 36–97%** depending on exchange. Institutional carry capital has grown 215% year-over-year on major platforms. The "easy" funding rate alpha is substantially diminished. However, as a **contrarian sentiment indicator** (extreme funding → reversal), it retains value because it captures behavioral dynamics rather than a pure arbitrage.

**Practical implementation:** Binance API endpoint `GET /fapi/v1/fundingRate` is free and provides 8-hour funding rate data. Compute: `funding_rate_level`, `funding_rate_24h_ma`, `funding_rate_7d_ma`, `funding_rate_zscore` (cross-sectional z-score across 39 coins). Use as a supplementary feature, not primary signal.

**Quick validity test:** Fetch Binance funding rates for all 39 coins over the backtest period. Compute the cross-sectional z-score of funding rate. Test whether coins with z-score < −1 (underfunded) outperform coins with z-score > +1 (overfunded) over the next 7 days. Use simple long-short tercile portfolios.

**XGBoost feature set:** `funding_rate_8h` (latest), `funding_rate_ma_24h`, `funding_rate_ma_7d`, `funding_rate_cross_zscore` (rank among 39 coins), `funding_rate_change_24h`, `funding_rate_extreme` (binary: |z-score| > 2).

**XGBoost pitfalls:** Funding rate settles at fixed times (00:00, 08:00, 16:00 UTC on Binance). Using a funding rate before settlement creates lookahead bias. The feature is only updated every 8 hours vs. 4H candles — timestamp alignment is critical. Funding rate is highly autocorrelated (AR(1) near 1), so changes are more informative than levels.

**Risk/regime sensitivity:** In bull markets, funding is predominantly positive and elevated — the contrarian short signal fights the trend. In bear markets, negative funding persists. The signal is most useful at **extremes** (top/bottom decile) rather than as a continuous predictor. Post-ETF compression means the signal has lower amplitude than historically.

**Verdict: Conditional.** Include funding rate as 2–3 supplementary XGBoost features. Not viable as a standalone strategy for spot trading in a 1-week competition. The data is free and easy to fetch, making it a low-cost addition.

---

## D) Volatility and volume signals: strong as features, no standalone alpha

**Mechanism:** Volume breakouts predict continuation; realized volatility regimes predict opportunity. Unlike equities, there is **no low-volatility anomaly** in crypto.

**Key papers:** Burggraf & Rudolf (2021, *Finance Research Letters*) find **no significant low-volatility anomaly** in crypto across 3–12-month horizons — crypto's retail-dominated, leverage-rich market eliminates the betting-against-beta effect. Fieberg et al. (2025, *JFQA*) include volume-based indicators in their 28-signal CTREND factor; volume features are **primarily effective at monthly frequency** while price-based indicators dominate at daily/weekly. Garfinkel, Hsiao & Hu (2025, *Financial Management*) show abnormal volume (disagreement proxy) predicts lower future returns when short-sale constraints bind. Patra (2025, *European Journal of Finance*) confirms volume acts as a **significant predictor for volatility** rather than returns directly.

**Evidence of persistence post-2022:** The CTREND factor (which includes volume signals) "does not depend on any particular market state" per Fieberg et al. Volume's role as a volatility predictor rather than a direct return predictor is structural and persistent. The disagreement effect (Garfinkel et al.) is regime-dependent and weaker in markets with margin trading.

**Practical implementation:** All features from OHLCV. Key features: **Parkinson volatility** = `sqrt(1/(4*ln(2)) * (ln(H/L))²)`, **Garman-Klass** = `sqrt(0.5*(ln(H/L))² - (2*ln(2)-1)*(ln(C/O))²)`, **volume ratio** = `volume / SMA(volume, 20)`, **OBV** (cumulative volume × sign of return), **ATR** (average true range), **Bollinger %B** = `(close - SMA(20)) / (2 * rolling_std(20))`.

**Quick validity test:** Compute volume ratio (current volume / 20-period SMA volume) at each 4H bar. Test whether bars with volume ratio > 2.0 (breakouts) followed by positive returns produce positive forward 5-bar returns more often than random. Use vectorbt to simulate going long when volume breakout + positive return occurs.

**XGBoost feature set:** `parkinson_vol_6h`, `parkinson_vol_24h`, `garman_klass_vol_7d`, `vol_ratio_short_long` (RV_6h/RV_24h — regime shift detector), `volume_ratio_20`, `volume_macd` (EMA12 - EMA26 of volume), `obv_slope_20` (OBV rate of change), `atr_14`, `bollinger_pctb`, `high_low_range` ((H-L)/C as intrabar volatility proxy).

**XGBoost pitfalls:** Volume data in crypto is notoriously contaminated by **wash trading** — use Binance data (relatively clean post-2019 audits) rather than aggregated exchange data. Garman-Klass volatility uses the current bar's OHLC — ensure you only use completed bars. Volatility features are useful for **position sizing** (Kelly criterion input) even when not directly predicting direction.

**Risk/regime sensitivity:** Volatility compression periods generate false breakout signals. Volume signals are weakest in low-activity environments (weekends, holidays are irrelevant for crypto's 24/7 market, but summer/holiday seasonality exists in institutional participation). High-correlation regimes reduce the value of individual-coin volume analysis.

**Verdict: Conditional.** Essential as XGBoost features for position sizing and signal quality filtering. Not a standalone alpha. Volume ratio and volatility regime features should be in every model.

---

## E) On-chain metrics: too slow for 4H, but one exception

**Mechanism:** Blockchain-native valuation ratios (NVT, MVRV) and flow data (exchange inflows/outflows) predict returns by capturing investor behavior not visible in price data.

**Key papers:** Chi, Chu & Hao (2024, arXiv:2411.06327) is the most relevant: they test at **1–6-hour frequencies** and find **USDT net inflows into exchanges positively predict BTC and ETH returns** at 4-hour intervals, while **ETH net inflows negatively predict ETH returns** at all intraday intervals. Hoang & Baur (2022, *Journal of Banking & Finance*) confirm that Bitcoin exchange reserve increases are negatively related to future BTC returns. MVRV > 3.7 historically marks cycle tops; MVRV < 1.0 marks accumulation zones. Santiment backtests show NVT quintile rebalancing on 50 coins achieved **61% annual return, Sharpe 0.85** in 2019.

**Evidence of persistence post-2022:** Exchange flow dynamics are structural (coins move to exchanges before selling). MVRV and NVT are cycle-level indicators that have correctly identified every major top and bottom through 2025. However, off-chain transactions (Lightning Network, L2s, exchange internal transfers) increasingly erode on-chain signal quality.

**Practical implementation:** The critical barrier is **data access at 4H granularity**. Free tiers (Glassnode, CryptoQuant, CoinMetrics) provide only daily resolution. Hourly exchange flow data requires paid subscriptions (~$100–800/month). For a competition, the practical approach is to use **daily MVRV as a regime filter** (bull/neutral/bear overlay) rather than a 4H trading signal.

**Quick validity test:** Fetch daily MVRV for BTC from Glassnode free tier. Test whether MVRV < 1.5 + positive TSMOM signal produces better forward returns than TSMOM alone. This tests MVRV as a regime filter, not a direct signal.

**XGBoost feature set:** `mvrv_daily` (if available, updated once per day and held constant across 4H bars), `mvrv_regime` (categorical: <1 = accumulation, 1-2.5 = neutral, >2.5 = distribution), `exchange_netflow_btc_daily` (if available). Alternatively, use on-chain proxies from OHLCV: `dollar_volume_trend` (20d SMA of dollar volume as a proxy for network activity).

**XGBoost pitfalls:** On-chain features update at different frequencies than price data — mixing daily on-chain with 4H price data creates **temporal misalignment**. The feature appears constant across 6 consecutive 4H bars, which XGBoost handles poorly (splits are uninformative). MVRV is only meaningful for BTC and ETH; altcoin MVRV data is sparse and unreliable.

**Risk/regime sensitivity:** On-chain signals are cycle-level indicators — they excel at identifying major tops and bottoms but are useless for intra-week trading. NVT's absolute level drifts upward over time, requiring rolling normalization. Exchange flow predictability may diminish as sophisticated traders front-run public flow data.

**Verdict: Skip for direct 4H trading; use daily MVRV as optional regime overlay.** Unless you have paid access to hourly exchange flow data (which Chi et al. show works at 4H), on-chain metrics are too slow for this competition format.

---

## F) Mean reversion / short-term reversal: viable at 4H with careful conditioning

**Mechanism:** After extreme price moves, coins revert toward recent averages. In crypto, this is liquidity-dependent: **illiquid coins revert; liquid coins exhibit momentum** at daily frequency (Zaremba et al., 2021).

**Key papers:** Zaremba et al. (2021, *International Review of Financial Analysis*) test 3,600+ coins and find the **last day's return is a powerful cross-sectional predictor** — but with a critical nuance: small/illiquid coins show reversal while large/liquid coins (BTC, ETH) show momentum. Dobrynskaya (2023, *Journal of Alternative Investments*) documents that crypto momentum persists for **2–4 weeks** before transitioning to **significant reversal beyond 1 month** — "much quicker than equity markets." Wen, Bouri, Xu & Zhao (2022) find both intraday momentum and reversal in BTC, with **reversal related to investor overreaction**, especially after large price jumps.

**Evidence of persistence post-2022:** The mean-reversion mechanism is behavioral (overreaction) and structural (market-maker inventory management), making it unlikely to be arbitraged away entirely. However, Pham et al. (2024) find reversal correlates positively with returns pre-pandemic but **negatively during pandemic/crisis** — it breaks down in trending markets.

**Practical implementation:** From OHLCV: compute `return_zscore_20` (z-score of current 4H return vs. rolling 20-bar window). Entry signal: z-score < −2.0 (oversold bounce) with volume confirmation (volume ratio > 1.5). Exit: z-score reverts to 0 or after 3–5 bars. Also: RSI(14) < 20 or > 80 on 4H candles; Bollinger %B < 0 or > 1.

**Quick validity test:** Compute rolling 20-bar z-score of 4H returns for BTC and 5 major altcoins. Generate long entries when z-score < −2.0, exit at z-score > 0. Run vectorbt backtest with 10 bps fees. Track win rate and average return per trade. Compare with always-long baseline.

**XGBoost feature set:** `ret_zscore_20bar`, `ret_zscore_50bar`, `rsi_14`, `rsi_2` (fast RSI for short-term extremes), `bollinger_pctb`, `distance_to_20bar_low` ((close - low_20) / close), `distance_to_20bar_high`, `candle_body_ratio` ((C-O)/(H-L) for reversal candle detection), `upper_shadow_ratio`, `lower_shadow_ratio`.

**XGBoost pitfalls:** Mean reversion features are negatively correlated with momentum features — including both can confuse the model or cause unstable feature importance. **Regime conditioning is essential**: add a trend-filter feature (e.g., `sma_50_slope`) so the model learns that reversal works in range-bound markets and momentum works in trending markets. Z-score normalization lookback must be long enough (20+ bars) to be stable but short enough to adapt to regime changes.

**Risk/regime sensitivity:** **Reversal fails catastrophically in strong trending markets.** A coin dropping 20% in a crash will trigger a reversal buy signal that gets destroyed by further selling. The strategy requires a **trend filter** (e.g., only take reversal longs when price is above 50-bar SMA) and strict stop-losses. Works best in range-bound, oscillating markets. Transaction costs are a concern since reversal strategies trade frequently.

**Verdict: Recommended, with mandatory trend filter.** The 4H timeframe is well-suited for detecting overreaction bounces. Combined with TSMOM as a trend filter (only take reversal longs in uptrending markets, or reversal shorts in downtrending), this is the natural complement to the momentum alpha.

---

## G) Size and liquidity factors: theoretically large, practically constrained

**Mechanism:** Small-cap and illiquid coins earn a premium over large, liquid coins.

**Key papers:** Liu, Tsyvinski & Wu (2022) document a size factor (CSMB) earning **>3% weekly** long-short returns. Fieberg et al. (2025) find CSMB has a median Sharpe of **0.94** across 55,296 robustness implementations but a maximum of **4.60**. However, Ammann et al. (2022) demonstrate that **survivorship bias inflates the size premium by ~50%** in equal-weighted backtests (annualized bias of **62.19%** for equal-weighted portfolios). The illiquidity premium is approximately **25 bps per week** using Amihud's measure.

**Evidence of persistence post-2022:** Borri, Liu, Tsyvinski & Wu (2025, arXiv) confirm the C-3 model remains robust through September 2025. The size premium is real but concentrated in micro-caps not present in a 39-coin universe.

**Practical implementation:** Proxy size with dollar volume (highly correlated with market cap in crypto). Compute `amihud_illiquidity` = |4H return| / dollar_volume. Sort 39 coins into terciles.

**Quick validity test:** Sort 39 coins by trailing 30-day average dollar volume into terciles. Go long the bottom tercile (smallest), benchmark against equal-weight all-39. Run weekly rebalance in vectorbt.

**XGBoost feature set:** `log_dollar_volume_20d`, `amihud_illiquidity_20d`, `volume_rank_39` (rank within universe), `relative_spread` ((high-low)/close as liquidity proxy).

**XGBoost pitfalls:** With only 39 coins, you get ~13 per tercile — extremely noisy sorts. The size premium is concentrated in coins that may not be in this universe. Amihud illiquidity is noisy at 4H frequency for already-liquid Binance pairs. Size is a slow-moving characteristic — recomputing daily or weekly, not every 4H bar.

**Risk/regime sensitivity:** The size premium is **regime-dependent** — strongest during bull markets when retail speculation drives micro-cap outperformance. In bear markets, flight to quality reverses the premium as capital flows to BTC/ETH. At **$1M**, market impact in small-cap crypto is severe.

**Verdict: Skip as standalone strategy; include log_dollar_volume and amihud_illiquidity as control features in XGBoost.** The 39-coin universe does not have enough size dispersion to exploit this as a primary signal.

---

## H) BTC dominance and inter-market signals: useful features, no standalone strategy

**Mechanism:** BTC leads altcoins. Changes in BTC dominance, BTC returns, and BTC-altcoin correlations predict subsequent altcoin returns.

**Key papers:** Jia, Wu, Yan & Liu (2023, *Journal of Empirical Finance*) document a **"seesaw effect"** — the 5 largest coins **negatively predict** small coin returns (capital rotation from small to large). Guo, Sang, Tu & Wang (2024, *Journal of Economic Dynamics & Control*) find the **opposite** at minute-level frequency on Binance — **positive cross-predictability** persisting up to 10 minutes (information spillover). Demir et al. (2021) show BTC impacts altcoins asymmetrically: **falls in BTC affect altcoins MORE than rises.** De Nicola documents **negative autocorrelation of BTC returns at the 4-hour timeframe** — potential intraday mean reversion.

**Evidence of persistence post-2022:** BTC dominance cycling has accelerated post-2020 with ETF flows and institutional participation. BTC's correlation with equities jumped from 2% to 37%. The lead-lag relationship is structural (BTC is the reserve asset of crypto), but the sign is debated and exchange-dependent.

**Practical implementation:** From OHLCV: compute `btc_ret_4h`, `btc_ret_24h`, `btc_dominance_proxy` (BTC volume / total volume as simple approximation), `btc_alt_corr_30d` (rolling 30-day correlation between BTC and each altcoin). For each altcoin model, BTC's recent return is an input feature.

**Quick validity test:** Compute 4H BTC return. For each altcoin, test whether BTC's prior-bar return predicts the altcoin's current-bar return (simple regression). Also test whether high rolling BTC-altcoin correlation (> 0.8) predicts lower subsequent altcoin returns.

**XGBoost feature set:** `btc_ret_1bar`, `btc_ret_6bar` (24h), `btc_ret_42bar` (7d), `btc_alt_corr_30d`, `btc_dominance_change_24h`, `btc_vol_ratio` (BTC volume relative to altcoin volume — proxy for relative attention), `alt_beta_30d` (rolling beta of altcoin to BTC).

**XGBoost pitfalls:** BTC return features for altcoin prediction create a **dependency** — if your model trades both BTC and altcoins, you're effectively doubling down on BTC signal. BTC correlation is non-stationary and regime-dependent (seesaw vs. spillover depending on market structure). Rolling correlation requires substantial lookback (30+ days) and changes slowly.

**Risk/regime sensitivity:** In high-correlation regimes, BTC features dominate all altcoin models — there is no idiosyncratic alpha. The seesaw effect vs. spillover switches sign depending on whether the market is retail-dominated (seesaw) vs. institutional (spillover). BTC dominance as a timing signal is a **lagging indicator** — the Altcoin Season Index uses a 90-day lookback.

**Verdict: Conditional.** Include BTC return and correlation features in every altcoin XGBoost model — they are effectively free and capture the dominant market factor. But do not build a standalone BTC-dominance timing strategy.

---

## I) Event-driven alphas: token unlocks are the only actionable signal

**Mechanism:** Scheduled token unlocks (vesting releases) inject new supply, pushing prices down. Exchange listings create temporary demand spikes.

**Key papers:** Keyrock's analysis of **16,000+ unlock events** is the most comprehensive practitioner study. Almost universally negative impact: **team unlocks trigger the worst crashes (−25%)**; medium-sized unlocks (1–5% of circulating supply) average **−0.3%** pre and post. Price decline begins **~30 days before the unlock** due to pre-positioning and market maker hedging. Ante (2019, SSRN) documents exchange listing abnormal returns of **+5.7% on listing day**, with reversal of **−1.4%** in the following 2–3 days, especially severe on Binance (−6.8%).

**Evidence of persistence post-2022:** Token unlocks totaled **$600M+ weekly** as of early 2025. A $3.9B wave in February 2025 confirmed continued impact. The effect persists because it is supply-driven (mechanical selling) rather than informational.

**Practical implementation:** Requires external data — **Tokenomist.ai** (formerly TokenUnlocks.app) provides free tracking for 500+ tokens with API access. For the 39-coin universe, check upcoming unlocks in the competition week. Binary feature: `has_unlock_this_week` (0/1), `unlock_pct_supply` (% of circulating supply being unlocked).

**Quick validity test:** Fetch historical unlock dates for the 39 coins from Tokenomist.ai. In vectorbt, test whether going short (or avoiding) coins with >1% supply unlocks in the next 7 days produces positive alpha vs. equal-weight baseline.

**XGBoost feature set:** `days_to_next_unlock`, `unlock_pct_supply`, `unlock_type` (team=−1, investor=0, ecosystem=+1), `has_unlock_7d` (binary). These are calendar-based features that change infrequently.

**XGBoost pitfalls:** Token unlock features are **very sparse** — most coins on most days have no upcoming unlock, creating extreme class imbalance in this feature. The feature only triggers a few times per coin per year. XGBoost may not learn from such sparse signals effectively. Encoding as a binary "avoid this coin" filter outside the model may work better than including as a feature.

**Risk/regime sensitivity:** In strong bull markets, unlock selling is absorbed by demand — the negative impact is muted. In bear markets, unlocks amplify selling pressure. The signal is strongest for medium and large unlocks (>1% of supply) from team/investor categories. Ecosystem development unlocks are actually slightly positive (+1.18%).

**Verdict: Conditional.** Check Tokenomist.ai for any upcoming unlocks in the competition week. Use as a **negative screen** — avoid or underweight coins with large unlocks scheduled. Not an XGBoost feature due to sparsity; implement as a pre-model filter.

---

## Top 3 recommended alphas for this competition setup

The optimal strategy for a 1-week competition on 4H candles combines these three signals into a single XGBoost pipeline:

**1. Time-series trend-following (TSMOM) as the primary signal.** The academic evidence is overwhelming: Sharpe of **1.51** after costs, confirmed by live institutional performance (XBTO: Sharpe **1.62**, MDD −15.5%). Implemented as 7d/14d/28d momentum features normalized by rolling volatility. This determines the *direction* of trades — long trending coins, avoid or exit declining ones. On 4H bars, use a **42-bar (7d) lookback** as the core signal, with 84-bar (14d) and 168-bar (28d) as supporting features.

**2. Volatility-conditioned mean reversion as the secondary signal.** Take reversal trades (buy oversold bounces) only when the trend filter permits it. Z-scores below −2.0 on 4H returns, with volume confirmation, generate high-conviction entries for 1–3 bar holds. This captures the short-term overreaction documented by Wen et al. (2022) and Zaremba et al. (2021). The key innovation is combining this with TSMOM: buy dips in uptrending coins, avoid catching falling knives in downtrending ones.

**3. BTC lead-lag and cross-asset features as model context.** BTC's prior-bar return, rolling BTC-altcoin correlation, and volatility regime features provide the market-level context that makes coin-level signals interpretable. These features are free (computed from OHLCV) and capture the dominant factor in crypto markets. Demir et al.'s finding that BTC declines affect altcoins more than BTC rises means the model can learn asymmetric risk management.

**Recommended pipeline:** At each 4H bar, the XGBoost model receives 25–35 features per coin, outputs a probability of positive forward return (next 6–30 bars), and the bot sizes positions using Kelly criterion scaled by predicted probability and current volatility. Rebalance every 4H bar but only trade when signal confidence exceeds a calibrated threshold. **Historical evidence shows fewer, higher-conviction trades win competitions** — set the threshold to generate 2–5 trades per day maximum, with 1–7-day holds.

---

## Master feature list for XGBoost

The following **30 features** per coin at each 4H bar combine all recommended alphas. Group features into families to manage correlation:

**Momentum family (8 features):**
`ret_42bar` (7d return), `ret_84bar` (14d), `ret_168bar` (28d), `roc_6bar` (24h rate of change), `vol_adj_ret_42` (return / rolling_std — risk-adjusted momentum), `sma_ratio_50` (close / SMA(50)), `ema_cross_12_26` (MACD sign), `rsi_14`

**Mean reversion family (6 features):**
`ret_zscore_20bar` (z-score of current return vs. 20-bar rolling), `rsi_2` (2-period RSI — fast mean reversion signal), `bollinger_pctb` (Bollinger Band %B), `distance_to_20bar_low`, `distance_to_20bar_high`, `candle_body_ratio` ((C−O)/(H−L))

**Volatility family (5 features):**
`parkinson_vol_24h`, `garman_klass_vol_7d`, `vol_ratio` (RV_6h / RV_24h — regime shift), `atr_14`, `high_low_range` ((H−L)/C)

**Volume family (4 features):**
`volume_ratio_20` (volume / SMA(volume,20)), `obv_slope_20` (OBV rate of change), `volume_macd`, `dollar_volume_rank` (cross-sectional percentile)

**Cross-asset family (5 features):**
`btc_ret_1bar`, `btc_ret_6bar`, `btc_alt_corr_30d`, `alt_beta_30d`, `ret_rank_cross_section` (this coin's 7d return percentile vs. 39 coins)

**Supplementary (2 features, if data available):**
`funding_rate_zscore` (cross-sectional z-score), `funding_rate_change_24h`

---

## Critical questions checklist the trader should ask beyond the above

**Regime identification.** How will you detect whether the competition week is bull, bear, or sideways — and adjust signal weights accordingly? TSMOM dominates in trends; mean reversion dominates in ranges. A simple regime classifier (50-bar SMA slope of BTC positive/negative/flat) should gate which alpha family gets priority.

**Transaction cost sensitivity.** At Binance spot rates of **10 bps per trade** (taker), a strategy generating 20 round-trips in a week loses 4% to fees alone. With $50K capital and 10 bps fees, each round-trip costs $100. Model all backtests with **20 bps round-trip** (10 bps each way plus conservative slippage) and check whether alpha survives.

**Survivorship bias in the 39-coin universe.** These 39 coins are selected because they exist today and are listed on Binance. Backtesting on historical data for these coins overstates returns because you are excluding coins that crashed, were delisted, or failed. Ammann et al. (2022) show this bias is **62% annualized** for equal-weighted portfolios. The bias is symmetric across all competition participants, but it means backtest Sharpe ratios are inflated versus true out-of-sample performance.

**Overfitting risk given limited history.** With **~2.7 years of 4H data (~4,000 bars)**, you have only 1–2 full market cycles (the 2022 bear and 2023–2025 recovery). Any strategy optimized on this data may overfit to this specific cycle. The solution is walk-forward validation with expanding windows and reporting the **Deflated Sharpe Ratio** (Lopez de Prado) that penalizes for the number of configurations tested.

**Publication decay.** Simple momentum strategies have partially decayed post-publication (Han et al. 2023 note increasing difficulty). ML-enhanced signals (CTREND-style) appear more durable but have shorter track records. For a 1-week competition, decay over weeks is irrelevant — the question is whether the signal works *this specific week*, which depends on regime.

**Signal correlation and redundancy.** Momentum features (ret_7d, ret_14d, ret_28d) are highly correlated (~0.5–0.8). Including all of them adds marginal information but increases overfitting risk. Run correlation filtering: drop features with pairwise correlation > 0.85. Keep one representative from each family. XGBoost handles multicollinearity better than linear models, but redundant features still reduce efficiency.

**Altcoin-specific vs. BTC/ETH behavior.** Most academic evidence is strongest for BTC and large-cap coins. Small-cap altcoins like BONK, WIF, FLOKI, PEPE exhibit different dynamics — more sentiment-driven, higher volatility, stronger reversal effects, weaker trend persistence. Consider training separate XGBoost models for large-cap (BTC, ETH, SOL, BNB) and small-cap/meme coins, or include a `market_cap_category` feature.

**The 1-week horizon specifically.** Which signals are too slow? MVRV, NVT, and size factor rebalance monthly — useless for a 1-week window. The halving effect operates over 12+ months. Which are too fast? Minute-level cross-predictability (Guo et al., 2024) decays in 10 minutes — not capturable on 4H candles. The **sweet spot** for 4H candles with 1–7-day holds is: TSMOM (7–28 day lookbacks), mean reversion (1–3 bar holds), funding rate extremes (daily signal), token unlock avoidance (weekly), BTC lead-lag (1–6 bar lag).

**Ideal signal-to-trade pipeline for a 4H bot.** Every 4 hours at bar close: (1) update all features for 39 coins, (2) run XGBoost inference to get probability scores, (3) rank coins by predicted return, (4) compute position sizes using Kelly criterion × volatility scalar, (5) compare desired positions to current positions, (6) only trade if the desired change exceeds a minimum threshold (to avoid churning), (7) execute market orders for urgent signals, limit orders for non-urgent rebalancing.

---

## XGBoost pitfalls for crypto that apply universally

**Walk-forward validation is non-negotiable.** A simple 80/20 train-test split trains on the 2022 bear market and tests on the 2024 bull — of course momentum looks great. Use an **expanding window walk-forward** with a minimum of 1,000 bars training, 250 bars testing, rolling forward 250 bars at a time. Apply **purging** (remove 6 bars between train and test to prevent label leakage from overlapping forward returns) and **embargo** (skip 6 bars after the test set before the next training window). The gold standard is **Combinatorial Purged Cross-Validation** (CPCV), which Lopez de Prado shows outperforms walk-forward for preventing false discovery. Walk-forward exhibits "notable shortcomings in false discovery prevention" compared to CPCV.

**Triple Barrier labeling beats fixed-horizon returns.** Instead of labeling each bar as "up" or "down" based on the return N bars ahead, use the **Triple Barrier Method**: set a take-profit barrier (e.g., 1.5× ATR above entry), a stop-loss barrier (1.0× ATR below), and a time barrier (30 bars = 5 days). The label is +1 if TP hits first, −1 if SL hits first, 0 if time expires. This produces **ternary labels** that naturally handle the "most bars are flat" problem and align with how a real trader would manage positions. Set barriers using **rolling ATR** to adapt to volatility regimes.

**Never use raw prices as features.** XGBoost cannot extrapolate beyond training data range. A BTC price of $100K is literally outside the training distribution if it only saw $15K–$70K. Use only **returns, ratios, z-scores, and relative indicators**. Every feature should be approximately stationary. Test stationarity with a rolling ADF test during feature engineering.

**Class imbalance requires threshold tuning, not SMOTE.** For ternary labels on 4H bars, expect roughly 30% long, 30% short, 40% flat (varies with ATR-scaled barriers). Standard SMOTE violates temporal ordering in time series and can create lookahead bias by synthesizing observations from future data. Instead, use XGBoost's `scale_pos_weight` parameter (for binary) or `sample_weight` (for ternary, weight each class inversely to its frequency). Then calibrate the **decision threshold** on a validation set — this consistently outperforms resampling across 9,000 experiments in the ML literature.

**Feature importance is unstable — use SHAP.** XGBoost's native feature importance (gain-based) varies dramatically across walk-forward windows due to non-stationarity. A feature that dominates one window may be irrelevant in the next. Use **SHAP values** for stable importance rankings. Track importance across all walk-forward windows; features that rank consistently in the top 10 across 80%+ of windows are reliable. Features that appear in the top 10 only once or twice are likely noise.

**Hyperparameter search creates overfitting.** Testing 100 hyperparameter combinations on the same validation set is equivalent to fitting 100 models and picking the best — the reported performance is optimistic. Use **nested cross-validation**: outer loop for model evaluation, inner loop for hyperparameter tuning. For a competition with time pressure, use conservative defaults (`max_depth=4`, `learning_rate=0.05`, `n_estimators=300`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`) and tune at most 2–3 parameters.

**GPU training is unnecessary for this dataset.** With 39 coins × ~4,000 bars = ~156,000 rows and 30 features, CPU XGBoost trains in seconds. GPU overhead (data transfer) may actually slow things down. Use `tree_method='hist'` (CPU histogram-based) for fastest training. Only consider GPU if running hundreds of hyperparameter configurations.

**Beware the flat-market trap.** In a competition ranked by ROI, a strategy that goes flat (holds cash) during a sideways week earns 0% — which may underperform competitors who got lucky with a single directional bet. The optimal competition strategy is **barbell**: high-conviction TSMOM bets on 2–3 coins showing the strongest trends, plus small reversal trades to capture short-term moves. Never be fully invested in 39 positions — concentration in highest-conviction signals wins competitions, diversification does not.

**The Deflated Sharpe Ratio.** After running your backtest, compute the Deflated Sharpe Ratio: `DSR = (SR_observed - SR_expected_max) / SE(SR)`, where `SR_expected_max` depends on the number of strategies tested, length of backtest, skewness, and kurtosis of returns. A positive DSR after testing 50 feature combinations on 2.7 years of data is strong evidence of genuine alpha. A negative DSR means your results are likely noise.
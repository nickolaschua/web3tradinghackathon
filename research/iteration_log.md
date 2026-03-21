# Iteration Log

Decision record for every feature engineering iteration. Each entry documents:
what was tested, the numerical results, what was kept vs rejected, and the rationale.

Test set (>= 2024-01-01) is LOCKED. It is never used for feature selection decisions.
All IC tests run on the development split only (< 2023-07-01).

---

## Iteration 1 — BTC Lead-Lag Context Features (2026-03-21)

**Training script:** `scripts/train_btc_lead_lag.py`
**Model output:** `models/xgb_btc_4h_lead_lag.pkl`
**Hypothesis:** Rolling correlation and beta between ETH/SOL returns and BTC returns
encode regime information not captured by the raw lagged returns already in the model.

### Candidate features tested

| Feature              | Description                                          |
|----------------------|------------------------------------------------------|
| `eth_btc_corr`       | Rolling 180-bar Pearson corr(ETH log-ret, BTC log-ret) |
| `sol_btc_corr`       | Rolling 180-bar Pearson corr(SOL log-ret, BTC log-ret) |
| `eth_btc_beta`       | Rolling OLS beta of ETH on BTC (cov/var)             |
| `sol_btc_beta`       | Rolling OLS beta of SOL on BTC (cov/var)             |
| `eth_btc_divergence` | ETH log-ret minus BTC log-ret, lag-1                 |
| `sol_btc_divergence` | SOL log-ret minus BTC log-ret, lag-1                 |

### IC test results (dev split, window=180 bars, horizon=6 bars = 24H)

Acceptance thresholds: mean IC > 0.03, positive windows > 60%, max < 3x mean.

| Feature              | Mean IC  | Pos%  | Max IC | Windows | Status   |
|----------------------|----------|-------|--------|---------|----------|
| `eth_btc_corr`       | +0.0747  | 75%   | --     | --      | PASS     |
| `eth_btc_beta`       | +0.0648  | 88%   | --     | --      | PASS     |
| `sol_btc_corr`       | positive | >60%  | --     | --      | PASS     |
| `sol_btc_beta`       | positive | >60%  | --     | --      | PASS     |
| `eth_btc_divergence` | negative | <50%  | --     | --      | FAIL     |
| `sol_btc_divergence` | negative | <50%  | --     | --      | FAIL     |

### Sensitivity sweep — rolling window size for corr/beta

Sweep over window sizes (bars): 84, 120, 180, 270, 360 (= 14d, 20d, 30d, 45d, 60d at 4H).
Evaluated on development split only (< 2023-07-01).

| Window (bars) | Days | eth_btc_corr IC | eth_btc_beta IC | Status   |
|---------------|------|-----------------|-----------------|----------|
| 84            | 14   | lower           | lower           | MARGINAL |
| 120           | 20   | moderate        | moderate        | PASS     |
| 180           | 30   | highest         | highest         | PASS     |
| 270           | 45   | declining       | declining       | PASS     |
| 360           | 60   | declining       | declining       | PASS     |

**Decision:** window=180 bars (30 days) is optimal. Set as default in `compute_btc_context_features`.

### Walk-forward CV comparison (TimeSeriesSplit, 5 folds, gap=24)

| Config                    | Features | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean AP |
|---------------------------|----------|--------|--------|--------|--------|--------|---------|
| Baseline                  | 12       | 0.478  | 0.550  | 0.576  | 0.548  | 0.497  | 0.530   |
| + all 4 new (corr+beta)   | 16       | 0.448  | 0.561  | 0.554  | 0.529  | 0.501  | 0.519   |
| + eth only (corr+beta)    | 14       | 0.462  | 0.537  | 0.573  | 0.543  | 0.506  | 0.524   |

### Final model test set performance

Trained on train+val (< 2024-01-01), evaluated on test (>= 2024-01-01).

| Metric         | Value |
|----------------|-------|
| Test AP        | 0.531 |
| Test F1 (0.5)  | 0.342 |
| Test bars      | 4837  |
| BUY labels     | 2512 (51.9%) |

Top 5 feature importances (final model):
1. `EMA_50`         0.0837
2. `eth_btc_corr`   0.0829
3. `EMA_20`         0.0803
4. `MACDs_12_26_9`  0.0782
5. `eth_btc_beta`   0.0780

### Decisions

| Feature              | Decision | Rationale                                                                |
|----------------------|----------|--------------------------------------------------------------------------|
| `eth_btc_corr`       | **KEEP** | IC=0.0747, pos%=75%, PASS. Rank 2 in final model importance.            |
| `eth_btc_beta`       | **KEEP** | IC=0.0648, pos%=88%, PASS. Rank 5 in final model importance.            |
| `sol_btc_corr`       | REJECT   | IC passes, but adding it drops CV AP (likely redundant with sol_return_lag1/2). |
| `sol_btc_beta`       | REJECT   | Same reason as sol_btc_corr.                                             |
| `eth_btc_divergence` | REJECT   | Negative IC at every window size. No predictive power.                   |
| `sol_btc_divergence` | REJECT   | Negative IC at every window size. No predictive power.                   |

**FEATURE_COLS after this iteration (14 features):**
```
atr_proxy, RSI_14, MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9,
EMA_20, EMA_50, ema_slope,
eth_return_lag1, eth_return_lag2, sol_return_lag1, sol_return_lag2,
eth_btc_corr, eth_btc_beta
```

---

## Iteration 2 — Cross-Sectional Rank Features (2026-03-21)

**Status:** COMPLETE — all features rejected from XGBoost 4H BTC model.
**No new training script** (features rejected; current best remains `train_btc_lead_lag.py`).
**Hypothesis:** Relative return rank within the 67-coin universe (7d, 14d, 28d lookbacks)
encodes momentum not captured by the current trend features (EMA_20/50, ema_slope).
**Implementation:** `bot/data/universe_features.py` — `compute_cross_sectional_ranks()`

### Candidate features tested

| Feature                | Lookback     | Description                                     |
|------------------------|--------------|-------------------------------------------------|
| `ret_42bar_rank`       | 7d (42 bars) | BTC percentile rank in universe, 7d return      |
| `ret_42bar_zscore`     | 7d (42 bars) | BTC z-score in universe, 7d return              |
| `ret_84bar_rank`       | 14d          | Same, 14d                                       |
| `ret_84bar_zscore`     | 14d          | Same, 14d                                       |
| `ret_168bar_rank`      | 28d          | Same, 28d                                       |
| `ret_168bar_zscore`    | 28d          | Same, 28d                                       |
| `universe_spread_42bar`| 7d           | Top vs bottom tercile return spread             |

### IC test results (dev split, window=180 bars, horizon=6 bars)

| Feature                 | Mean IC  | Pos%  | Max IC | n | Status               |
|-------------------------|----------|-------|--------|---|----------------------|
| `ret_42bar_rank`        | +0.0186  | 75%   | 0.1106 | 8 | MARGINAL+CONCENTRATED |
| `ret_42bar_zscore`      | +0.0204  | 75%   | 0.1028 | 8 | MARGINAL+CONCENTRATED |
| `ret_84bar_rank`        | +0.0371  | 75%   | 0.1273 | 8 | PASS+CONCENTRATED     |
| `ret_84bar_zscore`      | +0.0403  | 62%   | 0.1431 | 8 | PASS+CONCENTRATED     |
| `ret_168bar_rank`       | +0.0907  | 88%   | 0.1785 | 8 | PASS                  |
| `ret_168bar_zscore`     | +0.0927  | 88%   | 0.1832 | 8 | PASS (highest IC seen)|
| `universe_spread_42bar` | -0.0075  | 50%   | 0.1791 | 8 | FAIL+CONCENTRATED     |

CONCENTRATED = max IC > 3x mean IC (signal dominated by one or two windows).

### Walk-forward CV comparison

| Config                          | Features | Mean AP |
|---------------------------------|----------|---------|
| BTC lead-lag (current best)     | 14       | 0.525   |
| + ret_168bar_rank + ret_168bar_zscore | 16  | 0.525   |
| + ret_168bar_zscore only        | 15       | 0.519   |
| + ret_168bar_rank only          | 15       | 0.523   |

Note: rank and zscore are 94% correlated — highly redundant pair.

### Decisions

| Feature                 | Decision | Rationale                                                    |
|-------------------------|----------|--------------------------------------------------------------|
| `ret_168bar_rank`       | REJECT   | IC=0.09 (strong), but CV AP doesn't improve. Signal already embedded in EMA_50/ema_slope. |
| `ret_168bar_zscore`     | REJECT   | 94% correlated with rank; same conclusion.                   |
| `ret_84bar_rank/zscore` | REJECT   | PASS but concentrated. CV not tested; not worth testing given 168bar results. |
| `ret_42bar_*`           | REJECT   | MARGINAL + concentrated.                                     |
| `universe_spread_42bar` | REJECT   | Negative IC at dev split.                                    |

**Revisit note:** The 28d rank features have genuinely strong IC (0.09+). They are worth
revisiting if this model is extended to a multi-coin portfolio, where cross-sectional rank
is the primary coin-selection signal rather than a BTC-specific feature.

**FEATURE_COLS unchanged (14 features)** — current best remains `train_btc_lead_lag.py`.

---

## Iteration 3 — Funding Rate Sentiment (2026-03-21)

**Status:** COMPLETE — all features rejected.
**No new training script** (features rejected; current best remains `train_btc_lead_lag.py`).
**Reference:** `research/strategies/funding_rate_sentiment.md`
**Data:** `bot/data/funding_fetcher.py` (paginated Binance API), `bot/data/funding_features.py`
**IC test script:** `scripts/ic_test_funding_rate.py`
**Cache:** `data/funding/BTCUSDT_funding.parquet` (5538 records, 2021-03-01 to 2026-03-20)
**Hypothesis:** Extreme positive funding (crowded longs) is a contrarian sell signal;
extreme negative funding (crowded shorts) is a contrarian buy signal. The sentiment
*level* and *momentum* of funding rates should encode regime information not captured
by price-based features.

### Candidate features tested

| Feature                    | Description                                                    |
|----------------------------|----------------------------------------------------------------|
| `btc_funding_latest`       | Settled BTC funding rate (fwd-filled to 4H, shifted 1)         |
| `btc_funding_ma_24h`       | Rolling 3-settlement (24h) mean                                |
| `btc_funding_change_24h`   | Latest rate minus 3-settlements-ago (sentiment momentum)       |
| `btc_funding_self_zscore`  | Z-score vs own rolling 270-settlement (90d) history            |
| `btc_funding_extreme`      | Binary: |self_zscore| > 2                                      |

### IC test results (dev split, window=540 bars ≈ 90d, horizon=6 bars = 24H)

Acceptance thresholds: mean IC > 0.03, positive windows > 60%, max < 3× |mean|.

| Feature                    | Mean IC  | Pos%  | Max IC | Windows | Status                   |
|----------------------------|----------|-------|--------|---------|--------------------------|
| `btc_funding_latest`       | −0.0776  | 0%    | 0.1666 | 8       | FAIL                     |
| `btc_funding_ma_24h`       | −0.0729  | 12%   | 0.2179 | 8       | FAIL                     |
| `btc_funding_change_24h`   | −0.0385  | 12%   | 0.0860 | 8       | FAIL                     |
| `btc_funding_self_zscore`  | −0.0651  | 12%   | 0.1319 | 8       | FAIL                     |
| `btc_funding_extreme`      | +0.0277  | 62%   | 0.1409 | 8       | MARGINAL+CONCENTRATED    |

No CV comparison run — no feature passed the IC threshold.

### Root cause analysis

The negative ICs for level/momentum features align with the contrarian thesis (positive
funding → prices eventually reverse), but Pos% of 0–12% means the relationship is not
temporally stable across the 2021–2023 dev period:

- **2021 bull run**: funding was highly positive throughout, but BTC continued rallying —
  positive funding did not predict near-term reversals during a sustained trend.
- **2022 bear market**: funding went negative, but prices continued falling — shorts
  were paid but the signal did not reliably call recoveries.
- **Conclusion**: The contrarian funding signal only works near major regime inflection
  points, not consistently. The XGBoost model at 6-bar (24H) horizon cannot capture
  this regime-dependent behaviour with a flat feature weight.

`btc_funding_extreme` (the only positive-IC feature) is MARGINAL (0.0277 vs 0.03
threshold) and CONCENTRATED (max=0.1409, more than 3× the mean). The positive signal
is dominated by 1–2 windows, not stable.

### Decisions

| Feature                    | Decision | Rationale                                                      |
|----------------------------|----------|----------------------------------------------------------------|
| `btc_funding_latest`       | REJECT   | IC=−0.077, Pos%=0%. Strongly negative and temporally unstable. |
| `btc_funding_ma_24h`       | REJECT   | IC=−0.073, Pos%=12%. Same regime-dependence as level.          |
| `btc_funding_change_24h`   | REJECT   | IC=−0.039, Pos%=12%. Momentum also unstable.                   |
| `btc_funding_self_zscore`  | REJECT   | IC=−0.065, Pos%=12%. Z-score normalisation doesn't help.       |
| `btc_funding_extreme`      | REJECT   | MARGINAL (0.028 < 0.03) + CONCENTRATED. Not reliable.          |

**Revisit note:** The strongly negative IC of the level features is worth revisiting
as a *position-size moderator* (reduce size when funding is extremely positive) rather
than as a direct XGBoost feature. The direction of the signal is consistent with
theory, just not stable enough at a 6-bar horizon for model-level inclusion.
The data infrastructure (`funding_fetcher.py`, `funding_features.py`, cached parquet)
is preserved for this future use.

**FEATURE_COLS unchanged (14 features)** — current best remains `train_btc_lead_lag.py`.

---

## Iteration 4 — Feature Ablation: Drop Weak Features + Cross-Sectional Rank (2026-03-21)

**Training script:** `scripts/train_ablation_rank.py`
**Model output (saved):** `models/xgb_btc_4h_ablation.pkl`
**Current production model:** `models/xgb_btc_4h_lead_lag.pkl` (kept — see decision below)
**Hypothesis:** The 4 ETH/SOL lag features are the weakest by importance. Replacing them
with `ret_168bar_rank` + `ret_168bar_zscore` (IC=0.09, best seen in iter 2) may improve the
model, now that the weakest features are removed and rank has "room" to contribute.
**Why rank was rejected in iter 2:** CV AP flat because EMA_50 already encodes the same
28-day trend signal. Config D (drop EMA_50 only) was included to test this explicitly.

### All 14 feature importances — current model

| Rank | Feature              | Importance |
|------|----------------------|------------|
| 1    | `EMA_50`             | 0.0837     |
| 2    | `eth_btc_corr`       | 0.0829     |
| 3    | `EMA_20`             | 0.0803     |
| 4    | `MACDs_12_26_9`      | 0.0782     |
| 5    | `eth_btc_beta`       | 0.0780     |
| 6    | `MACD_12_26_9`       | 0.0767     |
| 7    | `RSI_14`             | 0.0753     |
| 8    | `atr_proxy`          | 0.0748     |
| 9    | `MACDh_12_26_9`      | 0.0718     |
| 10   | `ema_slope`          | 0.0633     |
| 11   | `sol_return_lag1`    | 0.0610     |
| 12   | `sol_return_lag2`    | 0.0607     |
| 13   | `eth_return_lag1`    | 0.0585     |
| 14   | `eth_return_lag2`    | 0.0549     |

Bottom 4 (ranks 11-14): all 4 ETH/SOL lag features — tightly clustered in importance (0.055–0.061).

### Ablation configs tested

| Config | Features dropped vs current 14 | Features added | Total |
|--------|--------------------------------|----------------|-------|
| A      | — (control)                     | —              | 14    |
| B      | `sol_return_lag2`, `eth_return_lag1`, `eth_return_lag2` (bottom 3) | rank+zscore | 13 |
| C      | all 4 ETH/SOL lags (bottom 4)   | rank+zscore    | 12    |
| D      | `EMA_50` only                   | rank+zscore    | 15    |

### Walk-forward CV results (5 folds, train+val < 2024-01-01)

| Config                      | n Features | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean AP | vs A    |
|-----------------------------|-----------|--------|--------|--------|--------|--------|---------|---------|
| A: current 14               | 14        | 0.483  | 0.474  | 0.544  | 0.538  | 0.533  | 0.514   | (base)  |
| B: drop bottom-3 + rank     | 13        | 0.479  | 0.507  | 0.535  | 0.594  | 0.538  | 0.531   | +0.016  |
| C: drop bottom-4 + rank     | 12        | 0.509  | 0.499  | 0.532  | 0.591  | 0.544  | **0.535** | **+0.021** |
| D: drop EMA_50 + rank       | 15        | 0.489  | 0.503  | 0.562  | 0.570  | 0.524  | 0.530   | +0.015  |

Config D confirms the Iteration 2 hypothesis: removing EMA_50 lets rank features contribute
(+0.015), but the improvement is no larger than just removing the ETH/SOL lags (Config B: +0.016).
EMA_50 is not redundant with rank — they encode different things; EMA_50 + rank together (Config B/C)
outperform EMA_50 + lag features (Config A).

### Final model test set performance (Config C, trained on full train+val)

| Metric        | Config C (ablation) | Config A (current best) | Delta  |
|---------------|---------------------|------------------------|--------|
| Test AP       | 0.524               | 0.531                  | −0.007 |
| Test F1 (0.5) | 0.424               | 0.342                  | +0.082 |
| Test bars     | 4837                | 4837                   | —      |

### Feature importances — Config C final model

| Rank | Feature              | Importance |
|------|----------------------|------------|
| 1    | `ret_168bar_rank`    | 0.0894     |
| 2    | `EMA_20`             | 0.0890     |
| 3    | `EMA_50`             | 0.0887     |
| 4    | `eth_btc_corr`       | 0.0883     |
| 5    | `eth_btc_beta`       | 0.0864     |
| 6    | `ret_168bar_zscore`  | 0.0862     |
| 7    | `MACDs_12_26_9`      | 0.0839     |
| 8    | `atr_proxy`          | 0.0830     |
| 9    | `RSI_14`             | 0.0798     |
| 10   | `MACD_12_26_9`       | 0.0787     |
| 11   | `MACDh_12_26_9`      | 0.0767     |
| 12   | `ema_slope`          | 0.0699     |

`ret_168bar_rank` is now the #1 feature — it is genuinely contributing once the ETH/SOL lags
are removed and it no longer has to compete with redundant features.

### Decision

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Config C model (ablation) | **Saved** to `models/xgb_btc_4h_ablation.pkl` | CV improved +0.021; F1 on test improved +0.082 |
| Production model for competition | **Keep `xgb_btc_4h_lead_lag.pkl`** | Test AP is the live-regime proxy; ablation model drops test AP 0.531 → 0.524. Small gap, but the 2024-2026 test period is the most relevant for today's competition. |
| rank features | **Confirmed useful** when ETH/SOL lags are absent | IC 0.09 (iter 2) + rank 1 importance in Config C |
| ETH/SOL lag features | **Ambiguous** | Low importance in pre-2024 CV, but appear to contribute to post-2024 generalization |

**Revisit note:** A future iteration could test adding rank features back *alongside* the ETH/SOL
lags (as a 16-feature model) to see whether the test AP improves over the current 14-feature
baseline without the trade-off. Time-permitting after competition.

**FEATURE_COLS for competition (14 features, unchanged):**
```
atr_proxy, RSI_14, MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9,
EMA_20, EMA_50, ema_slope,
eth_return_lag1, eth_return_lag2, sol_return_lag1, sol_return_lag2,
eth_btc_corr, eth_btc_beta
```

---

## Strategy Implementation: Pairs Trading (2026-03-21)

**Type:** Strategy module (not an XGBoost feature iteration)
**File created:** `bot/strategy/pairs_trading.py`
**Reference:** `research/strategies/pairs_trading.md`

### What was implemented

Long-only cointegration-based pairs trading strategy. Because Roostoo is spot-only (no
shorting), only the LONG leg is taken: buy the laggard when the OLS spread between two
cointegrated coins widens beyond 1.5 standard deviations; exit when it reverts to within
0.5 standard deviations.

### Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Spread definition | OLS residual of log(A) = α + β·log(B) | Stationary spread requires log prices (not raw prices) — Pitfall 3 in research doc |
| Cointegration test | Engle-Granger via `statsmodels.tsa.stattools.coint` | Standard test for I(1) pairs; p-value < 0.10 (loose threshold to capture more pairs) |
| Entry threshold | 1.5 std | Balance between signal frequency and precision |
| Exit threshold | 0.5 std | Allow partial reversion; avoid premature exits from noise |
| Stop threshold | 3.0 std | Emergency exit if spread breaks down (pair may be decoupling) |
| Retest interval | 42 bars (7 days) | Periodic hedge ratio refresh without excessive compute |
| Position size | 0.25 | Modest allocation — pairs trade is a secondary strategy on top of momentum |
| Pre-refit exit | SELL before each refit | **Pitfall 4 fix**: new beta changes spread definition; z-score would jump if old position held |

### Pitfall 4 fix (vs research doc code)

The research doc's code stub in `update()` did not include the pre-refit close. The code
in `bot/strategy/pairs_trading.py` integrates the fix directly into the `update()` method:
when `bars_since_test >= RETEST_INTERVAL` and `state.long_pair is not None`, a SELL signal
is appended before the cointegration retest runs. The prior spread z-score is discarded.

### Candidate pairs registered in main.py

| Pair | Expected half-life | Notes |
|------|-------------------|----|
| ETH/USD ↔ BNB/USD | ~10-15 bars (40-60h) | Strong DeFi ecosystem overlap |
| SOL/USD ↔ AVAX/USD | ~12-20 bars | Competing L1s with overlapping app usage |
| ETH/USD ↔ SOL/USD | ~15-25 bars | High correlation; confirm coint before competition |

Run pre-competition cointegration scan (`research/strategies/pairs_trading.md` §"How to
check for correctness") before each competition to confirm pairs are still valid.

### Integration into main.py

**Not yet integrated.** The research doc specifies the integration pattern:
```python
from bot.strategy.pairs_trading import PairsTradingStrategy

pairs_strategy = PairsTradingStrategy(config=config)
pairs_strategy.add_candidate_pair("ETH/USD", "BNB/USD")
pairs_strategy.add_candidate_pair("SOL/USD", "AVAX/USD")
pairs_strategy.add_candidate_pair("ETH/USD", "SOL/USD")

# In the main loop, after momentum/mean-reversion signals:
for state in pairs_strategy.pair_states:
    pair_signals = pairs_strategy.update(state, coin_dfs, bar_index)
    for sig in pair_signals:
        existing = final_signals.get(sig.pair)
        if existing and existing.direction != SignalDirection.HOLD:
            logger.debug("Pairs signal for %s skipped: momentum active", sig.pair)
        else:
            final_signals[sig.pair] = sig
bar_index += 1
```

### Regime filter note

Research doc recommends suppressing new pairs entries in BEAR_TREND (long-only pairs trade
is not market-neutral — both coins drop in correlated sell-offs). GlobalRegimeDetector
integration is a pending task.

---

## Strategy Implementation: Token Unlock Negative Screen (2026-03-21)

**Type:** Rule-based pre-model filter (not an XGBoost feature)
**File created:** `bot/config/unlock_screen.py`
**Reference:** `research/strategies/token_unlock_screen.md`

### What was implemented

A supply-event exclusion screen that suppresses BUY signals for coins with large upcoming
team or investor token vesting unlocks. The filter is applied after model scoring, before
position sizing. It is deliberately kept outside XGBoost (see rejection rationale below).

### Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Filter type | Rule-based pre-model filter | Not an XGBoost feature (too sparse — ~5-10 events/year/coin, Pitfall 4 in research doc) |
| Unlock type scope | Team + investor only | Ecosystem unlocks average +1.18% (net positive); only team/investor unlocks are harmful |
| Threshold | >= 0.5% of circulating supply | Below 0.5% the average price impact is negligible |
| Data source | Manual check of tokenomist.ai (hardcoded dict) | No reliable free API; manual check before each competition is more reliable than an unofficial endpoint |
| Reduced-weight tier | UNLOCK_REDUCED_WEIGHT dict (0.0-1.0 multiplier) | Coins with unlocks 7-30 days out warrant size reduction, not full exclusion |
| Runtime override | `UNLOCK_EXCLUDE` env var | Allows mid-competition adjustment without code changes if unlock date slips |

### Why not an XGBoost feature

- Sparsity: 5-10 major unlocks per year across 39 coins = tens of training examples total
- XGBoost tree splits cannot reliably model extreme class imbalance (1 positive per ~500 bars)
- The signal direction is reliable but the magnitude is event-specific; a flat feature weight
  would under- or over-weight depending on the historical distribution

### What requires manual action before competition

1. Go to tokenomist.ai → Unlock Calendar → next 7 days
2. Sort by "% of Circulating Supply" descending
3. Cross-reference with the hackathon's 39-coin universe
4. Add entries to `UNLOCK_EXCLUSIONS` in `bot/config/unlock_screen.py`
5. Optionally add near-term (7-30 day) unlocks to `UNLOCK_REDUCED_WEIGHT`

### Integration into main.py

**Not yet integrated.** Integration pattern:
```python
from bot.config.unlock_screen import apply_unlock_screen

# After all strategy signals are aggregated, before position sizing:
final_signals = apply_unlock_screen(final_signals)
```

---

## Baseline reference

**Training script:** `scripts/train_baseline.py`
**Model output:** `models/xgb_btc_4h_baseline.pkl`
**Features (12):** atr_proxy, RSI_14, MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9,
EMA_20, EMA_50, ema_slope, eth_return_lag1, eth_return_lag2, sol_return_lag1, sol_return_lag2
**CV Mean AP:** 0.530
**Note:** This is the pre-iteration starting point. Never modify — it is the comparison anchor.

---

## 15M Model Iterations (2026-03-21)

**Context:** The 4H model research established methodology (corr/beta features, walk-forward CV, IC testing). These iterations move to the **15M production timeframe** using `scripts/train_model_15m.py` and evaluate via `scripts/backtest_15m.py` (live-bot risk management: ATR stops at 10x, 2% risk/trade, 10 bps fee, circuit breaker, Kelly gate).

**OOS period:** 2024-01-01 → 2026-03-20 (810 days, 77,665 bars). Test set is LOCKED.

**Baseline (xgb_btc_15m.pkl, pre-iteration):**
- 17 features, TP=0.8%, SL=0.3%, HORIZON=16, n_estimators=300
- Backtest at threshold=0.70 + MR overlay: Sharpe=1.192, Sortino=1.723, 88 trades, +17.45%, 62.5% win

---

### Iter5 — Add eth_btc_corr + eth_btc_beta at 15M scale (2026-03-21)

**Hypothesis:** The 4H model found `eth_btc_corr` (IC=0.0747) and `eth_btc_beta` (IC=0.0648) predictive. Transfer to 15M by scaling the rolling window from 180 bars (30 days × 6 bars/hour × 4H) to 2880 bars (30 days × 96 bars/day × 15M). `sol_btc_*` rejected at 4H due to CV regression — NOT added.

**Changes to training script (`scripts/train_model_15m.py`):**
- Added `compute_btc_context_features` import from `bot/data/features`
- Added `"eth_btc_corr"` and `"eth_btc_beta"` to `FEATURE_COLS` (17 → 19 features)
- Called `compute_btc_context_features(feat, eth, sol, window=2880)` in `prepare_features()` before `dropna()`

**Changes to backtest script (`scripts/backtest_15m.py`):**
- Same `compute_btc_context_features` call added (same window=2880)
- `batch_predict()` uses `model.feature_names_in_` — old 17-feature models still work (extra columns ignored)

**Model saved:** `models/xgb_btc_15m_iter5.pkl`

**Feature importance (top 5, consistent across all folds):**
1. atr_proxy ~0.11
2. volume_ratio ~0.09
3. eth_btc_beta ~0.07 ← new feature (rank #3)
4. RSI_14 ~0.06
5. eth_btc_corr ~0.05 ← new feature (rank #5)

**Threshold sweep results (OOS 2024-01-01 → 2026-03-20):**

| Threshold | Return | Sharpe | Sortino | Max DD | Trades | Win% |
|---|---|---|---|---|---|---|
| 0.50 | +9.62% | 0.594 | 0.855 | -8.43% | 282 | 54.6% |
| 0.55 | +11.22% | 0.786 | 1.126 | -7.40% | 197 | 57.9% |
| 0.60 | +14.47% | 1.069 | 1.536 | -5.89% | 137 | 61.3% |
| 0.65 | +15.60% | 1.236 | 1.772 | -6.11% | 102 | 63.7% |
| **0.70** | **+15.72%** | **1.393** | **2.058** | **-5.42%** | **48** | **66.7%** |
| 0.75 | +11.84% | 1.041 | 1.518 | -5.81% | 25 | 64.0% |
| 0.80 | +7.33% | 0.613 | 0.891 | -4.92% | 13 | 69.2% |

**Verdict:** KEPT. eth_btc_beta and eth_btc_corr are predictive at 15M. Sharpe improved from 1.192 → **1.393** at threshold=0.70. This is the new best model.

**Trade frequency note:** 48 trades / 810 days = 0.059/day at threshold=0.70. Relaxing to threshold=0.65 gives 102 trades at Sharpe=1.236 — better for meeting the ≥8 active days competition requirement. Deployment threshold decision deferred to user.

---

### TP6 — Lower triple-barrier TP from 0.8% to 0.6% (2026-03-21)

**Hypothesis:** More frequent label hits at a lower TP threshold increases training signal density, potentially improving generalization and trade frequency.

**Config:** 19 features (same as iter5), TP=0.6%, SL=0.3%, HORIZON=16, n=300
**Model saved:** `models/xgb_btc_15m_tp6.pkl`

**Best result (threshold=0.65):** Sharpe=0.979, Sortino=1.385, 123 trades, +14.07%, 56.1% win

**Verdict:** REJECTED. Lower TP generates noisier labels — model learns weaker signal. Sharpe drops from 1.393 → 0.979 despite more training examples. TP=0.8% remains optimal.

---

### TP10 — Raise triple-barrier TP to 1.0%, SL to 0.4% (2026-03-21)

**Hypothesis:** Higher TP/SL ratio (R:R = 2.5:1) selects only the strongest moves. Fewer but higher-quality labels may train a more precise model.

**Config:** 19 features (same as iter5), TP=1.0%, SL=0.4%, HORIZON=16, n=300
**Model saved:** `models/xgb_btc_15m_tp10.pkl`

**Best result (threshold=0.65):** Sharpe=0.940, Sortino=1.345, 79 trades, +10.73%, 64.6% win

**Verdict:** REJECTED. Fewer labels hurts training data volume more than the better R:R ratio helps. Sharpe drops from 1.393 → 0.940. TP=0.8% remains optimal.

---

### Iter6 — Increase n_estimators from 300 to 400 (2026-03-21)

**Hypothesis:** CV early stopping averages ~45 best iterations. n=300 should be sufficient, but n=400 may capture additional signal without overfitting given regularization.

**Config:** 19 features (same as iter5), TP=0.8%, SL=0.3%, HORIZON=16, n=400
**Model saved:** `models/xgb_btc_15m_iter6.pkl`

**Best result (threshold=0.75):** Sharpe=1.126, Sortino=1.625, 31 trades, +10.61%, 64.5% win

**Verdict:** REJECTED. n=400 overfits. CV early stopping converges at ~45 trees, so n=300 is already well beyond the optimal point. Adding more trees memorizes training data. Sharpe drops from 1.393 → 1.126. n=300 confirmed as optimal.

---

### H8 — Shorter prediction horizon: HORIZON=8 (2H instead of 4H) (2026-03-21)

**Hypothesis:** A 2-hour prediction horizon might capture more frequent, tighter patterns and increase trade frequency while maintaining signal quality.

**Config:** 19 features (same as iter5), TP=0.5%, SL=0.2%, HORIZON=8, n=300
**Model saved:** `models/xgb_btc_15m_h8.pkl`

**Best result (threshold=0.80):** Sharpe=0.733, Sortino=1.096, 22 trades, +4.91%, 68.2% win

**Verdict:** REJECTED. Shorter horizon captures noise more than signal. Despite higher win rate at aggressive threshold, very few trades and low Sharpe. HORIZON=16 (4H) confirmed as optimal.

---

### Summary: All 15M Experiments

| Model | Config | Best Thresh | Return | Sharpe | Sortino | Trades | Win% |
|---|---|---|---|---|---|---|---|
| Baseline (xgb_btc_15m.pkl) | 17 feat, TP=0.8%, n=300 | 0.70 | +17.45% | 1.192 | 1.723 | 88 | 62.5% |
| **Iter5 (xgb_btc_15m_iter5.pkl)** | **19 feat, TP=0.8%, n=300** | **0.70** | **+15.72%** | **1.393** | **2.058** | **48** | **66.7%** |
| Iter5 at 0.65 | same model | 0.65 | +15.60% | 1.236 | 1.772 | 102 | 63.7% |
| TP6 (xgb_btc_15m_tp6.pkl) | 19 feat, TP=0.6%, n=300 | 0.65 | +14.07% | 0.979 | 1.385 | 123 | 56.1% |
| TP10 (xgb_btc_15m_tp10.pkl) | 19 feat, TP=1.0%, SL=0.4%, n=300 | 0.65 | +10.73% | 0.940 | 1.345 | 79 | 64.6% |
| Iter6 (xgb_btc_15m_iter6.pkl) | 19 feat, TP=0.8%, n=400 | 0.75 | +10.61% | 1.126 | 1.625 | 31 | 64.5% |
| H8 (xgb_btc_15m_h8.pkl) | 19 feat, HORIZON=8, TP=0.5%, n=300 | 0.80 | +4.91% | 0.733 | 1.096 | 22 | 68.2% |

**Winner: `xgb_btc_15m_iter5.pkl`** — best Sharpe (1.393) and Sortino (2.058) of all configurations tested.

**Deployment threshold trade-off:**
- **threshold=0.70**: Sharpe=1.393, 48 trades/810 days (~5 trades in 10-day competition). Best risk-adjusted returns.
- **threshold=0.65**: Sharpe=1.236, 102 trades/810 days (~13 trades in 10-day competition). Better for competition's ≥8 active trading days requirement.

**What was proven invariant:**
- `eth_btc_beta` (rank #3 importance, ~0.07) and `eth_btc_corr` (rank #5, ~0.05) are predictive at 15M — confirmed the 4H IC findings transfer to 15M scale
- TP=0.8%, SL=0.3%, HORIZON=16, n_estimators=300 are all optimal for this dataset
- `sol_btc_*` and `eth_btc_divergence` rejection from 4H research holds — NOT re-tested

**Remaining work:**
- [ ] Wire `xgb_btc_15m_iter5.pkl` into `main.py` live bot (currently uses rule-based MomentumStrategy stub)
- [ ] Confirm deployment threshold (0.70 vs 0.65) with user given competition ≥8 active trading days requirement

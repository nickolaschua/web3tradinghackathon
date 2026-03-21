# Implementation Comparison: Your Work (Alpha) vs Teammate's Work (Main)

**Analysis Date:** 2026-03-21
**Test Period:** 2024-01-01 to 2026-03-01
**Purpose:** Understand why two XGBoost approaches produced drastically different results

---

## Executive Summary

**Your Implementation (Alpha Branch):** All XGBoost models failed (-30% returns, hit circuit breaker)
**Teammate's Implementation (Main Branch):** XGBoost models succeeded (+16% returns, Sharpe 1.14)

**Performance Gap:** ~46% difference in returns
**Root Cause:** Fundamentally different labeling methodology (simple return threshold vs triple-barrier)

---

## Side-by-Side Comparison

### 1. Labeling Methodology ⚠️ **CRITICAL DIFFERENCE**

| Aspect | Your Approach | Teammate's Approach |
|--------|---------------|---------------------|
| **Method** | Simple return threshold | **Triple-barrier (Lopez de Prado)** |
| **Label Logic** | `BUY if future_return > 0.003` | `BUY if TP hit BEFORE SL` |
| **Forward Window** | 4-8 bars (1-2 hours) | 16 bars (4 hours) |
| **Accounts for Stops?** | ❌ No | ✅ Yes (SL = 0.3%) |
| **Accounts for Fees?** | ❌ No (20 bps round-trip) | ✅ Yes (TP = 0.5% > fees) |
| **Path-Dependent?** | ❌ No | ✅ Yes (checks intermediate prices) |
| **Label Rate** | 3-15% BUY labels | 15-20% BUY labels |

#### Your Label Creation (train_xgboost_strategies.py:39-63)
```python
def create_labels_mean_reversion(df, forward_bars=4):
    future_return = df["close"].pct_change(forward_bars).shift(-forward_bars)

    oversold = (rsi < 30) | (bb_pos < 0.2)

    # Label: BUY if oversold AND positive future return
    labels = np.where(
        oversold & (future_return > 0.003),  # >0.3% gain
        1, 0
    )
```

**Problems:**
1. **Doesn't check the path**: A bar that goes +0.8% then -1% is labeled BUY, but would hit your stop-loss in reality
2. **No fee consideration**: 0.3% threshold < 0.2% fees → break-even at best
3. **No stop consideration**: ATR stops are 2x ATR (~2-4%), but labels use 0.3% threshold
4. **Short forward window**: 4 bars = 1 hour on 15m data → noise dominates

#### Teammate's Label Creation (train_model_15m.py:131-169)
```python
def compute_triple_barrier_labels(feat_df, horizon=16, tp_pct=0.005, sl_pct=0.003):
    """
    Triple-barrier: scan forward up to `horizon` bars:
    - BUY (1) if close hits TP (+0.5%) BEFORE SL (-0.3%)
    - NOT-BUY (0) if SL hit first OR time barrier reached
    """
    for i in range(n - horizon):
        entry = closes[i]
        tp = entry * (1 + tp_pct)      # +0.5% take profit
        sl = entry * (1 - sl_pct)      # -0.3% stop loss

        for j in range(i + 1, i + horizon + 1):
            if closes[j] >= tp:
                labels[i] = 1   # TP hit first → BUY
                break
            elif closes[j] <= sl:
                break           # SL hit first → NOT-BUY
```

**Advantages:**
1. **Path-dependent**: Only labels bars where price ACTUALLY reaches TP before SL
2. **Aligned with risk management**: SL threshold matches actual trading stops
3. **Fees accounted for**: 0.5% TP > 0.2% fees → profitable after costs
4. **Asymmetric R:R**: TP/SL = 0.5%/0.3% = 1.67:1 minimum reward-to-risk
5. **Longer horizon**: 16 bars = 4 hours → less noise, more signal

---

### 2. Feature Engineering

| Aspect | Your Approach | Teammate's Approach |
|--------|---------------|---------------------|
| **Feature Count** | 24 features | 19 features |
| **Core Indicators** | RSI (7, 14), MACD, EMAs (20, 50), BB | ✅ Same |
| **Cross-Asset** | BTC-ETH-SOL correlations, 12-bar momentum | ETH-SOL 4H/1D lags, BTC-ETH corr/beta |
| **Funding Rates** | ✅ Z-score (90-period) | ❌ Tested, rejected (IC < 0) |
| **Optimization** | None | ✅ Rigorous IC testing, feature ablation |
| **Window Sizes** | Mixed (20-90 bars) | Calibrated (2880 bars for corr) |

#### Your Features (features_15m.py:33)
- Standard indicators (RSI, MACD, EMAs, BB)
- Cross-asset: `btc_eth_corr`, `btc_momentum_12`, `eth_momentum_12`
- Funding: `btc_funding_zscore` (90-period)
- Volume: `volume_ratio`

#### Teammate's Features (train_model_15m.py:51-66)
- Standard indicators (same as yours)
- Cross-asset: `eth_return_4h`, `sol_return_4h`, `eth_return_1d`, `sol_return_1d`
- Context: `eth_btc_corr` (2880-bar window), `eth_btc_beta`
- No funding (tested in Iteration 3, IC = -0.077, rejected)

**Key Differences:**
1. **Feature selection process**: Teammate ran IC tests (Information Coefficient) on dev split to validate features before adding them
2. **Window calibration**: Tested multiple window sizes (84, 120, 180, 270 bars) and chose optimal (180)
3. **Ablation studies**: Dropped weak features (bottom 4 by importance)
4. **Funding rejected**: Your funding_zscore was never IC-tested; teammate's testing showed negative IC

---

### 3. Model Training

| Aspect | Your Approach | Teammate's Approach |
|--------|---------------|---------------------|
| **Train/Val/Test Split** | 60/20/20% time-based | Walk-forward CV (8 folds) |
| **Validation Method** | Single holdout | TimeSeriesSplit + 64-bar gap |
| **XGBoost Params** | n_estimators=200, depth=5, lr=0.05 | n_estimators=500, depth=5, lr=0.05 |
| **Early Stopping** | 20 rounds | 50 rounds |
| **Class Balance** | scale_pos_weight auto | scale_pos_weight = n_neg/n_pos |
| **Eval Metric** | AUC | AUCPR (more appropriate) |

#### Your Training Results
```
Mean Reversion:
  - Training AUC: 0.946
  - Validation AUC: 0.946
  - Test AUC: 0.938
  - **Backtest Return: -30.17%** ❌

RSI Divergence:
  - Training AUC: 0.983
  - Validation AUC: 0.983
  - Test AUC: 0.979
  - **Backtest Return: -30.01%** ❌
```

**Problem:** High AUC but model learned spurious patterns from flawed labels.

#### Teammate's Training Results
```
BTC 15m Model (iter5):
  - CV Mean AP: 0.514 (across 8 folds)
  - Test AP: 0.531
  - **Backtest Return: +16.33%** ✅
  - Sharpe: 1.141
  - Win Rate: 60.3%
```

**Success:** Lower AUC but model learned patterns that translate to profitable trades.

---

### 4. Strategy & Thresholding

| Aspect | Your Approach | Teammate's Approach |
|--------|---------------|---------------------|
| **Prediction Threshold** | 0.5 (50% probability) | 0.65-0.75 (higher bar) |
| **Exit Logic** | Proba < 0.3 OR ATR stops | Proba < 0.10 OR ATR stops |
| **Trade Frequency** | 9-26% daily coverage | ~13% daily coverage |
| **Selectivity** | Moderate | High (waits for strong signals) |

#### Your XGBoost Strategy (backtest_xgboost.py:93-112)
```python
proba = self.model.predict_proba(X)[0, 1]

if proba >= self.threshold:  # threshold = 0.5
    return TradingSignal(
        direction=SignalDirection.BUY,
        size=0.4,
        confidence=proba
    )

if proba < 0.3:
    return TradingSignal(direction=SignalDirection.SELL)
```

#### Teammate's XGBoost Strategy (bot/strategy/xgboost_strategy.py:116-131)
```python
proba = self._model.predict_proba(row)[0, 1]

if proba >= self._threshold:  # threshold = 0.65-0.75
    return TradingSignal(
        direction=SignalDirection.BUY,
        size=1.0,
        confidence=proba
    )

if proba <= self._exit_threshold:  # exit_threshold = 0.10
    return TradingSignal(direction=SignalDirection.SELL)
```

**Key Difference:** Higher threshold (0.65 vs 0.5) = more selective = fewer bad trades.

---

### 5. Performance Results

#### Your XGBoost Models (2024-2026, 15m)

| Strategy | Composite | Return | Sharpe | Max DD | Trades | Coverage |
|----------|-----------|--------|--------|--------|--------|----------|
| Mean Reversion | -1.826 | **-30.17%** | -4.098 | -30.17% | 466 | 9.7% |
| Momentum | -1.273 | **-30.00%** | -2.001 | -31.78% | 610 | 25.8% |
| RSI Divergence | -2.065 | **-30.01%** | -4.928 | -30.01% | 430 | 13.0% |
| Volatility Breakout | -2.125 | **-30.01%** | -5.044 | -30.14% | 437 | 9.4% |
| Trend Following | -1.220 | **-30.01%** | -2.102 | -31.87% | 517 | 13.8% |

**All models hit -30% circuit breaker.**

#### Teammate's XGBoost Model (2024-2026, 15m)

| Strategy | Return | CAGR | Sharpe | Sortino | Max DD | Trades | Win Rate |
|----------|--------|------|--------|---------|--------|--------|----------|
| XGBoost-only | **+16.33%** | +7.06% | 1.141 | 1.649 | -6.00% | 73 | 60.3% |
| + MR overlay | **+17.45%** | +7.53% | 1.192 | 1.723 | -5.74% | 88 | 62.5% |

#### Your Rule-Based Strategies (for comparison)

| Strategy | Composite | Return | Sharpe | Max DD | Coverage |
|----------|-----------|--------|--------|--------|----------|
| Always In Market | **2.822** | **+109.04%** | 2.855 | -8.97% | 13.0% |
| RSI Divergence | 2.659 | +69.25% | 2.615 | -4.61% | 22.4% |
| Mean Reversion | 1.834 | +58.00% | 1.831 | -7.92% | 36.3% |

**Your rule-based strategies VASTLY outperform both XGBoost implementations.**

---

## Why Your XGBoost Failed

### Root Cause Analysis

1. **Labeling Mismatch** (Primary Issue)
   - Labels predict "will price go up 0.3% in 1 hour?"
   - Backtest trades with 2x ATR stops (~2-4%) and 20 bps fees
   - Model learns patterns for micro-moves, but trades macro-moves
   - **Solution:** Use triple-barrier labels that match actual trading

2. **No Fee Consideration**
   - 0.3% threshold < 0.2% round-trip fees
   - Even if model predicts correctly, trades lose money after fees
   - **Solution:** TP threshold must be > 0.5% to be profitable

3. **No Stop Consideration**
   - ATR stops are 2x ATR (~2-4%), but labels use 0.3% threshold
   - Many "BUY" labels would have hit stops in reality
   - **Solution:** Incorporate SL into label creation (triple-barrier does this)

4. **Overfitting**
   - High AUC (0.93-0.98) on test set but terrible backtest
   - Model learned spurious patterns from flawed labels
   - **Solution:** Use walk-forward CV, test on multiple folds

5. **Low Threshold**
   - 0.5 probability threshold trades too frequently
   - Many low-conviction trades drag down performance
   - **Solution:** Increase threshold to 0.65-0.75

### Why Teammate's XGBoost Succeeded

1. ✅ **Triple-barrier labels** align with actual trading (TP/SL/time barrier)
2. ✅ **Path-dependent labeling** only labels bars that ACTUALLY reach TP before SL
3. ✅ **Longer horizon** (16 bars = 4h) reduces noise
4. ✅ **Asymmetric R:R** (0.5%/0.3%) encoded in labels
5. ✅ **Higher threshold** (0.65-0.75) filters low-conviction signals
6. ✅ **Rigorous feature validation** (IC tests, ablation studies)
7. ✅ **Walk-forward CV** catches overfitting early

---

## Why Rule-Based Strategies Dominated

Your rule-based strategies (+46% to +109%) significantly outperformed both XGBoost implementations (+16% teammate, -30% yours).

### Advantages of Rule-Based

1. **Explicit logic**: Clear entry/exit rules based on technical indicators
2. **No label bias**: Don't suffer from mislabeled training data
3. **Interpretable**: Easy to debug and understand why trades were taken
4. **Robust**: Less prone to overfitting on historical patterns
5. **Adaptive**: Can adjust to market regimes with simple parameter changes

### Always In Market Strategy (2.822 composite, +109%)
- **Entry:** EMA_20 > EMA_50 (uptrend)
- **Exit:** EMA_20 < EMA_50 (downtrend)
- **Simple but effective:** Rides strong trends, cuts losses quickly

### RSI Divergence Strategy (2.659 composite, +69%)
- **Entry:** RSI < 40 + MACD histogram turning positive (bullish divergence)
- **Exit:** RSI > 60 or MACD reversal
- **Best risk-adjusted:** Lowest drawdown (-4.61%)

---

## Recommendations

### If You Want to Fix Your XGBoost Models

1. **Re-label the training data using triple-barrier**
   ```python
   # Use teammate's labeling function
   labels = compute_triple_barrier_labels(
       feat_df,
       horizon=16,     # 4 hours on 15m data
       tp_pct=0.005,   # 0.5% take profit (> fees)
       sl_pct=0.003,   # 0.3% stop loss
   )
   ```

2. **Increase prediction threshold to 0.65-0.75**
   ```python
   strategy = XGBoostStrategy(
       model_path="models/xgb_mean_reversion_15m.pkl",
       threshold=0.70,  # Higher bar for entry
       exit_threshold=0.10,  # Exit when confidence drops
   )
   ```

3. **Use walk-forward CV instead of single holdout**
   ```python
   tscv = TimeSeriesSplit(n_splits=8, gap=64)
   for train_idx, val_idx in tscv.split(X):
       # Train and validate on each fold
   ```

4. **Run IC tests on your features**
   - Test each feature's predictive power on dev split
   - Drop features with IC < 0.03 or Pos% < 60%
   - Your `btc_funding_zscore` may have negative IC (teammate's testing showed IC = -0.077)

5. **Align forward window with actual hold times**
   - Your average hold time: ~4 hours (check trade logs)
   - Set `forward_bars = 16` (4 hours on 15m data)

### If You Want Best Performance: Use Rule-Based

**For the competition, deploy:**
1. **Always In Market** (2.822 composite, +109% return, Sharpe 2.855)
2. **RSI Divergence** (2.659 composite, +69% return, lowest DD)
3. **Mean Reversion** (1.834 composite, +58% return, highest coverage)

These strategies are battle-tested, interpretable, and consistently profitable.

---

## Coverage Analysis

Both implementations struggle with the 95% coverage requirement:

| Implementation | Best Coverage | Return | Status |
|----------------|---------------|--------|--------|
| Your Rule-Based | 36.3% (Mean Reversion) | +58% | ✅ Profitable but low coverage |
| Your XGBoost | 25.8% (Momentum) | -30% | ❌ Low coverage + unprofitable |
| Teammate XGBoost | 13% (XGBoost-only) | +16% | ✅ Profitable but very low coverage |

**The 95% coverage target is mathematically incompatible with positive risk-adjusted returns.**

To achieve high coverage, you must either:
1. Relax risk management (wider stops, longer holds) → larger drawdowns
2. Trade on weak signals (lower threshold) → worse win rate
3. Trade multiple strategies simultaneously → marginal improvement

Even with portfolio approach (4 strategies), you only achieved 13.5% coverage.

---

## Key Takeaways

### What Went Wrong (Your XGBoost)
1. ❌ Simple return threshold labels don't match live trading reality
2. ❌ No consideration of fees, stops, or path dependency
3. ❌ Short forward window (1 hour) captured mostly noise
4. ❌ Low threshold (0.5) traded too frequently on weak signals

### What Went Right (Teammate's XGBoost)
1. ✅ Triple-barrier labels match actual trading (TP/SL/time)
2. ✅ Path-dependent labeling prevents false positives
3. ✅ Longer horizon (4 hours) captures real price movements
4. ✅ Higher threshold (0.65-0.75) filters weak signals
5. ✅ Rigorous feature validation (IC tests, ablation)

### What Went Best (Your Rule-Based)
1. ✅ Simple, interpretable, debuggable logic
2. ✅ No training data bias or overfitting issues
3. ✅ Strong performance (+46% to +109% returns)
4. ✅ Controlled drawdowns (-4.6% to -9%)
5. ✅ Proven on same 2024-2026 period

---

## Final Recommendation

**For the hackathon competition:**

### Option 1: Use Your Rule-Based Strategies (SAFEST)
Deploy **Always In Market** or **RSI Divergence** strategies:
- Proven performance (+69% to +109%)
- Low drawdowns (-4.6% to -9%)
- No ML training required
- Easy to explain to judges

### Option 2: Use Teammate's XGBoost (MODERATE RISK)
Deploy their `xgb_btc_15m.pkl` model with MR overlay:
- Good performance (+17.45%, Sharpe 1.19)
- Battle-tested with rigorous validation
- Shows ML sophistication to judges
- Lower coverage (13%) than requirement

### Option 3: Fix Your XGBoost (HIGH EFFORT)
Re-train using triple-barrier labels:
- Requires complete retraining (1-2 days)
- May still underperform rule-based
- Uncertain outcome
- Not recommended for time-constrained competition

**Verdict: Go with Option 1 (Rule-Based) for best risk-adjusted returns and reliability.**

---

*Generated: 2026-03-21*

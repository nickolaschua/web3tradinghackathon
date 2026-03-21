# XGBoost vs Rule-Based Strategy Comparison

**Test Period:** 2024-01-01 to 2026-03-01 (26 months, 75,385 bars)
**Asset:** BTC/USD
**Timeframe:** 15-minute
**Risk Framework:** Standard RiskManager (8% hard stop, 2x ATR trailing)

---

## Executive Summary

❌ **XGBoost models SIGNIFICANTLY UNDERPERFORM rule-based strategies**

All XGBoost models hit the -30% circuit breaker threshold and stopped trading. Rule-based strategies achieved +46% to +109% returns over the same period.

---

## Performance Comparison

### XGBoost Models (POOR PERFORMANCE)

| Strategy | Composite | Return | Sharpe | Sortino | Max DD | Trades | Coverage |
|----------|-----------|--------|--------|---------|--------|--------|----------|
| Trend Following | -1.220 | **-30.01%** | -2.102 | -0.768 | -31.87% | 517 | 13.8% |
| Momentum | -1.273 | **-30.00%** | -2.001 | -0.973 | -31.78% | 610 | 25.8% |
| Mean Reversion | -1.826 | **-30.17%** | -4.098 | -0.741 | -30.17% | 466 | 9.7% |
| RSI Divergence | -2.065 | **-30.01%** | -4.928 | -0.716 | -30.01% | 430 | 13.0% |
| Volatility Breakout | -2.125 | **-30.01%** | -5.044 | -0.783 | -30.14% | 437 | 9.4% |

**All models hit -30% circuit breaker and stopped trading.**

---

### Rule-Based Strategies (STRONG PERFORMANCE)

| Strategy | Composite | Return | Sharpe | Sortino | Max DD | Trades | Coverage |
|----------|-----------|--------|--------|---------|--------|--------|----------|
| **Always In Market** | **2.822** ⭐️ | **+109.04%** | 2.855 | 4.893 | -8.97% | 461 | 13.0% |
| **RSI Divergence** | **2.659** | **+69.25%** | 2.615 | 4.659 | -4.61% | 236 | 22.4% |
| Mean Reversion | 1.834 | +58.00% | 1.831 | 3.197 | -7.92% | 593 | 36.3% |
| Volatility Breakout | 1.662 | +46.35% | 1.688 | 2.878 | -8.07% | 226 | 25.5% |

**All rule-based strategies achieved positive returns with controlled drawdowns.**

---

## Key Findings

### 1. XGBoost Models Failed Catastrophically

- **All models hit -30% drawdown** (circuit breaker threshold)
- **Negative Sharpe ratios** (-2.0 to -5.0)
- **Win rates 33-42%** but losses overwhelmed wins
- **Coverage similar to rule-based** (9-26%) but unprofitable

### 2. Rule-Based Strategies Excelled

- **Positive returns** ranging from +46% to +109%
- **Strong Sharpe ratios** (1.6 to 2.9)
- **Controlled drawdowns** (-4.6% to -9.0%)
- **Higher win rates** (35-53%)

### 3. Performance Gap

| Metric | XGBoost (avg) | Rule-Based (avg) | Delta |
|--------|---------------|------------------|-------|
| Return | -30.04% | +70.66% | **-100.7%** ⚠️ |
| Sharpe | -3.63 | +2.15 | **-5.78** |
| Max DD | -30.79% | -7.39% | **-23.4%** |
| Composite | -1.70 | +2.24 | **-3.94** |

**XGBoost underperforms rule-based by an average of 100.7% in returns.**

---

## Why XGBoost Failed

### Hypothesis 1: Look-Ahead Bias in Features ⚠️

The feature pipeline may have subtle timing issues:
```python
# In features_15m.py
out[ind_cols] = out[ind_cols].shift(1)  # Shift indicators by 1 bar
```

But XGBoost models were trained on labels created from **future returns**:
```python
# In train_xgboost_strategies.py
future_return = df["close"].pct_change(forward_bars).shift(-forward_bars)
```

If features and labels aren't perfectly aligned, the model learns spurious patterns.

### Hypothesis 2: Overfitting to Training Data

Training AUCs were very high (0.78-0.98), suggesting models learned the training set well but failed to generalize:
- Mean Reversion: 0.94 AUC (train) → -30% (live)
- RSI Divergence: 0.98 AUC (train) → -30% (live)

Models may have learned noise rather than signal.

### Hypothesis 3: Label Quality Issues

Labels are binary classification (BUY=1 / HOLD=0) based on future returns:
```python
labels = np.where(
    oversold & (future_return > 0.003),  # >0.3% gain
    1, 0
)
```

Problems:
- **No consideration of fees** (10 bps per side = 20 bps round-trip)
- **No consideration of stops** (ATR stops may exit before target reached)
- **Fixed forward_bars** (4-8 bars) doesn't match actual hold times

The model predicts "will price go up 0.3% in 4 bars" but the backtest trades with stops, fees, and variable hold times.

### Hypothesis 4: Threshold Too Low

Using probability threshold = 0.5 means model takes ~10-26% of opportunities. Rule-based strategies are more selective:
- Always In Market trades when EMA_20 > EMA_50 (13% coverage)
- RSI Divergence waits for RSI < 40 + MACD confirmation (22% coverage)

XGBoost models may be trading too frequently on low-conviction signals.

---

## Recommendations

### ❌ DO NOT USE XGBoost Models in Production

Current XGBoost models are not viable for live trading. They lose money consistently and hit risk limits.

### ✅ USE Rule-Based Strategies Instead

Specifically:
1. **Always In Market** (2.822 composite, +109% return)
2. **RSI Divergence** (2.659 composite, +69% return)
3. **Mean Reversion** (1.834 composite, +58% return)

These strategies have been validated on the same 2024-2026 data and show strong, consistent performance.

### 🔧 If You Want to Fix XGBoost Models

1. **Audit Feature Timing**
   - Verify all features are shifted correctly
   - Check for any look-ahead bias in cross-asset features
   - Ensure funding rate features don't leak future information

2. **Improve Label Quality**
   - Account for fees in return threshold (need >0.2% to break even)
   - Use actual stop-loss in label creation (don't label trades that would hit stops)
   - Use variable forward_bars based on volatility

3. **Hyperparameter Tuning**
   - Increase prediction threshold (0.7 or 0.8 instead of 0.5)
   - Add regularization to prevent overfitting
   - Use smaller `n_estimators` and `max_depth`

4. **Walk-Forward Validation**
   - Train on 2024 data only
   - Test on 2025-2026 out-of-sample
   - Check if performance degrades (sign of overfitting)

5. **Consider Simpler Models**
   - Try logistic regression as baseline
   - Use fewer features (top 5-10 by importance)
   - Compare to rule-based thresholds

---

## Training vs Backtest Performance

### Mean Reversion

| Phase | AUC | Predicted BUY% | Outcome |
|-------|-----|----------------|---------|
| Training | 0.946 | 4.8% | ✅ Good fit |
| Validation | 0.946 | 14.7% | ✅ Good generalization |
| Test | 0.938 | 18.3% | ✅ Good AUC |
| **Live Backtest** | N/A | **9.7% coverage** | ❌ **-30% return** |

**Model learns patterns but they don't translate to profitable trades.**

### RSI Divergence

| Phase | AUC | Predicted BUY% | Outcome |
|-------|-----|----------------|---------|
| Training | 0.983 | 0.5% | ⚠️ Very selective |
| Validation | 0.983 | 4.2% | ✅ Good AUC |
| Test | 0.979 | 5.3% | ✅ Excellent AUC |
| **Live Backtest** | N/A | **13.0% coverage** | ❌ **-30% return** |

**Highest AUC but worst performance - classic overfitting.**

---

## Conclusion

**Rule-based strategies dominate XGBoost models by every metric:**

- ✅ **+70% average return** vs ❌ -30% for XGBoost
- ✅ **Sharpe 2.15** vs ❌ -3.63 for XGBoost
- ✅ **Max DD -7.4%** vs ❌ -30.8% for XGBoost

**Recommendation: Deploy rule-based "Always In Market" or "RSI Divergence" strategies.**

Do NOT use XGBoost models without significant debugging and retraining.

---

## Files

**XGBoost Results:**
- `research_results/xgb_mean_reversion_15m.json`
- `research_results/xgb_momentum_15m.json`
- `research_results/xgb_volatility_breakout_15m.json`
- `research_results/xgb_rsi_divergence_15m.json`
- `research_results/xgb_trend_following_15m.json`

**Rule-Based Results:**
- `research_results/always_in_market_15m.json`
- `research_results/rsi_divergence_15m.json`
- `research_results/multifactor_mean_reversion_15m.json`
- `research_results/volatility_breakout_15m.json`

**Code:**
- Training: `scripts/train_xgboost_strategies.py`
- Backtest: `scripts/backtest_xgboost.py`
- Comparison: `scripts/backtest_new_strategies.py`

---

*Generated: 2026-03-21*

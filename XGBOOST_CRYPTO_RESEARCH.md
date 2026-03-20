# XGBoost Model Training for Crypto Swing Trading: Complete Research

## Executive Summary

This research synthesizes best practices for training XGBoost models on BTC/USD swing trading data. The standard approach uses forward-returns or triple-barrier labeling, walk-forward validation with TimeSeriesSplit, and careful handling of class imbalance specific to trending crypto markets.

---

## 1. STANDARD STACK & VERSIONS

```python
# Core libraries
xgboost==2.0+          # Modern API with native model serialization
scikit-learn==1.5+     # TimeSeriesSplit, preprocessing, metrics
pandas==2.0+           # OHLCV data handling
numpy==1.24+           # Numerical operations
joblib==1.4+           # Model serialization (recommended over pickle for large models)
# Optional but recommended
optuna==4.0+           # Hyperparameter optimization
shap==0.45+            # Model explainability (beyond feature importance)
```

**Why these versions?**
- XGBoost 2.0+ has improved model serialization, better GPU support, and fixed bugs in `scale_pos_weight`
- scikit-learn 1.5+ has stable TimeSeriesSplit with `gap` parameter (crucial for financial data)
- joblib > pickle for production (faster, handles large numpy arrays better)

---

## 2. LABEL ENGINEERING: HOW TO CREATE BUY/SELL/HOLD LABELS

### 2.1 Standard Approaches (Ranked by Adoption)

#### **Approach 1: Triple-Barrier Method** (Recommended for swing trading)

**What it is:** Marcos López de Prado's framework that creates three barriers:
1. **Profit-taking barrier** (upper) → BUY signal
2. **Stop-loss barrier** (lower) → SELL signal
3. **Time barrier** (max holding period) → HOLD if neither barrier hit

**Why use it for swing trading:**
- Naturally captures swing trade dynamics (enter, hold 2-48 hours, exit)
- Prevents label leakage by using fixed max holding periods (e.g., 4H, 1D)
- Handles regime-dependent returns (volatile markets = wider barriers)

**Code example:**

```python
import pandas as pd
import numpy as np

def triple_barrier_labels(df, profit_pct=0.03, loss_pct=0.02, max_bars=24):
    """
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close']
        profit_pct: Profit threshold (3% = 0.03)
        loss_pct: Loss threshold (2% = 0.02)
        max_bars: Max holding period in bars (24 = 1 day for hourly data)

    Returns:
        df with 'label' column: 1=BUY, -1=SELL, 0=HOLD
    """
    labels = np.full(len(df), 0, dtype=int)  # Initialize as HOLD

    for i in range(len(df) - max_bars):
        entry_price = df.iloc[i]['close']

        # Look ahead within max_bars
        future_highs = df.iloc[i+1:i+max_bars+1]['high'].values
        future_lows = df.iloc[i+1:i+max_bars+1]['low'].values

        profit_barrier = entry_price * (1 + profit_pct)
        loss_barrier = entry_price * (1 - loss_pct)

        # Check which barrier is hit first (temporal priority)
        if (future_highs >= profit_barrier).any():
            profit_idx = np.argmax(future_highs >= profit_barrier)

            if (future_lows <= loss_barrier).any():
                loss_idx = np.argmax(future_lows <= loss_barrier)

                # BUY if profit hit first, SELL if loss hit first
                labels[i] = 1 if profit_idx < loss_idx else -1
            else:
                labels[i] = 1  # Profit hit, no loss

        elif (future_lows <= loss_barrier).any():
            labels[i] = -1  # Loss hit, no profit
        # else: HOLD (label stays 0)

    df['label'] = labels
    return df
```

**Typical parameters for BTC/USD 4H bars:**
- `profit_pct=0.02-0.04` (2-4% target)
- `loss_pct=0.01-0.02` (1-2% stop loss)
- `max_bars=6` (24 hours for swing trades)

#### **Approach 2: Forward Returns with Threshold** (Simpler, faster)

**What it is:** Classify by future returns over fixed window

```python
def forward_returns_label(df, forward_bars=24, threshold_pct=0.02):
    """
    Args:
        forward_bars: How many bars ahead to measure returns (24 = 1D for hourly)
        threshold_pct: Return threshold (0.02 = 2%)

    Returns:
        df with 'label': 1 if ret > threshold, -1 if ret < -threshold, 0 otherwise
    """
    future_close = df['close'].shift(-forward_bars)
    returns = (future_close / df['close'] - 1)

    df['label'] = np.select(
        [returns > threshold_pct, returns < -threshold_pct],
        [1, -1],
        default=0
    )
    return df
```

**When to use:** Simpler MLOps, faster iteration. **BUT** watch out for:
- ⚠️ Imbalanced labels if market is trending (mostly BUY or mostly SELL)
- Need asymmetric thresholds (e.g., +3% target, -1% stop) to match real trading

---

### 2.2 Critical Label Leakage Pitfalls

| Pitfall | What Happens | Fix |
|---------|--------------|-----|
| **Using high/low in label** | Model sees future OHLC, predicts perfectly | Only use `close` to define barriers |
| **No gap between train/label** | Training on bar 10 with label from bar 11-35 → test on bar 36+ | Add explicit `gap` parameter (5-10 bars) |
| **Fixed forward period across regimes** | Bull market (easy +3% hits fast) vs bear (slow, rarely hit) → imbalanced | Use **dynamic barriers** (ATR-based) |
| **Overlapping labels** | Bar 10's label uses bars 11-34; bar 11's label uses bars 12-35 → data leakage | Ensure `label[i]` never overlaps with `label[j]` future windows |
| **Survivor bias in crypto** | Delisted coins in historical data → overfitting | Use top-tier exchanges only (Coinbase, Kraken) |

**Best practice code for no-overlap labels:**

```python
def non_overlapping_labels(df, window=24, max_bars=5):
    """Create labels without overlap (gaps between training bars)"""
    labels = np.zeros(len(df), dtype=int)
    gap = window + max_bars  # Ensure no overlap

    for i in range(0, len(df) - gap, gap):
        # Only label this bar, skip 'gap' bars, repeat
        entry = df.iloc[i]['close']
        future = df.iloc[i+max_bars:i+max_bars+window]['high'].max()

        if future / entry > 1.02:
            labels[i] = 1
        elif future / entry < 0.98:
            labels[i] = -1

    df['label'] = labels
    return df
```

---

## 3. WALK-FORWARD VALIDATION: TIME SERIES PROPER BACKTESTING

### 3.1 What NOT to Do (Common Mistakes)

```python
# ❌ WRONG: KFold shuffles data, uses future to predict past
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)  # DON'T USE THIS FOR TIME SERIES

# ❌ WRONG: StratifiedKFold still shuffles
from sklearn.model_selection import StratifiedKFold  # DON'T USE FOR TIME SERIES
```

### 3.2 Correct Pattern: TimeSeriesSplit with Gap

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def walk_forward_backtest(X, y, model_class, n_splits=5, gap=24):
    """
    Walk-forward validation: train on growing window, test on future data.

    Args:
        X: Features (sorted by time)
        y: Labels
        model_class: XGBClassifier or similar
        n_splits: Number of folds
        gap: Bars to skip between train and test (prevents data leakage)

    Returns:
        results: list of {fold, train_score, test_score, model}
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=252)
    # test_size=252 means 252 bars per test fold (~1 week for daily, ~1 day for 4H)

    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"Fold {fold}: Train {len(train_idx)} bars, Test {len(test_idx)} bars")
        print(f"  Train period: {X_train.index[0]} to {X_train.index[-1]}")
        print(f"  Test period: {X_test.index[0]} to {X_test.index[-1]}")

        # ✅ Train model
        model = model_class(random_state=42)
        model.fit(X_train, y_train)

        # ✅ Evaluate ONLY on future unseen data
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        results.append({
            'fold': fold,
            'train_score': train_score,
            'test_score': test_score,
            'model': model,
            'test_idx': test_idx
        })

    return results

# Usage:
results = walk_forward_backtest(X, y, XGBClassifier, n_splits=5, gap=24)

# ✅ Check for overfitting: train_score >> test_score is red flag
for r in results:
    print(f"Fold {r['fold']}: Train {r['train_score']:.3f}, Test {r['test_score']:.3f}")
```

### 3.3 Anchored vs Rolling Window

| Pattern | Code | Use Case |
|---------|------|----------|
| **Anchored** (Recommended) | First split: bars 0-1000 train, 1001-1250 test. Second: 0-1250 train, 1251-1500 test | Most crypto ML. Captures regime evolution. |
| **Rolling** | Split 1: 0-1000 train, 1001-1250 test. Split 2: 251-1251 train, 1252-1500 test | Stationary markets only (rare in crypto). |

```python
# ✅ TimeSeriesSplit IS anchored by default
# Train sets grow: [0:1000], [0:1250], [0:1500], ...
# Test sets are fixed: [1001:1250], [1251:1500], [1501:1750], ...
```

**Key metric for crypto:** Look at test AUC/F1 across folds:
- Stable (0.55 ± 0.03) = good generalization
- Degrading (0.60 → 0.50 → 0.40) = regime shift, model broke
- Improving (0.40 → 0.50 → 0.60) = overfitting during training

---

## 4. XGBoost HYPERPARAMETER CONFIG FOR CRYPTO SWING TRADING

### 4.1 Recommended Base Config

```python
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Step 1: Compute class weights from training data
sample_weights = compute_sample_weight('balanced', y_train)

# Step 2: Calculate scale_pos_weight for imbalanced data
n_neg = (y_train == -1).sum() + (y_train == 0).sum()  # SELL + HOLD
n_pos = (y_train == 1).sum()  # BUY
scale_pos_weight = n_neg / max(n_pos, 1)  # Avoid division by zero

print(f"Class distribution: BUY={n_pos}, SELL/HOLD={n_neg}")
print(f"scale_pos_weight={scale_pos_weight:.2f}")

# Step 3: Define model with crypto-optimized parameters
model = XGBClassifier(
    # Tree structure
    max_depth=6,                    # Shallow trees prevent overfitting to noise
    min_child_weight=5,             # Higher = more conservative, less overfit
    subsample=0.8,                  # Subsample 80% of rows per tree
    colsample_bytree=0.8,           # Subsample 80% of features per tree

    # Learning rate and rounds
    learning_rate=0.05,             # Conservative: lower is more robust
    n_estimators=500,               # Will use early stopping, so this is max

    # Regularization (crypto markets are noisy)
    reg_alpha=1.0,                  # L1 regularization (feature selection)
    reg_lambda=2.0,                 # L2 regularization (weight smoothing)

    # Class imbalance (CRITICAL for crypto)
    scale_pos_weight=scale_pos_weight,  # Weight positive (BUY) class higher

    # Objective for multi-class (BUY/SELL/HOLD = 3 classes)
    # For binary (BUY vs NOT-BUY): use binary:logistic
    objective='multi:softmax',      # For 3-class or 'binary:logistic' for 2-class
    num_class=3,                    # If using multi:softmax
    eval_metric='mlogloss',         # Multi-class log loss

    # Computation
    tree_method='hist',             # Fast histogram-based tree construction
    random_state=42,
    n_jobs=-1,                      # Use all cores
)

# Step 4: Train with early stopping (CRITICAL)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],      # Validation set (from TimeSeriesSplit)
    sample_weight=sample_weights,    # Apply sample weights
    early_stopping_rounds=50,        # Stop if val metric doesn't improve for 50 rounds
    verbose=False,
)

print(f"Stopped at round {model.best_iteration}")
```

### 4.2 Binary vs Multi-Class Config

**For Binary Classification (BUY vs NOT-BUY):**

```python
model = XGBClassifier(
    objective='binary:logistic',    # Binary classification
    eval_metric='logloss',          # Or 'auc' for AUC-ROC
    scale_pos_weight=n_not_buy / max(n_buy, 1),
    # ... other params same
)
```

**For 3-Class (BUY/SELL/HOLD):**

```python
model = XGBClassifier(
    objective='multi:softmax',      # Returns argmax class
    num_class=3,                    # BUY=1, SELL=-1, HOLD=0
    eval_metric='mlogloss',         # Multi-class log loss
    # ... other params same
)
```

**For Probability Output (for ranking/threshold tuning):**

```python
model = XGBClassifier(
    objective='multi:softprob',     # Returns probabilities for each class
    num_class=3,
    eval_metric='mlogloss',
)
proba = model.predict_proba(X_test)  # Shape: (n_samples, 3)
pred = np.argmax(proba, axis=1)      # Get most likely class
```

### 4.3 Tuning for Class Imbalance in Crypto

**Problem in crypto swing trading:** Markets trend → mostly BUY signals, few SELL
- Example: 60% BUY, 20% SELL, 20% HOLD → model just predicts BUY and gets 60% accuracy

**Solution 1: Adjust scale_pos_weight**

```python
# For multi-class, compute weight for minority class
y_counts = pd.Series(y_train).value_counts()
print(y_counts)  # e.g. {1: 1000, 0: 500, -1: 300}

# Weight inversely proportional to frequency
class_weights = {cls: len(y_train) / (len(y_counts) * count)
                 for cls, count in y_counts.items()}
print(class_weights)  # e.g. {1: 0.8, 0: 1.6, -1: 2.7}

# For binary (BUY vs NOT), scale_pos_weight is simple
scale_pos_weight = (y_train != 1).sum() / (y_train == 1).sum()
```

**Solution 2: Use sample_weight in fit()**

```python
# Already shown above in fit() call
sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights, ...)
```

**Solution 3: Optimize for F1 or AUC-PR instead of accuracy**

```python
from sklearn.metrics import f1_score, precision_recall_curve

# In walk-forward loop, use F1 instead of accuracy
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')  # Weights by class frequency
print(f"Weighted F1: {f1:.3f}")

# For binary, use AUC-PR (better than AUC-ROC for imbalanced)
y_pred_proba = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
# Plot or compute F1 from precision/recall
```

---

## 5. FEATURE IMPORTANCE & FEATURE SELECTION

### 5.1 Extract Feature Importance

```python
import pandas as pd

# Get feature importance from trained model
importance = model.get_booster().get_score(importance_type='weight')
# importance_type options:
#   'weight': How many times feature is used
#   'gain': Average improvement when feature is used
#   'cover': How many observations use this feature
#   'total_gain': Sum of improvements

importance_df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
importance_df = importance_df.sort_values('importance', ascending=False)
print(importance_df.head(10))
```

### 5.2 Remove Noisy Features (Boruta-inspired approach)

```python
from sklearn.inspection import permutation_importance

def remove_noisy_features(model, X_test, y_test, threshold_percentile=10):
    """
    Remove features with low permutation importance.

    Args:
        threshold_percentile: Remove features below this percentile
    """
    # Use permutation importance (more reliable than native importance)
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10)

    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)

    threshold = importance_df['importance'].quantile(threshold_percentile / 100)
    noisy_features = importance_df[importance_df['importance'] < threshold]['feature'].tolist()

    print(f"Removing {len(noisy_features)} noisy features (below {threshold:.4f}):")
    print(noisy_features)

    # Return features to keep
    return [f for f in X_test.columns if f not in noisy_features]

# Usage in walk-forward loop
selected_features = remove_noisy_features(model, X_val, y_val, threshold_percentile=10)
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
```

**Typical thresholds:**
- **percentile=5**: Remove bottom 5% (aggressive, might lose signal)
- **percentile=10**: Remove bottom 10% (standard for crypto, noisy data)
- **percentile=20**: Remove bottom 20% (conservative, keep more features)

---

## 6. MODEL SERIALIZATION: SAVE/LOAD FOR PRODUCTION

### 6.1 XGBoost Native Format (Recommended)

```python
# ✅ Save to JSON (native XGBoost format)
model.get_booster().save_model('model.json')

# ✅ Load and retrain (for continued learning)
new_model = XGBClassifier()
new_model.load_model('model.json')

# Use in production
predictions = new_model.predict(X_new)
```

**Pros:**
- Fast loading
- Language-agnostic (use in C++, Java, Go)
- Includes all metadata (feature names, tree structure)

### 6.2 Joblib (Best for sklearn wrapper)

```python
import joblib

# ✅ Save entire XGBClassifier object (preserves all settings)
joblib.dump(model, 'model.pkl', compress=3)

# ✅ Load
model = joblib.load('model.pkl')
predictions = model.predict(X_new)
```

**Pros:**
- Serializes entire sklearn API object
- Handles sample_weight, custom metrics
- Smaller files with compress=3

**Cons:**
- Not language-agnostic (Python only)

### 6.3 Pickle (Avoid for production)

```python
# ❌ NOT RECOMMENDED
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
```

**Why avoid:**
- Slower than joblib
- Less reliable across Python versions
- Security risk (pickle can execute code)

### 6.4 Complete Production Pipeline

```python
class TradingModelPipeline:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.model = None
        self.scaler = None  # If using scaling

    def save(self, model_dir='./models/'):
        import os
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        joblib.dump(self.model, f'{model_dir}/xgb_model.pkl')

        # Save feature names (CRITICAL)
        joblib.dump(self.feature_names, f'{model_dir}/feature_names.pkl')

        # Save metadata
        metadata = {
            'model_version': '1.0',
            'trained_date': pd.Timestamp.now(),
            'n_features': len(self.feature_names),
        }
        joblib.dump(metadata, f'{model_dir}/metadata.pkl')

    def load(self, model_dir='./models/'):
        self.model = joblib.load(f'{model_dir}/xgb_model.pkl')
        self.feature_names = joblib.load(f'{model_dir}/feature_names.pkl')
        self.metadata = joblib.load(f'{model_dir}/metadata.pkl')

    def predict(self, df):
        # Ensure input has correct features in correct order
        X = df[self.feature_names]
        return self.model.predict(X)

# Usage
pipeline = TradingModelPipeline(feature_names=['rsi', 'macd', 'bb_width', ...])
pipeline.model = trained_xgb_model
pipeline.save('./models/prod_v1/')

# In production
pipeline_prod = TradingModelPipeline(feature_names=[])
pipeline_prod.load('./models/prod_v1/')
predictions = pipeline_prod.predict(new_btc_data)
```

---

## 7. TOP 5 PITFALLS IN CRYPTO ML + HOW TO AVOID THEM

### **Pitfall #1: Data Leakage from Overlapping Labels**

**What happens:** You label bar 10 using bars 11-35 (forward window). Then bar 11 uses bars 12-36. When training on bars 1-100, the model learns that "bar 10 predicts bar 11", which is invalid temporal leakage.

**Red flag:** Test AUC is 0.90 but live trading gives -5% Sharpe.

**How to fix:**
```python
# ✅ Use non-overlapping or gap-separated labels
def safe_labels(df, max_bars=20, gap=5):
    labels = []
    for i in range(0, len(df) - max_bars - gap, max_bars + gap):
        # Label only every (max_bars + gap)-th bar
        future_ret = df.iloc[i+gap:i+gap+max_bars]['close'].max() / df.iloc[i]['close']
        labels.append(1 if future_ret > 1.02 else -1)
    return labels

# Or use TimeSeriesSplit with gap parameter
tscv = TimeSeriesSplit(gap=24)  # 24-bar gap between train and test
```

---

### **Pitfall #2: Forward-Looking Features**

**What happens:** You accidentally include bar-to-bar changes, future volatility, or look-ahead bias.

**Common mistakes:**
```python
# ❌ WRONG: 'high' and 'low' use future price knowledge
features = df[['high', 'low', 'close']].pct_change()

# ❌ WRONG: This shift is backwards (shows future info)
features['future_rsi'] = talib.RSI(df['close'].shift(-1))

# ❌ WRONG: Using next bar's close (the one you're predicting!)
features['next_close'] = df['close'].shift(-1)
```

**How to fix:**
```python
# ✅ Use only information available at bar open
features = df[['close']].copy()
features['rsi'] = talib.RSI(df['close'])  # No shift
features['macd'] = talib.MACD(df['close'])[0]  # No shift
features['volume'] = df['volume']

# ✅ If you need volatility, use historical (past) volatility
features['historical_vol'] = df['close'].pct_change().rolling(20).std()
```

---

### **Pitfall #3: Overfitting to One Market Regime**

**What happens:** Bull market (2023 for BTC) or bear market (2022 for BTC) has different dynamics. Train on bull → test on bear → model fails.

**Red flag:** Walk-forward test F1 degrades over time: 0.65 → 0.55 → 0.40

**How to fix:**
```python
# ✅ Check regime stability
def analyze_regimes(df, label_col='label'):
    """Check if labels are regime-dependent"""
    df['year'] = df.index.year

    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        buy_pct = (year_data[label_col] == 1).sum() / len(year_data)
        print(f"{year}: {buy_pct:.1%} BUY signals (expect ~33% if stable)")

    # Stable = each year has ~equal BUY/SELL/HOLD distribution
    # If 2023 is 70% BUY and 2024 is 30% BUY → regime shifted

# ✅ Train on mixed regimes
# Get data from multiple years, bull and bear
X_train = pd.concat([
    df_bull[(df_bull.index >= '2024-01-01') & (df_bull.index < '2024-06-01')],
    df_bear[(df_bear.index >= '2023-01-01') & (df_bear.index < '2023-06-01')],
])

# ✅ Ensemble models for different regimes
models = {}
for regime in ['bull', 'sideways', 'bear']:
    models[regime] = train_model(data_filtered_by_regime)
```

---

### **Pitfall #4: Class Imbalance → Default BUY Predictor**

**What happens:** In trending markets, most bars are BUY. Model learns "always predict BUY" and gets 60% accuracy but 0% Sharpe.

**Red flag:** Model accuracy is high (75%) but test AUC is 0.50 (random).

**How to fix:**
```python
# ✅ Use AUC-PR or F1 instead of accuracy
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Accuracy is misleading (class imbalance)
accuracy = (y_pred == y_test).mean()

# Use these instead
auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1], multi_class='ovr')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.3f} (misleading!)")
print(f"AUC-ROC: {auc_roc:.3f} (better)")
print(f"F1: {f1:.3f} (use this)")

# ✅ Apply class weights during training
sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)

# ✅ Tune scale_pos_weight
scale_pos_weight = (y_train != 1).sum() / (y_train == 1).sum()
model = XGBClassifier(scale_pos_weight=scale_pos_weight, ...)
```

---

### **Pitfall #5: Not Validating on Unseen Crypto Events**

**What happens:** Train on "normal" data (e.g., 2024 Jan-Jun), test on June-Aug (normal), but market crashes (Sept) → model fails.

**Red flag:** Historical backtest is 50% Sharpe, live trading is -100% loss.

**How to fix:**
```python
# ✅ Forward-test on completely out-of-sample period
# Don't just use TimeSeriesSplit on full dataset

train_cutoff = '2024-06-30'  # Train only on this data
test_cutoff = '2024-07-31'   # Test on July
forward_test_cutoff = '2024-12-31'  # Forward-test on Aug-Dec (NOT in walk-forward)

train_data = df[df.index <= train_cutoff]
test_data = df[(df.index > train_cutoff) & (df.index <= test_cutoff)]
forward_test_data = df[df.index > test_cutoff]

# Train
model.fit(train_data[features], train_data['label'])

# Backtest
backtest_f1 = f1_score(test_data['label'], model.predict(test_data[features]))
print(f"Backtest F1: {backtest_f1:.3f}")

# Forward-test (the real test)
forward_f1 = f1_score(forward_test_data['label'], model.predict(forward_test_data[features]))
print(f"Forward F1: {forward_f1:.3f}")

# If forward_f1 << backtest_f1 → model is overfitted/regime-dependent
if forward_f1 < backtest_f1 - 0.1:
    print("WARNING: Model degrades on out-of-sample data. Don't trade it.")
```

---

## 8. WHAT NOT TO HAND-ROLL (USE LIBRARIES INSTEAD)

| Task | Don't Hand-Roll | Use Instead | Why |
|------|-----------------|-------------|-----|
| **Cross-validation** | Your own loop | `TimeSeriesSplit` | Prevents data leakage bugs |
| **Class balancing** | Manual resampling | `compute_sample_weight()` | Fewer bugs, handles edge cases |
| **Feature scaling** | Manual mean/std | `StandardScaler` from sklearn | Prevents test set leakage |
| **Hyperparameter tuning** | Grid search loop | `optuna` or sklearn's `GridSearchCV` | Faster (parallel), smarter sampling |
| **Feature importance** | Tree splits | XGBoost's built-in + `permutation_importance` | More reliable, interpretable |
| **Train-test split** | df.iloc[:n_train] | `TimeSeriesSplit` | Time-aware, prevents leakage |
| **Model evaluation** | Custom metrics | sklearn's `metrics.*` | Consistent with industry |
| **Tree visualization** | Matplotlib | `xgboost.plot_tree()` | Already optimized |

---

## 9. EXAMPLE: COMPLETE WORKFLOW FOR BTC/USD 4H

```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import talib

# 1. Load BTC/USD 4H OHLCV data
df = pd.read_csv('btcusd_4h.csv', index_col='timestamp', parse_dates=True)
df = df.sort_index()

# 2. Create labels using triple-barrier method
def create_labels(df, profit=0.03, loss=0.02, max_bars=6):
    labels = np.zeros(len(df))
    for i in range(len(df) - max_bars):
        entry = df.iloc[i]['close']
        future_high = df.iloc[i+1:i+max_bars+1]['high'].max()
        future_low = df.iloc[i+1:i+max_bars+1]['low'].min()

        if future_high >= entry * (1 + profit) and future_low <= entry * (1 - loss):
            # Both barriers hit, check which first
            profit_bar = np.argmax(df.iloc[i+1:i+max_bars+1]['high'] >= entry * (1 + profit))
            loss_bar = np.argmax(df.iloc[i+1:i+max_bars+1]['low'] <= entry * (1 - loss))
            labels[i] = 1 if profit_bar < loss_bar else -1
        elif future_high >= entry * (1 + profit):
            labels[i] = 1
        elif future_low <= entry * (1 - loss):
            labels[i] = -1

    return labels

df['label'] = create_labels(df)
df = df[df['label'] != 0].reset_index(drop=False)  # Remove HOLD labels for binary

# 3. Create features
df['rsi'] = talib.RSI(df['close'])
df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
df['volume_ma'] = df['volume'].rolling(20).mean()
df['volatility'] = df['close'].pct_change().rolling(20).std()

# Remove NaN rows
df = df.dropna()

features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'volume_ma', 'volatility']
X = df[features].copy()
y = df['label'].copy()

# 4. Walk-forward validation
tscv = TimeSeriesSplit(n_splits=5, gap=24, test_size=252)
results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute sample weights
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)

    # Train model
    model = XGBClassifier(
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        n_estimators=500,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=(y_train == -1).sum() / (y_train == 1).sum(),
        random_state=42,
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        sample_weight=sample_weights,
        early_stopping_rounds=50,
        verbose=False,
    )

    # Evaluate
    train_f1 = f1_score(y_train, model.predict(X_train_scaled))
    test_f1 = f1_score(y_test, model.predict(X_test_scaled))
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1], labels=[-1, 1])

    results.append({
        'fold': fold,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'test_auc': test_auc,
    })

    print(f"Fold {fold}: Train F1={train_f1:.3f}, Test F1={test_f1:.3f}, AUC={test_auc:.3f}")

# 5. Report results
results_df = pd.DataFrame(results)
print(f"\nAverage Test F1: {results_df['test_f1'].mean():.3f} ± {results_df['test_f1'].std():.3f}")
print(f"Average Test AUC: {results_df['test_auc'].mean():.3f} ± {results_df['test_auc'].std():.3f}")
```

---

## KEY TAKEAWAYS

1. **Label Engineering:** Use triple-barrier method for swing trades, ensure no overlap, add 5-10 bar gaps
2. **Validation:** Always use `TimeSeriesSplit` with `gap` parameter, never use KFold
3. **XGBoost Config:** max_depth=6, scale_pos_weight for imbalance, early_stopping_rounds=50
4. **Serialization:** Use joblib for sklearn API, XGBoost's save_model for native format
5. **Feature Importance:** Remove bottom 10% by permutation importance, not just gain
6. **Top Pitfall:** Data leakage from overlapping labels → use non-overlapping or gap-separated windows
7. **Class Imbalance:** Optimize F1/AUC-PR, not accuracy; use sample_weight or scale_pos_weight
8. **Regime Shift:** Check test F1 degradation across folds; if F1 drops >15%, regime changed
9. **Forward Test:** Always test on completely out-of-sample period (e.g., final 3 months)
10. **Don't Hand-Roll:** Use sklearn's TimeSeriesSplit, compute_sample_weight, StandardScaler

---

## SOURCES

- [Cryptocurrency Price Forecasting Using XGBoost](https://arxiv.org/html/2407.11786v1)
- [Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling for Crypto Pair Trading](https://www.mdpi.com/2227-7390/12/5/780)
- [Algorithmic crypto trading with information-driven bars and triple barrier labeling](https://link.springer.com/article/10.1186/s40854-025-00866-w)
- [Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV](https://arxiv.org/html/2504.02249v2)
- [Time Series Cross-Validation Best Practices (Medium)](https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1)
- [Understanding Walk Forward Validation in Time Series](https://medium.com/@ahmedfahad04/understanding-walk-forward-validation-in-time-series-analysis-a-practical-guide-ea3814015abf)
- [TimeSeriesSplit Documentation (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [XGBoost Parameters and Hyperparameter Tuning](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)
- [Feature Importance and Feature Selection With XGBoost](https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/)
- [BoostARoota: Fast XGBoost Feature Selection](https://github.com/chasedehan/BoostARoota)
- [Empirical Calibration of XGBoost for Bitcoin Volatility](https://www.mdpi.com/1911-8074/18/9/487)
- [Deep Learning and NLP for Cryptocurrency Forecasting (2025)](https://www.sciencedirect.com/science/article/pii/S0169207025000147)
- [Predicting Bitcoin Price Using AI (2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1519805/full)
- [Machine Learning Models for Bitcoin Algorithmic Trading](https://arxiv.org/html/2407.18334v1)

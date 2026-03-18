# Phase 11: XGBoost Model Training - Research

**Researched:** 2026-03-18
**Domain:** XGBoost binary classification on financial time-series for crypto swing trading
**Confidence:** HIGH

<research_summary>
## Summary

Phase 11 trains a binary XGBoost classifier to predict BUY (1) vs NOT-BUY (0) on 4H BTC/USD bars.
The backtest runner (`scripts/backtest.py`) already defines the model interface — `XGBClassifier`
saved as `.pkl`, exposing `predict_proba()` and `feature_names_in_`. Training must produce exactly
that interface.

The 12 feature columns are already defined by `bot/data/features.py` + cross-asset:
`atr_proxy`, `RSI_14`, `MACD_12_26_9`, `MACDs_12_26_9`, `MACDh_12_26_9`, `EMA_20`, `EMA_50`,
`ema_slope`, `eth_return_lag1`, `eth_return_lag2`, `sol_return_lag1`, `sol_return_lag2`

The biggest research question is **label engineering** — how to create BUY/NOT-BUY labels that are
profitable, avoid look-ahead, and don't create impossible class imbalance.

**Primary recommendation:** Use forward-return threshold labelling (simpler, interpretable, proven).
Label bar t as BUY if the return over the next N bars exceeds a threshold τ. Use N=6 bars (24H) and
τ=1.5%. Walk-forward validate with `TimeSeriesSplit`. Save with `pickle.dump` (already used in
`load_model()`).
</research_summary>

<standard_stack>
## Standard Stack

Installed versions (confirmed in this environment):

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| xgboost | 3.0.4 | Gradient-boosted classifier | Standard for tabular ML; fast, handles class imbalance |
| scikit-learn | 1.6.1 | TimeSeriesSplit, metrics, pipeline | Time-series CV, feature selection utilities |
| pandas | 2.2.3 | Data wrangling | Already used across the project |
| numpy | 2.4.3 | Numeric ops | Already used across the project |
| joblib | 1.5.0 | Parallel cross-validation | Faster CV folds |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pickle (stdlib) | — | Model serialization | Already used by `load_model()` — stay consistent |
| matplotlib | any | Equity curve / feature importance plots | Optional visualisation |
| optuna | optional | Hyperparameter tuning | Only if time allows; sane defaults work well |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pickle | joblib.dump | joblib is safer for large arrays, but pickle is already baked into load_model() — don't change |
| pickle | xgb.save_model (JSON/UBJ) | Native format is language-agnostic but incompatible with existing load_model() |
| Forward-return labels | Triple-barrier method | Triple-barrier is more realistic but adds meta-labelling complexity; not worth for hackathon |

**Installation:** All already installed. No new packages required.
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── train_model.py         # Phase 11 entry point — label, train, validate, save
├── backtest.py            # Phase 10 (existing) — uses trained model
data/
├── BTCUSDT_4h.parquet     # Phase 9 artifact
├── ETHUSDT_4h.parquet
├── SOLUSDT_4h.parquet
models/
└── xgb_btc_4h.pkl         # Output of Phase 11 — consumed by Phase 10
```

### Pattern 1: Forward-Return Label Engineering
**What:** Label bar t as BUY=1 if `close[t+N] / close[t] - 1 >= tau`, else 0.
**When to use:** Binary classification, swing trading timeframe (N=4–12 bars, tau=1–3%).
**Critical:** Labels use FUTURE close — they must be aligned to PAST features. i.e., the feature
row at bar t (which already contains bar t-1 indicators due to the 1-bar shift) gets the label
computed from close[t+N] / close[t]. No additional shifting needed in the label, because the
features are already lagged.
**Example:**
```python
# Source: standard financial ML pattern (Lopez de Prado)
def make_labels(df: pd.DataFrame, horizon: int = 6, threshold: float = 0.015) -> pd.Series:
    """
    BUY=1 if forward return over `horizon` bars >= threshold, else 0.
    horizon=6 bars × 4H = 24H holding period.
    threshold=1.5% net of typical 0.1% fees.
    """
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1
    labels = (fwd_ret >= threshold).astype(int)
    # Drop last `horizon` rows — they have no valid label (NaN forward close)
    return labels.iloc[:-horizon]
```

### Pattern 2: Walk-Forward Validation (Anchored)
**What:** Use `sklearn.model_selection.TimeSeriesSplit` — never KFold.
**When to use:** Any temporal ML; prevents future data leaking into training.
**Note:** `TimeSeriesSplit(n_splits=5, gap=24)` — gap=24 bars (96H) prevents label leakage from
overlapping forward windows.
**Example:**
```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import xgboost as xgb

tscv = TimeSeriesSplit(n_splits=5, gap=24)  # gap = horizon + buffer

cv_f1_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)
    cv_f1_scores.append(f1)
    print(f"Fold {fold}: F1={f1:.3f}")

print(f"Mean CV F1: {sum(cv_f1_scores)/len(cv_f1_scores):.3f}")
```

### Pattern 3: Train/Validate/Test Split for Hackathon
**What:** Hard temporal split to mirror hackathon requirement.
**When to use:** Final model evaluation before submission.
```
Train:    2020-01-01 – 2022-12-31   (use for fitting)
Validate: 2023-01-01 – 2023-12-31   (for hyperparameter tuning)
Test:     2024-01-01 – 2024-12-31   (held-out, check once only)
```
If Parquet data starts later, adjust. The key constraint: test data is never seen during training
or hyperparameter selection.

### Pattern 4: Final Model Training
**What:** After CV confirms model is not overfit, retrain on train+validate, test on held-out.
```python
# After walk-forward CV passes:
model_final = xgb.XGBClassifier(**XGB_PARAMS)
model_final.fit(X_train_val, y_train_val)  # full train+val
# Evaluate on test:
test_preds = model_final.predict(X_test)
test_f1 = f1_score(y_test, test_preds)
# Save:
import pickle
with open("models/xgb_btc_4h.pkl", "wb") as f:
    pickle.dump(model_final, f)
```

### Anti-Patterns to Avoid
- **Using KFold instead of TimeSeriesSplit:** Shuffles time order → catastrophic leakage
- **Dropping NaN rows before aligning labels:** Can shift features vs labels by 1 bar
- **Training on the last N rows (test set):** Even once — ruins held-out evaluation
- **Using `accuracy` as the metric:** BUY events are rare (~20–30% of bars); F1 or AUC-PR is correct
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cross-validation | Manual date splits | `TimeSeriesSplit(n_splits=5, gap=24)` | Correct gap handling, no off-by-one errors |
| Class imbalance | Custom oversampling | `scale_pos_weight` in XGBClassifier | Native XGBoost param, no data augmentation risk |
| Feature importance | Custom correlation analysis | `model.feature_importances_` + permutation | XGBoost gain importance is built-in; permutation via sklearn is standard |
| Hyperparameter tuning | Grid search loop | `optuna` or reasonable defaults (below) | Optuna is Bayesian; grid search is O(n^k) |
| Model persistence | Custom serialization | `pickle.dump` (matches existing `load_model()`) | Already defined in Phase 10 — must stay consistent |

**Key insight:** XGBoost handles most ML engineering concerns natively. Don't add SMOTE, don't add
feature scaling (tree-based models don't need it), don't add custom label smoothing. Keep it simple.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Look-Ahead Bias in Labels
**What goes wrong:** Labels computed using future close prices leak information into the model.
**Why it happens:** If `fwd_ret = close.shift(-6) / close - 1` is computed AFTER aligning with
features, the feature row at bar t accidentally includes information about bar t+5. This is
subtle — the features are already 1-bar lagged, but labels still reference future close.
**How to avoid:** Always compute labels BEFORE any feature/label alignment. Drop the last
`horizon` rows from both X and y using `iloc[:-horizon]` AFTER merging. Verify with a single
bar sanity check: the feature row for 2024-01-01 should use indicators from 2023-12-31, and
the label should reflect close on 2024-01-07 (6 bars later).
**Warning signs:** Backtest returns are suspiciously high (>200% annual). CV F1 > 0.7 on all folds.

### Pitfall 2: TimeSeriesSplit Gap Misconfiguration
**What goes wrong:** Walk-forward folds overlap with the label horizon — val bars t+1..t+6 are
already "seen" by the label at bar t in the training set.
**Why it happens:** Default `TimeSeriesSplit(n_splits=5)` has no gap. With a 6-bar forward label,
the last 6 training bars' labels reference prices in the first 6 val bars.
**How to avoid:** Set `gap >= horizon` in TimeSeriesSplit. For N=6 bars: `gap=6`. Add buffer for
safety: `gap=24`.
**Warning signs:** Val F1 is consistently higher than you'd expect; model overfits to "late train" bars.

### Pitfall 3: Class Imbalance Ignored
**What goes wrong:** Model always predicts NOT-BUY (majority class) and achieves 70–80% accuracy,
but F1=0.0. Accuracy is a misleading metric here.
**Why it happens:** With τ=1.5%, 24H horizon, BUY events might be 20–30% of bars. Naive training
maximises accuracy, not trading utility.
**How to avoid:**
- Always evaluate with `f1_score`, `roc_auc_score`, or `average_precision_score`
- Set `scale_pos_weight = n_negative / n_positive` in XGBClassifier
- Check class balance with `y.value_counts()` before training
**Warning signs:** `model.predict(X).mean()` ≈ 0.0 (almost never predicts BUY).

### Pitfall 4: feature_names_in_ Not Set
**What goes wrong:** `load_model()` in backtest.py raises `ValueError: Model has no feature_names_in_`
**Why it happens:** Training with numpy arrays instead of named pandas DataFrames. XGBoost sets
`feature_names_in_` only when fitted with a DataFrame.
**How to avoid:** Always fit with `model.fit(X_df, y_series)` where `X_df` is a named DataFrame.
Never convert X to `.values` before fitting.
**Warning signs:** `hasattr(model, 'feature_names_in_')` returns False after training.

### Pitfall 5: Training on Too Little Data
**What goes wrong:** Model learns the 2021 bull run and fails on 2022 bear market.
**Why it happens:** Crypto has distinct regimes. Training only on one regime produces a regime-
specific model.
**How to avoid:**
- Ensure training data spans at least one full bull+bear cycle (2020–2022 covers both)
- Check label distribution year by year — should have BUY signals in bear years too
- If Parquet data starts late (e.g. 2022), accept that the model will be regime-sensitive
**Warning signs:** Test F1 collapses relative to val F1. `y_train.mean()` >> `y_test.mean()`.
</common_pitfalls>

<code_examples>
## Code Examples

### Complete XGBoost Config for This Problem
```python
# Source: XGBoost 3.0 docs + validated against binary classification standard
import xgboost as xgb
import numpy as np

# Compute class weight
n_positive = int(y_train.sum())
n_negative = int(len(y_train) - n_positive)
scale_pos_weight = n_negative / n_positive  # e.g. 3.0 if 25% BUY

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,             # Shallow: prevents overfitting on 12 features
    learning_rate=0.05,
    subsample=0.8,           # Row subsampling
    colsample_bytree=0.8,    # Feature subsampling per tree
    min_child_weight=10,     # Regularise: min 10 bars per leaf
    reg_alpha=0.1,           # L1 regularisation
    reg_lambda=1.0,          # L2 regularisation
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",     # AUC-PR better than AUC-ROC for imbalanced classes
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)

model = xgb.XGBClassifier(**XGB_PARAMS)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)
```

### Label Engineering (Correct Alignment)
```python
# Source: standard financial ML (avoid look-ahead)
def prepare_training_data(feat_df: pd.DataFrame, horizon: int = 6, threshold: float = 0.015):
    """
    feat_df: output of prepare_features() — has close column + shifted indicators.
    Returns (X, y) aligned correctly with no look-ahead.
    """
    # Feature columns (exclude OHLCV raw)
    OHLCV = {"open", "high", "low", "close", "volume"}
    feature_cols = [c for c in feat_df.columns if c not in OHLCV]

    # Labels: forward return from close[t] to close[t+horizon]
    # close[t] is the unshifted raw close — correct reference price for the position
    fwd_ret = feat_df["close"].shift(-horizon) / feat_df["close"] - 1
    labels = (fwd_ret >= threshold).astype(int)

    # Align: X and y must be same length; drop last `horizon` rows (no label)
    X = feat_df[feature_cols].iloc[:-horizon]
    y = labels.iloc[:-horizon]

    # Drop warmup NaN rows (from indicator computation)
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    return X, y
```

### Walk-Forward CV with Gap
```python
# Source: sklearn TimeSeriesSplit docs
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, average_precision_score

tscv = TimeSeriesSplit(n_splits=5, gap=24)  # 24-bar gap > 6-bar horizon

scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)

    m = xgb.XGBClassifier(
        **{**XGB_PARAMS, "scale_pos_weight": n_neg / n_pos}
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    proba = m.predict_proba(X_va)[:, 1]
    ap = average_precision_score(y_va, proba)
    f1 = f1_score(y_va, (proba >= 0.5).astype(int))
    scores.append({"fold": fold, "ap": ap, "f1": f1})
    print(f"Fold {fold}: AP={ap:.3f}  F1={f1:.3f}")

print(f"\nMean AP: {sum(s['ap'] for s in scores)/len(scores):.3f}")
```

### Model Save (Compatible with load_model())
```python
# Source: Phase 10 load_model() already uses pickle.load
import pickle
from pathlib import Path

Path("models").mkdir(exist_ok=True)

# Train on full train+val data
model_final = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": n_neg_all / n_pos_all})
model_final.fit(X_train_val, y_train_val)

# Verify interface requirements from load_model()
assert hasattr(model_final, "predict_proba"), "Must have predict_proba"
assert hasattr(model_final, "feature_names_in_"), "Must train with named DataFrame"

with open("models/xgb_btc_4h.pkl", "wb") as f:
    pickle.dump(model_final, f)

print(f"Saved: models/xgb_btc_4h.pkl")
print(f"Feature columns: {list(model_final.feature_names_in_)}")
```

### Sanity Check After Save
```python
# Verify the model works end-to-end with the backtest runner interface
with open("models/xgb_btc_4h.pkl", "rb") as f:
    loaded = pickle.load(f)

# Spot-check: single row prediction
sample = X_test.iloc[[0]]
proba = loaded.predict_proba(sample)[0][1]
print(f"Sample P(BUY) = {proba:.3f}")  # Should be a float in [0, 1]
print(f"Feature names match: {list(loaded.feature_names_in_) == list(X_test.columns)}")
```
</code_examples>

<sota_updates>
## State of the Art (2024-2025)

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| xgboost 1.x API | xgboost 3.x (installed: 3.0.4) | 2024 | `early_stopping_rounds` moved to constructor, not `fit()` |
| `eval_metric="auc"` | `eval_metric="aucpr"` preferred for imbalanced | 2023+ | AUC-PR more sensitive to precision/recall tradeoff |
| Manual class weights | `scale_pos_weight` param | Always standard | Cleaner than SMOTE for tree models |
| `model.save_model()` | Both native + pickle valid | — | Existing `load_model()` uses pickle — keep it |

**New in XGBoost 3.x (relevant):**
- `early_stopping_rounds` is now a constructor param, not in `.fit()` — code examples above reflect this
- `device="cuda"` for GPU training if available (not needed for 12 features)

**What's NOT worth doing for this project:**
- SHAP values for explainability (nice to have, not needed for hackathon)
- Ensemble methods (random forest + XGBoost) — overfitting risk, more complexity
- Deep learning (LSTM/Transformer) — needs far more data; XGBoost dominates on tabular
- Feature engineering beyond what Phase 4 already produces — risk of look-ahead

**Deprecated/outdated:**
- `xgb.train()` (low-level API) — use `XGBClassifier` sklearn API instead (matches existing interface)
- SMOTE for class imbalance — tree models handle via `scale_pos_weight`; SMOTE can introduce artifacts
</sota_updates>

<open_questions>
## Open Questions

1. **How many bars of data are available?**
   - What we know: Phase 9 downloaded BTC/ETH/SOL 4H candles. Binance public API typically provides 2+ years.
   - What's unclear: Exact date range of downloaded Parquet files.
   - Recommendation: Run `pd.read_parquet("data/BTCUSDT_4h.parquet").index[[0,-1]]` at script start to log available range and adjust train/val/test split accordingly.

2. **Optimal label threshold τ and horizon N**
   - What we know: τ=1.5%, N=6 bars (24H) are reasonable defaults for 4H swing trading.
   - What's unclear: Whether BTC's regime in the available data makes 1.5% too tight or too loose.
   - Recommendation: Check `y.value_counts(normalize=True)` — target 20–40% BUY rate. Adjust τ if BUY rate is < 15% or > 50%.

3. **Whether to include OHLCV as model features**
   - What we know: The existing `feature_cols` excludes raw OHLCV. `atr_proxy` and returns capture price info in normalised form.
   - What's unclear: Whether adding `log(close/close.shift(1))` as a raw feature would help.
   - Recommendation: Start with the 12 existing features (already shift-1 lagged). Don't add raw OHLCV.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- XGBoost 3.0 installed locally (3.0.4) — verified constructor signature for `early_stopping_rounds`
- `scripts/backtest.py` Phase 10 code — defines exact model interface (`predict_proba`, `feature_names_in_`, `pickle.load`)
- `bot/data/features.py` — defines all 12 feature columns available for training
- sklearn 1.6.1 docs — `TimeSeriesSplit(n_splits, gap)` parameter confirmed

### Secondary (MEDIUM confidence)
- Advances in Financial Machine Learning (Lopez de Prado) — forward-return and triple-barrier label patterns
- XGBoost docs (2024) — `scale_pos_weight`, `eval_metric="aucpr"` for imbalanced classification

### Tertiary (LOW confidence - needs validation)
- "crypto XGBoost swing trading" community patterns — τ=1.5%, N=6 defaults; validate with `y.value_counts()`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: XGBoost 3.0.4 binary classification
- Ecosystem: scikit-learn TimeSeriesSplit, pickle serialization
- Patterns: Forward-return labelling, walk-forward CV, train/val/test split
- Pitfalls: Look-ahead bias, gap misconfiguration, class imbalance, feature_names_in_

**Confidence breakdown:**
- Standard stack: HIGH — all packages installed and version-confirmed
- Architecture: HIGH — derived from existing Phase 10 interface constraints
- Label engineering: MEDIUM — defaults reasonable; optimal τ/N depends on available data range
- Pitfalls: HIGH — derived from code inspection of backtest.py + standard ML literature
- Code examples: HIGH — match XGBoost 3.0 API (early_stopping_rounds in constructor)

**Research date:** 2026-03-18
**Valid until:** 2026-06-18 (90 days — XGBoost API stable; sklearn stable)
</metadata>

---

*Phase: 11-xgboost-model-training*
*Research completed: 2026-03-18*
*Ready for planning: yes*

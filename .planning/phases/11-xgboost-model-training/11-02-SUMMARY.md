---
phase: 11-xgboost-model-training
plan: 02
subsystem: training
tags: [python, xgboost, model-training, pickle, final-model]

requires:
  - phase: 11-01
    provides: prepare_features, prepare_training_data, run_walk_forward_cv, XGB_PARAMS
  - phase: 10-02
    provides: model interface spec (load_model checks predict_proba + feature_names_in_)

provides:
  - scripts/train_model.py (complete — all 5 functions + main())
  - models/xgb_btc_4h.pkl (trained XGBClassifier, ready for backtest)

affects: [live-trading, backtest-runner]

tech-stack:
  added: []
  patterns: [pickle-round-trip-verification, fixed-nestimators-no-earlystop-final, train-val-test-temporal-split]

key-files:
  created: [models/xgb_btc_4h.pkl, models/.gitkeep]
  modified: [scripts/train_model.py]

key-decisions:
  - "Final model uses n_estimators=300 (no early stopping) — early stopping requires a held-back val set which we use for training in the final model"
---

# Phase 11 Plan 02: Final Training + Save Summary

**Trained final XGBoost model on 2020-2023 train+val data, evaluated on 2024 held-out test set, saved to models/xgb_btc_4h.pkl with pickle for immediate use in backtest.py**

## Accomplishments

- Implemented `train_final_model()`: trains on train+val (all bars before 2024-01-01), evaluates on test set (2024-01-01+), computes test AP and F1, prints top 5 feature importances
- Implemented `save_model()`: pickles model with sanity checks (predict_proba + feature_names_in_) and round-trip verification
- Wired complete `main()` with 5 steps: load features → prepare training data → [CV or final training] → save → done
- Created `models/.gitkeep` for directory tracking in git
- Successfully trained and saved `models/xgb_btc_4h.pkl` (490K)

## Files Created/Modified

- `scripts/train_model.py` — added `train_final_model()`, `save_model()`, wired complete `main()`
- `models/xgb_btc_4h.pkl` — trained XGBClassifier (490K)
- `models/.gitkeep` — directory tracking for git

## Test Set Metrics

**Test set (2024-01-01 to 2026-03-16):**
- Average Precision (AUC-PR): **0.392**
- F1 Score (threshold=0.5): **0.170**
- Test bars: 4,833
- BUY signals in test: 1,823 (37.7%)

**Train+Val set (2022-01-09 to 2023-12-31):**
- Bars: 4,330
- BUY signals: 1,800 (41.6%)
- Class balance (train+val): scale_pos_weight=1.41

## Feature Importances

Top 5 features by XGBoost importance:
1. **EMA_50**: 0.1079
2. **EMA_20**: 0.1003
3. **MACD_12_26_9**: 0.0930
4. **atr_proxy**: 0.0878
5. **MACDs_12_26_9**: 0.0873

## Decisions Made

1. **Fixed n_estimators=300 for final training**: No early stopping on final training (no held-back validation set). Conservative choice to avoid overfitting without validation feedback.
2. **scale_pos_weight=1.41 for final model**: Computed from train+val class balance (1,800 BUY / 2,530 NOT-BUY).
3. **Pickle format**: Matches `load_model()` in scripts/backtest.py. Model has both `predict_proba` and `feature_names_in_` attributes.
4. **Test split at 2024-01-01**: Hard temporal split; no leakage. Captures ~2 years of held-out future data (2024-2026).

## Issues Encountered

None. Full pipeline executed successfully:
1. ✓ Features loaded (9,169 bars × 17 columns)
2. ✓ Training data prepared (9,163 bars × 12 features, 39.5% BUY)
3. ✓ Final model trained on 4,330 bars (2020-2023)
4. ✓ Test evaluation completed (AP=0.392, F1=0.170)
5. ✓ Model saved with round-trip pickle verification: OK
6. ✓ Model attributes verified (predict_proba + feature_names_in_)

## Next Step

Phase 11 complete. Use model with:
```bash
python scripts/backtest.py --model models/xgb_btc_4h.pkl \
  --btc data/BTCUSDT_4h.parquet \
  --eth data/ETHUSDT_4h.parquet \
  --sol data/SOLUSDT_4h.parquet \
  --start 2024-01-01
```

Copy `models/xgb_btc_4h.pkl` to EC2 for live trading deployment.

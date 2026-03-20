#!/usr/bin/env python3
"""
Train XGBoost model using ONLY Oil + DXY macro features.

Usage:
    python scripts/train_oil_dxy_model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

# Configuration
DATA_DIR = Path("research_data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_oil_dxy.pkl"

# Oil + DXY features from alpha_research.py results
OIL_DXY_FEATURES = [
    "oil_return_1d",
    "oil_return_5d",
    "oil_vol_5d",
    "oil_acceleration",
    "dxy_return_1d",
    "dxy_return_5d",
    "dxy_vol_5d",
    "dxy_acceleration",
]

# XGBoost parameters (from alpha_research.py)
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    verbosity=1,
)

# Label configuration
LABEL_HORIZON = 6  # 6 bars ahead = 24 hours (4H bars)
LABEL_THRESHOLD = 0.01  # 1% price increase


def load_and_prepare_data():
    """Load BTC data and macro data, compute Oil + DXY features."""
    # Load BTC OHLCV
    btc_path = DATA_DIR / "BTCUSDT_4h.parquet"
    if not btc_path.exists():
        raise FileNotFoundError(f"BTC data not found at {btc_path}")
    btc = pd.read_parquet(btc_path)
    btc.columns = btc.columns.str.lower()

    # Load macro data
    oil_path = DATA_DIR / "oil_daily.parquet"
    dxy_path = DATA_DIR / "dxy_daily.parquet"

    if not oil_path.exists() or not dxy_path.exists():
        raise FileNotFoundError("Oil or DXY data not found")

    oil = pd.read_parquet(oil_path)
    dxy = pd.read_parquet(dxy_path)

    print(f"Loaded BTC: {len(btc)} bars")
    print(f"Loaded Oil: {len(oil)} daily bars")
    print(f"Loaded DXY: {len(dxy)} daily bars")

    # Compute macro features (matching alpha_research.py logic)
    feat = btc[["open", "high", "low", "close", "volume"]].copy()

    for name, mdf in [("oil", oil), ("dxy", dxy)]:
        close = mdf["close"].copy()
        # Resample daily to 4H by forward-filling
        close_4h = close.resample("4h").ffill()
        aligned = close_4h.reindex(feat.index, method="ffill")

        # Feature engineering (shifted to avoid lookahead bias)
        feat[f"{name}_return_1d"] = aligned.pct_change(6).shift(1)  # 6 bars = 1 day
        feat[f"{name}_return_5d"] = aligned.pct_change(30).shift(1)  # 30 bars = 5 days
        feat[f"{name}_vol_5d"] = aligned.pct_change().rolling(30).std().shift(1)
        r1d = aligned.pct_change(6)
        feat[f"{name}_acceleration"] = (r1d - r1d.shift(6)).shift(1)

    # Compute labels
    fwd_ret = feat["close"].shift(-LABEL_HORIZON) / feat["close"] - 1
    labels = (fwd_ret >= LABEL_THRESHOLD).astype(int)

    # Remove last LABEL_HORIZON bars (no labels)
    feat = feat.iloc[:-LABEL_HORIZON].copy()
    labels = labels.iloc[:-LABEL_HORIZON].copy()

    # Drop rows with missing features
    valid_idx = feat[OIL_DXY_FEATURES].dropna().index
    feat = feat.loc[valid_idx]
    labels = labels.loc[valid_idx]

    print(f"\nFeature matrix: {feat.shape}")
    print(f"Label distribution: {labels.mean():.1%} BUY")

    return feat, labels


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier with class balancing."""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"  Positive: {n_pos} ({y_train.mean():.1%})")
    print(f"  Negative: {n_neg}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": scale_pos_weight})

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # Evaluate on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, (y_pred_proba >= 0.5).astype(int), zero_division=0)

    print(f"\nValidation metrics:")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Feature importance
    print(f"\nFeature importance:")
    feat_imp = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(feat_imp.to_string(index=False))

    return model


def main():
    print("=" * 70)
    print("  TRAINING OIL + DXY XGBOOST MODEL")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    feat, labels = load_and_prepare_data()

    # Train/test split: 2022-2023 for training, 2024+ for testing
    print("\n[2/3] Splitting data...")
    train_cutoff = pd.Timestamp("2024-01-01", tz="UTC")

    train_mask = feat.index < train_cutoff
    X_train = feat.loc[train_mask, OIL_DXY_FEATURES]
    y_train = labels.loc[train_mask]

    X_val = feat.loc[~train_mask, OIL_DXY_FEATURES]
    y_val = labels.loc[~train_mask]

    print(f"Train period: {feat.loc[train_mask].index[0]} to {feat.loc[train_mask].index[-1]}")
    print(f"Test period:  {feat.loc[~train_mask].index[0]} to {feat.loc[~train_mask].index[-1]}")

    # Train model
    print("\n[3/3] Training model...")
    model = train_model(X_train, y_train, X_val, y_val)

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\n{'='*70}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Ready for backtesting with:")
    print(f"  python scripts/backtest.py --model {MODEL_PATH} --start 2024-01-01")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

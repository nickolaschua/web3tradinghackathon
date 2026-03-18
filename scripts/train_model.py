import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path so bot package can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_cross_asset_features, compute_features


FEATURE_COLS = [
    "atr_proxy", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    "eth_return_lag1", "eth_return_lag2", "sol_return_lag1", "sol_return_lag2",
]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    early_stopping_rounds=50,   # XGBoost 3.x: goes in constructor, NOT in .fit()
    random_state=42,
    n_jobs=-1,
)


def prepare_features(btc_path: str, eth_path: str, sol_path: str) -> pd.DataFrame:
    """
    Load and compute features for BTC, ETH, SOL.

    Reuses the same pipeline as backtest.py:
    1. Load all 3 parquets
    2. Normalize column names to lowercase
    3. compute_features(btc)
    4. compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    5. dropna()
    """
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)
    sol = pd.read_parquet(sol_path)

    for df in (btc, eth, sol):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    feat = compute_features(btc)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    feat = feat.dropna()

    return feat


def prepare_training_data(
    feat_df: pd.DataFrame, horizon: int = 6, threshold: float = 0.00015
):
    """
    Build aligned (X, y) with no look-ahead bias.

    feat_df: output of prepare_features() — has 'close' column (unshifted) + shifted indicators.
    horizon: forward bars for label (default 6 = 24H at 4H bars).
    threshold: minimum forward return to label as BUY (default 0.015% = 0.00015, adjusted for data scale).

    Labels: BUY=1 if close[t+horizon]/close[t] - 1 >= threshold, else 0.
    Features are already 1-bar lagged by compute_features() — no extra shift needed.
    """
    # Forward return label: close[t+horizon] / close[t] - 1
    fwd_ret = feat_df["close"].shift(-horizon) / feat_df["close"] - 1
    labels = (fwd_ret >= threshold).astype(int)

    # Exclude OHLCV from features; keep only indicator columns
    OHLCV = {"open", "high", "low", "close", "volume"}
    cols = [c for c in FEATURE_COLS if c in feat_df.columns]

    # Drop last `horizon` rows — no valid forward label
    X = feat_df[cols].iloc[:-horizon]
    y = labels.iloc[:-horizon]

    # Drop any remaining NaN rows (warmup period)
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    print(f"Training data: {len(X)} bars | Features: {X.shape[1]}")
    print(f"Date range: {X.index[0]} to {X.index[-1]}")
    n_buy = int(y.sum())
    n_total = len(y)
    print(f"Class balance: BUY={n_buy} ({n_buy/n_total:.1%}), NOT-BUY={n_total - n_buy} ({1 - n_buy/n_total:.1%})")

    return X, y


def run_walk_forward_cv(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """
    Walk-forward cross-validation with TimeSeriesSplit.

    Uses gap=24 bars (96H) to prevent label leakage from 6-bar forward horizon.
    Computes scale_pos_weight per fold (not global) for correct class weighting.
    Evaluates with AUC-PR (average precision) and F1 at 0.5 threshold.
    """
    tscv = TimeSeriesSplit(n_splits=5, gap=24)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        if n_pos == 0:
            print(f"Fold {fold}: SKIP — no positive labels in training fold")
            continue
        spw = n_neg / n_pos

        model = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": spw})
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        proba = model.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, proba)
        f1 = f1_score(y_va, (proba >= 0.5).astype(int), zero_division=0)
        best_iter = model.best_iteration if hasattr(model, "best_iteration") else "N/A"

        scores.append({"fold": fold, "ap": ap, "f1": f1, "best_iter": best_iter,
                       "n_train": len(X_tr), "n_val": len(X_va)})
        print(f"Fold {fold}: AP={ap:.3f}  F1={f1:.3f}  best_iter={best_iter}  "
              f"train={len(X_tr)}  val={len(X_va)}")

    if scores:
        mean_ap = sum(s["ap"] for s in scores) / len(scores)
        mean_f1 = sum(s["f1"] for s in scores) / len(scores)
        print(f"\nCV Summary: Mean AP={mean_ap:.3f}  Mean F1={mean_f1:.3f}  ({len(scores)} folds)")

    return scores


def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost BTC/USD 4H classifier")
    p.add_argument("--btc", default="data/BTCUSDT_4h.parquet")
    p.add_argument("--eth", default="data/ETHUSDT_4h.parquet")
    p.add_argument("--sol", default="data/SOLUSDT_4h.parquet")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--threshold", type=float, default=0.00015, help="Forward return threshold (default 0.00015 = 0.015%%)")
    p.add_argument("--cv-only", action="store_true", help="Run CV only, skip final training")
    p.add_argument("--output", default="models/xgb_btc_4h.pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feat = prepare_features(args.btc, args.eth, args.sol)
    X, y = prepare_training_data(feat, args.horizon, args.threshold)

    if args.cv_only:
        print("\nRunning walk-forward CV...")
        run_walk_forward_cv(X, y)
    else:
        print("Data prepared. Use --cv-only for CV, or run 11-02 plan for full training.")

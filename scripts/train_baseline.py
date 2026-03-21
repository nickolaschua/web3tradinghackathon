"""
Baseline model — 12 features, no BTC context or cross-sectional features.

This script is frozen. Do NOT modify it. It is the comparison anchor for all
future iterations. See research/iteration_log.md for context.

CV Mean AP: 0.530
Output: models/xgb_btc_4h_baseline.pkl

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --cv-only
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_cross_asset_features, compute_features


FEATURE_COLS = [
    # Baseline technical indicators
    "atr_proxy", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    # Cross-asset lagged returns (ETH/SOL leading BTC)
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
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)


def prepare_features(btc_path: str, eth_path: str, sol_path: str) -> pd.DataFrame:
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


def prepare_training_data(feat_df: pd.DataFrame, horizon: int = 6, threshold: float = 0.00015):
    fwd_ret = feat_df["close"].shift(-horizon) / feat_df["close"] - 1
    labels = (fwd_ret >= threshold).astype(int)

    cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    X = feat_df[cols].iloc[:-horizon]
    y = labels.iloc[:-horizon]

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
    tscv = TimeSeriesSplit(n_splits=5, gap=24)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        if n_pos == 0:
            print(f"Fold {fold}: SKIP -- no positive labels in training fold")
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


def train_final_model(X: pd.DataFrame, y: pd.Series, test_cutoff: str = "2024-01-01"):
    cutoff = pd.Timestamp(test_cutoff, tz="UTC")

    X_train_val = X[X.index < cutoff]
    y_train_val = y[y.index < cutoff]
    X_test = X[X.index >= cutoff]
    y_test = y[y.index >= cutoff]

    print(f"Train+Val: {len(X_train_val)} bars ({X_train_val.index[0].date()} to {X_train_val.index[-1].date()})")
    print(f"Test:      {len(X_test)} bars ({X_test.index[0].date()} to {X_test.index[-1].date()})")

    n_pos = int(y_train_val.sum())
    n_neg = int(len(y_train_val) - n_pos)
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Train+Val class balance: BUY={n_pos} ({n_pos/len(y_train_val):.1%}), scale_pos_weight={spw:.2f}")

    final_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = 300
    final_params["scale_pos_weight"] = spw

    print("\nTraining final model (n_estimators=300, no early stopping)...")
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train_val, y_train_val, verbose=False)

    metrics = {}
    if len(X_test) > 0 and y_test.sum() > 0:
        proba_test = model.predict_proba(X_test)[:, 1]
        ap_test = average_precision_score(y_test, proba_test)
        f1_test = f1_score(y_test, (proba_test >= 0.5).astype(int), zero_division=0)
        n_buy_test = int(y_test.sum())
        metrics = {"test_ap": ap_test, "test_f1": f1_test,
                   "n_test": len(X_test), "n_buy_test": n_buy_test}
        print(f"\nTest set evaluation:")
        print(f"  AP (AUC-PR): {ap_test:.3f}")
        print(f"  F1 (thresh=0.5): {f1_test:.3f}")
        print(f"  Test bars: {len(X_test)} | BUY signals: {n_buy_test} ({n_buy_test/len(X_test):.1%})")

    importances = pd.Series(
        model.feature_importances_, index=model.feature_names_in_
    ).sort_values(ascending=False)
    print(f"\nTop 5 feature importances:")
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    return model, metrics


def save_model(model: xgb.XGBClassifier, output_path: str) -> None:
    assert hasattr(model, "predict_proba"), "Model must have predict_proba"
    assert hasattr(model, "feature_names_in_"), (
        "Model has no feature_names_in_ -- did you train with a named DataFrame?"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nSaved: {output_path}")
    print(f"Feature columns ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")

    with open(output_path, "rb") as f:
        reloaded = pickle.load(f)
    assert list(reloaded.feature_names_in_) == list(model.feature_names_in_)
    print("Round-trip pickle verification: OK")


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline XGBoost model (12 features)")
    p.add_argument("--btc", default="data/BTCUSDT_4h.parquet")
    p.add_argument("--eth", default="data/ETHUSDT_4h.parquet")
    p.add_argument("--sol", default="data/SOLUSDT_4h.parquet")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--threshold", type=float, default=0.00015)
    p.add_argument("--cv-only", action="store_true")
    p.add_argument("--output", default="models/xgb_btc_4h_baseline.pkl")
    return p.parse_args()


def main():
    args = parse_args()

    print("=== Baseline model (12 features) ===")
    print("Step 1: Loading features...")
    feat = prepare_features(args.btc, args.eth, args.sol)
    print(f"  Feature matrix: {feat.shape[0]} bars x {feat.shape[1]} columns")

    print("\nStep 2: Preparing training data...")
    X, y = prepare_training_data(feat, args.horizon, args.threshold)

    if args.cv_only:
        print("\nStep 3: Running walk-forward CV...")
        run_walk_forward_cv(X, y)
        return

    print("\nStep 3: Training final model...")
    model, metrics = train_final_model(X, y, test_cutoff="2024-01-01")

    print("\nStep 4: Saving model...")
    save_model(model, args.output)
    print(f"\nDone. Model: {args.output}")


if __name__ == "__main__":
    main()

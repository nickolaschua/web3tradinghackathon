#!/usr/bin/env python3
"""
15-minute XGBoost control baseline — training script.

This is a BACKTEST-ONLY stub. No live trading, no Roostoo API calls.
Purpose: control/comparison baseline for research strategies.

Usage:
  python scripts/train_model_15m.py                   # full pipeline
  python scripts/train_model_15m.py --cv-only         # CV only, no save
  python scripts/train_model_15m.py --threshold 0.001 # custom return threshold

Downloads required:
  python scripts/download_data.py --interval 15m --symbols BTC ETH SOL
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

# Add project root to path so bot package can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_btc_context_features, compute_cross_asset_features, compute_features


# ── Constants ──────────────────────────────────────────────────────────────────

# 15M resolution: 16 bars = 4H equivalent forward horizon (4 bars/hour × 4H)
HORIZON_15M = 16

# Annualisation: 365.25 × 24 × 4 = 35,064  (rounded to 35,040 for clean mult.)
PERIODS_15M = 35040

# Walk-forward: gap = 4 × horizon (leakage guard proportional to forward window)
CV_GAP = 64        # 64 bars = 16 hours — safely past the 16-bar label horizon
CV_SPLITS = 8      # more folds than 4H baseline thanks to 16× more data

# Feature columns for the improved 15M baseline.
# Cross-asset lags use 16-bar (4H) and 96-bar (1D) timescales — at 15M resolution
# a 1-bar lag is only 15 minutes (noise), but a 16-bar lag captures meaningful
# inter-market spillover at the same timescale as our label horizon.
FEATURE_COLS = [
    # Core trend / momentum
    "atr_proxy", "RSI_14", "RSI_7",
    "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    # Volatility regime and price structure
    "bb_width", "bb_pos",
    # Volume and candle conviction
    "volume_ratio", "candle_body",
    # Cross-asset lags at meaningful timescales (4H and 1D)
    "eth_return_4h", "sol_return_4h",
    "eth_return_1d", "sol_return_1d",
    # BTC/ETH rolling correlation and beta (window=2880 bars = 30 days at 15M)
    # Proven at 4H (IC=0.0747, 0.0648 respectively). sol_btc_* rejected (CV regress).
    "eth_btc_corr", "eth_btc_beta",
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


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_features(btc_path: str, eth_path: str, sol_path: str) -> pd.DataFrame:
    """
    Load 15M parquets, run feature pipeline, and return cleaned feature matrix.

    Pipeline (must match backtest_15m.py for consistency):
    1. Load BTCUSDT_15m, ETHUSDT_15m, SOLUSDT_15m parquets
    2. compute_features(btc)
    3. compute_cross_asset_features(feat, {eth, sol})  — adds lag1/lag2 cols
    4. Add meaningful cross-asset lags: 16-bar (4H) and 96-bar (1D)
    5. dropna()

    Note: lag1/lag2 from compute_cross_asset_features are 15-minute lags
    (signal = noise at this resolution). We compute 16-bar (4H) and 96-bar (1D)
    lags here to capture inter-market spillover at meaningful timescales.
    """
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)
    sol = pd.read_parquet(sol_path)

    for df in (btc, eth, sol):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    feat = compute_features(btc)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})

    # Add 4H (16-bar) and 1D (96-bar) cross-asset log-return lags.
    # These capture meaningful inter-market momentum at timescales that matter.
    for asset, df in [("eth", eth), ("sol", sol)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    # BTC/ETH rolling correlation + beta (30-day window at 15M = 2880 bars).
    # Must be called BEFORE dropna() to preserve row alignment.
    feat = compute_btc_context_features(feat, eth, sol, window=2880)

    feat = feat.dropna()

    return feat


# ── Training data ──────────────────────────────────────────────────────────────

def compute_triple_barrier_labels(
    feat_df: pd.DataFrame,
    horizon: int = HORIZON_15M,
    tp_pct: float = 0.005,
    sl_pct: float = 0.003,
) -> pd.Series:
    """
    Triple-barrier labeling (Lopez de Prado, "Advances in Financial ML").

    For each bar N, scan forward up to `horizon` bars:
    - BUY (1)     if close hits take-profit (+tp_pct) BEFORE stop-loss (-sl_pct)
    - NOT-BUY (0) if stop-loss hit first OR time barrier reached without TP

    Why this beats simple return threshold:
    - Conditions on the PATH, not just the endpoint: a bar that rallies +0.8%
      then crashes -1% is correctly labeled NOT-BUY, not BUY.
    - Encodes asymmetric R:R into the label itself (TP > SL → only label moves
      where the upside was meaningfully larger than the risk taken).
    - Naturally lowers BUY rate to ~15-20% without arbitrary threshold tuning.

    Uses close prices. Last `horizon` rows are labeled 0 (no future data).
    """
    closes = feat_df["close"].values
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        entry = closes[i]
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        for j in range(i + 1, i + horizon + 1):
            if closes[j] >= tp:
                labels[i] = 1   # take-profit hit first → BUY
                break
            elif closes[j] <= sl:
                break           # stop-loss hit first → NOT-BUY (stays 0)
        # time barrier (neither hit): stays 0

    return pd.Series(labels, index=feat_df.index)


def prepare_training_data(
    feat_df: pd.DataFrame,
    horizon: int = HORIZON_15M,
    tp_pct: float = 0.005,
    sl_pct: float = 0.003,
):
    """
    Build aligned (X, y) using triple-barrier labels.

    horizon: 16 bars = 4H equivalent at 15M.
    tp_pct:  take-profit level (default 0.5%) — BUY if price hits this first.
    sl_pct:  stop-loss level  (default 0.3%) — NOT-BUY if price hits this first.
             Asymmetric (TP > SL) encodes 1.67:1 minimum R:R into the labels.
    """
    labels = compute_triple_barrier_labels(feat_df, horizon, tp_pct, sl_pct)

    cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    missing = [c for c in FEATURE_COLS if c not in feat_df.columns]
    if missing:
        print(f"WARNING: FEATURE_COLS missing from data: {missing}")

    X = feat_df[cols].iloc[:-horizon]
    y = labels.iloc[:-horizon]

    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    print(f"Training data: {len(X):,} bars | Features: {X.shape[1]}")
    print(f"Date range: {X.index[0]} to {X.index[-1]}")
    n_buy = int(y.sum())
    n_total = len(y)
    print(f"Class balance: BUY={n_buy:,} ({n_buy/n_total:.1%}), NOT-BUY={n_total - n_buy:,} ({1 - n_buy/n_total:.1%})")

    return X, y


# ── Walk-forward CV ────────────────────────────────────────────────────────────

def run_walk_forward_cv(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """
    Walk-forward CV with 8 splits and 64-bar gap (leakage guard for 16-bar horizon).
    """
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)
    scores = []

    print(f"Walk-forward CV: {CV_SPLITS} splits, gap={CV_GAP} bars ({CV_GAP * 15 // 60}H)")

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
              f"train={len(X_tr):,}  val={len(X_va):,}")

    if scores:
        mean_ap = sum(s["ap"] for s in scores) / len(scores)
        mean_f1 = sum(s["f1"] for s in scores) / len(scores)
        print(f"\nCV Summary: Mean AP={mean_ap:.3f}  Mean F1={mean_f1:.3f}  ({len(scores)} folds)")

    return scores


# ── Final model ────────────────────────────────────────────────────────────────

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_cutoff: str = "2024-01-01",
    n_estimators: int = 300,
):
    """
    Train on all data before test_cutoff, evaluate on held-out test set.
    No early stopping for final training — fixed n_estimators=300.
    """
    cutoff = pd.Timestamp(test_cutoff, tz="UTC")

    X_train_val = X[X.index < cutoff]
    y_train_val = y[y.index < cutoff]
    X_test = X[X.index >= cutoff]
    y_test = y[y.index >= cutoff]

    print(f"Train+Val: {len(X_train_val):,} bars ({X_train_val.index[0].date()} to {X_train_val.index[-1].date()})")
    print(f"Test:      {len(X_test):,} bars ({X_test.index[0].date()} to {X_test.index[-1].date()})")

    if len(X_test) == 0:
        print("WARNING: No test data available after cutoff. Adjust --test-cutoff.")

    n_pos = int(y_train_val.sum())
    n_neg = int(len(y_train_val) - n_pos)
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Train+Val class balance: BUY={n_pos:,} ({n_pos/len(y_train_val):.1%}), scale_pos_weight={spw:.2f}")

    final_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = n_estimators
    final_params["scale_pos_weight"] = spw

    print(f"\nTraining final model (n_estimators={n_estimators}, no early stopping)...")
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
        print(f"  Test bars: {len(X_test):,} | BUY signals: {n_buy_test} ({n_buy_test/len(X_test):.1%})")
    else:
        print("WARNING: Skipping test evaluation — no test data or no BUY labels in test set")

    importances = pd.Series(
        model.feature_importances_, index=model.feature_names_in_
    ).sort_values(ascending=False)
    print(f"\nTop 5 feature importances:")
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    return model, metrics


# ── Save / load ────────────────────────────────────────────────────────────────

def save_model(model: xgb.XGBClassifier, output_path: str) -> None:
    """Save model with pickle (matches load_model() in scripts/backtest_15m.py)."""
    assert hasattr(model, "predict_proba"), "Model must have predict_proba"
    assert hasattr(model, "feature_names_in_"), (
        "Model has no feature_names_in_ — did you train with a named DataFrame?"
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train 15M XGBoost BTC/USD control baseline (backtest-only)"
    )
    p.add_argument("--btc", default="data/BTCUSDT_15m.parquet")
    p.add_argument("--eth", default="data/ETHUSDT_15m.parquet")
    p.add_argument("--sol", default="data/SOLUSDT_15m.parquet")
    p.add_argument("--horizon", type=int, default=HORIZON_15M,
                   help=f"Forward bars for label (default {HORIZON_15M} = 4H at 15M)")
    p.add_argument("--tp-pct", type=float, default=0.008,
                   help="Take-profit %% for triple-barrier label (default 0.008 = 0.8%%)")
    p.add_argument("--sl-pct", type=float, default=0.003,
                   help="Stop-loss %% for triple-barrier label (default 0.003 = 0.3%%)")
    p.add_argument("--test-cutoff", default="2024-01-01",
                   help="Train/test split date (default 2024-01-01)")
    p.add_argument("--n-estimators", type=int, default=300,
                   help="n_estimators for final model (default 300, no early stopping)")
    p.add_argument("--cv-only", action="store_true",
                   help="Run walk-forward CV only, skip final training")
    p.add_argument("--output", default="models/xgb_btc_15m.pkl")
    return p.parse_args()


def main():
    args = parse_args()

    print("Step 1: Loading features...")
    feat = prepare_features(args.btc, args.eth, args.sol)
    print(f"  Feature matrix: {feat.shape[0]:,} bars x {feat.shape[1]} columns")
    print(f"  Date range: {feat.index[0]} to {feat.index[-1]}")

    print(f"\nStep 2: Preparing training data (triple-barrier: TP={args.tp_pct:.1%}, SL={args.sl_pct:.1%})...")
    X, y = prepare_training_data(feat, args.horizon, args.tp_pct, args.sl_pct)

    print("\nStep 3: Walk-forward CV...")
    run_walk_forward_cv(X, y)

    if args.cv_only:
        print("\nCV complete. Re-run without --cv-only to train and save the final model.")
        return

    print("\nStep 4: Training final model...")
    model, metrics = train_final_model(X, y, args.test_cutoff, args.n_estimators)

    print("\nStep 5: Saving model...")
    save_model(model, args.output)
    print(f"\nDone. Run backtest with: python scripts/backtest_15m.py --model {args.output}")


if __name__ == "__main__":
    main()

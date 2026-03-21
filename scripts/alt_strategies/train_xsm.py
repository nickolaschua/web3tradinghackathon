#!/usr/bin/env python3
"""
Train XGBoost cross-sectional momentum (XSM) model.

The model predicts: "Will this coin be among the top-K performers over the
next N bars AND have a positive absolute return?"

This is strictly better than simple return ranking because XGBoost learns
multi-factor selection rules from 20 features: multi-horizon momentum,
cross-sectional ranks, BTC beta/correlation, and volume/volatility regime.

Usage:
    python scripts/alt_strategies/train_xsm.py
    python scripts/alt_strategies/train_xsm.py --top-k 5 --horizon 96
    python scripts/alt_strategies/train_xsm.py --cv-only
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_features
from scripts.alt_strategies.strategies import XSM_FEATURE_COLS, build_xsm_panel

DATA_DIR = project_root / "data"
MODEL_DIR = project_root / "models"
RESULTS_DIR = project_root / "research_results"

_BINANCE_TO_ROOSTOO: dict[str, str] = {
    "BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD", "BNBUSDT": "BNB/USD",
    "SOLUSDT": "SOL/USD", "ADAUSDT": "ADA/USD", "AVAXUSDT": "AVAX/USD",
    "DOGEUSDT": "DOGE/USD", "LINKUSDT": "LINK/USD", "DOTUSDT": "DOT/USD",
    "UNIUSDT": "UNI/USD", "XRPUSDT": "XRP/USD", "LTCUSDT": "LTC/USD",
    "AAVEUSDT": "AAVE/USD", "CRVUSDT": "CRV/USD", "NEARUSDT": "NEAR/USD",
    "FILUSDT": "FIL/USD", "FETUSDT": "FET/USD", "HBARUSDT": "HBAR/USD",
    "ZECUSDT": "ZEC/USD", "ZENUSDT": "ZEN/USD", "CAKEUSDT": "CAKE/USD",
    "PAXGUSDT": "PAXG/USD", "XLMUSDT": "XLM/USD", "TRXUSDT": "TRX/USD",
    "CFXUSDT": "CFX/USD", "SHIBUSDT": "SHIB/USD", "ICPUSDT": "ICP/USD",
    "APTUSDT": "APT/USD", "ARBUSDT": "ARB/USD", "SUIUSDT": "SUI/USD",
    "FLOKIUSDT": "FLOKI/USD", "PEPEUSDT": "PEPE/USD",
    "PENDLEUSDT": "PENDLE/USD", "WLDUSDT": "WLD/USD", "SEIUSDT": "SEI/USD",
    "BONKUSDT": "BONK/USD", "WIFUSDT": "WIF/USD", "ENAUSDT": "ENA/USD",
    "TAOUSDT": "TAO/USD",
}

TRAIN_CUTOFF = "2024-01-01"


def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all 15M parquets and return {pair: raw_ohlcv_df}."""
    data = {}
    for path in sorted(DATA_DIR.glob("*_15m.parquet")):
        sym = path.stem.replace("_15m", "")
        pair = _BINANCE_TO_ROOSTOO.get(sym)
        if pair is None:
            continue
        df = pd.read_parquet(path)
        df.columns = df.columns.str.lower()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        data[pair] = df
    print(f"Loaded {len(data)} coins")
    return data


def compute_all_features(raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute technical features for each coin."""
    features = {}
    for pair, df in raw_data.items():
        feat = compute_features(df)
        feat = feat.dropna(subset=["RSI_14", "atr_proxy", "EMA_20", "EMA_50"])
        if len(feat) > 500:
            features[pair] = feat
    print(f"Computed features for {len(features)} coins")
    return features


def compute_labels(
    panel: pd.DataFrame,
    all_features: dict[str, pd.DataFrame],
    horizon: int = 96,
    top_k: int = 5,
    min_return: float = 0.002,
) -> pd.Series:
    """
    Label: 1 if coin is in top-K forward returners AND return > min_return.

    Vectorized computation using forward returns in the panel.
    """
    print("  Computing forward returns per coin...")
    fwd_series_list = []
    for pair, feat_df in all_features.items():
        close = feat_df["close"]
        fwd = (close.shift(-horizon) / close) - 1.0
        fwd = fwd.to_frame("fwd_return")
        fwd["pair"] = pair
        fwd_series_list.append(fwd)

    fwd_all = pd.concat(fwd_series_list)
    fwd_all = fwd_all.set_index("pair", append=True)

    # Align to panel index
    panel_fwd = fwd_all.reindex(panel.index)["fwd_return"]

    # Rank within each timestamp (vectorized)
    print("  Ranking forward returns cross-sectionally...")
    rank = panel_fwd.groupby(level=0).rank(ascending=False, method="first")
    count = panel_fwd.groupby(level=0).transform("count")

    # Label: top-K rank AND positive return AND enough coins
    labels = (
        (rank <= top_k) &
        (panel_fwd > min_return) &
        (count >= top_k)
    ).astype(np.int8)

    # NaN forward returns → label 0
    labels = labels.fillna(0).astype(np.int8)

    n_labeled = int(labels.sum())
    pos_rate = n_labeled / max(len(labels), 1) * 100
    print(f"  Labels: {n_labeled:,} positive out of {len(labels):,} ({pos_rate:.1f}%)")
    return labels


def train_xsm_model(
    panel: pd.DataFrame,
    labels: pd.Series,
    train_cutoff: str = TRAIN_CUTOFF,
    cv_only: bool = False,
) -> xgb.XGBClassifier | None:
    """Train XGBoost cross-sectional momentum model with walk-forward CV."""
    cutoff_ts = pd.Timestamp(train_cutoff, tz="UTC")

    # Filter to training period
    train_mask = panel.index.get_level_values(0) < cutoff_ts
    X_train_full = panel.loc[train_mask, XSM_FEATURE_COLS].copy()
    y_train_full = labels.loc[train_mask].copy()

    # Drop rows with all NaN features
    valid_mask = X_train_full.notna().sum(axis=1) >= 10
    X_train_full = X_train_full.loc[valid_mask]
    y_train_full = y_train_full.loc[valid_mask]

    print(f"\nTraining set: {len(X_train_full):,} rows, "
          f"{y_train_full.sum():,} positive ({y_train_full.mean()*100:.1f}%)")
    print(f"Features: {len(XSM_FEATURE_COLS)}")

    # Class balance
    pos_count = y_train_full.sum()
    neg_count = len(y_train_full) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    xgb_params = dict(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    # Walk-forward CV
    print("\n── Walk-Forward CV ──")
    # Use unique timestamps for splitting
    unique_ts = X_train_full.index.get_level_values(0).unique().sort_values()
    n_ts = len(unique_ts)
    n_splits = 5
    gap = 96  # 1 day gap

    fold_metrics = []
    for fold in range(n_splits):
        train_end_idx = int(n_ts * (0.5 + 0.1 * fold))
        val_start_idx = train_end_idx + gap
        val_end_idx = min(val_start_idx + int(n_ts * 0.1), n_ts)

        if val_start_idx >= n_ts or val_end_idx <= val_start_idx:
            continue

        train_ts = unique_ts[:train_end_idx]
        val_ts = unique_ts[val_start_idx:val_end_idx]

        fold_train_mask = X_train_full.index.get_level_values(0).isin(train_ts)
        fold_val_mask = X_train_full.index.get_level_values(0).isin(val_ts)

        X_tr = X_train_full.loc[fold_train_mask]
        y_tr = y_train_full.loc[fold_train_mask]
        X_va = X_train_full.loc[fold_val_mask]
        y_va = y_train_full.loc[fold_val_mask]

        if len(X_tr) < 1000 or len(X_va) < 100 or y_va.sum() < 10:
            continue

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        probas = model.predict_proba(X_va)[:, 1]
        preds = (probas > 0.5).astype(int)

        auc = roc_auc_score(y_va, probas) if y_va.nunique() > 1 else 0
        ap = average_precision_score(y_va, probas) if y_va.nunique() > 1 else 0
        f1 = f1_score(y_va, preds, zero_division=0)

        fold_metrics.append({"fold": fold, "auc": auc, "ap": ap, "f1": f1,
                             "train_rows": len(X_tr), "val_rows": len(X_va)})
        print(f"  Fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  "
              f"train={len(X_tr):,}  val={len(X_va):,}")

    if fold_metrics:
        mean_auc = np.mean([m["auc"] for m in fold_metrics])
        mean_ap = np.mean([m["ap"] for m in fold_metrics])
        mean_f1 = np.mean([m["f1"] for m in fold_metrics])
        print(f"\n  Mean: AUC={mean_auc:.4f}  AP={mean_ap:.4f}  F1={mean_f1:.4f}")

    if cv_only:
        return None

    # Final model on full training set
    print("\n── Training Final Model ──")
    model = xgb.XGBClassifier(**xgb_params)

    # Split last 10% as eval set for early stopping
    split_idx = int(len(X_train_full) * 0.9)
    X_tr = X_train_full.iloc[:split_idx]
    y_tr = y_train_full.iloc[:split_idx]
    X_va = X_train_full.iloc[split_idx:]
    y_va = y_train_full.iloc[split_idx:]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    # Feature importance
    importances = model.feature_importances_
    fi = sorted(zip(XSM_FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    print("\n  Feature importance (top 10):")
    for name, imp in fi[:10]:
        print(f"    {name:25s}: {imp:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost cross-sectional momentum model")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K coins to label as positive")
    parser.add_argument("--horizon", type=int, default=96, help="Forward return horizon in bars (96=1day)")
    parser.add_argument("--min-return", type=float, default=0.002, help="Min forward return for positive label")
    parser.add_argument("--cv-only", action="store_true", help="Only run CV, don't save model")
    parser.add_argument("--output", type=str, default=None, help="Custom model output path")
    args = parser.parse_args()

    t0 = time.time()

    # 1. Load data
    raw_data = load_all_data()
    btc_close = raw_data["BTC/USD"]["close"].copy() if "BTC/USD" in raw_data else None

    # 2. Compute features
    all_features = compute_all_features(raw_data)

    # 3. Build panel
    print("\nBuilding cross-sectional panel...")
    panel = build_xsm_panel(all_features, btc_close)
    print(f"Panel shape: {panel.shape}")

    # 4. Compute labels
    print("\nComputing labels...")
    labels = compute_labels(
        panel, all_features,
        horizon=args.horizon,
        top_k=args.top_k,
        min_return=args.min_return,
    )

    # 5. Train
    model = train_xsm_model(panel, labels, cv_only=args.cv_only)

    if model is not None:
        MODEL_DIR.mkdir(exist_ok=True)
        out_path = Path(args.output) if args.output else MODEL_DIR / f"xgb_xsm_top{args.top_k}_h{args.horizon}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()

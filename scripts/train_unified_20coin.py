#!/usr/bin/env python3
"""
Unified 20-coin XGBoost training script.

Trains a single XGBoost classifier on pooled 15M data from 20 liquid coins.
Uses ATR-normalized triple-barrier labels so each coin gets ~15-20% BUY rate
regardless of volatility.

Usage:
  python scripts/train_unified_20coin.py                    # full pipeline
  python scripts/train_unified_20coin.py --cv-only          # CV only, no save
  python scripts/train_unified_20coin.py --tp-mult 2.5 --sl-mult 0.8  # custom barriers

Spec: docs/superpowers/specs/2026-03-21-unified-20coin-xgboost-design.md
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import (
    compute_features,
    compute_market_context_features,
    compute_coin_identity_features,
)

# ── Constants ─────────────────────────────────────────────────────────────────

HORIZON_15M = 16       # 4H forward horizon at 15M resolution
PERIODS_15M = 35_040   # annualisation: 365.25 x 24 x 4
CV_GAP = 64            # 16 hours leakage guard
CV_SPLITS = 8

# 20-coin universe (verified Roostoo-tradeable + data available)
COIN_UNIVERSE = [
    "BTC", "ETH", "BNB",                                          # tier 1 (mega)
    "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC",   # tier 2 (large)
    "UNI", "NEAR", "SUI", "APT", "PEPE", "ARB", "SHIB", "FIL", "HBAR",  # tier 3 (mid)
]

LIQUIDITY_TIERS = {
    "BTC": 1, "ETH": 1, "BNB": 1,
    "SOL": 2, "XRP": 2, "DOGE": 2, "ADA": 2, "AVAX": 2,
    "LINK": 2, "DOT": 2, "LTC": 2,
    "UNI": 3, "NEAR": 3, "SUI": 3, "APT": 3, "PEPE": 3,
    "ARB": 3, "SHIB": 3, "FIL": 3, "HBAR": 3,
}

# 21 features (spec Section 2)
FEATURE_COLS = [
    # Per-coin technicals (13)
    "atr_proxy", "RSI_14", "RSI_7",
    "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    "bb_width", "bb_pos", "volume_ratio", "candle_body",
    # Market leader context (4)
    "btc_return_4h", "btc_return_1d", "eth_return_4h", "eth_return_1d",
    # Coin-to-market identity (4)
    "btc_corr_30d", "relative_vol", "vol_rank", "liquidity_tier",
]

XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)


# ── ATR-normalized triple-barrier labels ──────────────────────────────────────

def compute_atr_normalized_labels(
    df: pd.DataFrame,
    horizon: int = HORIZON_15M,
    tp_mult: float = 2.5,
    sl_mult: float = 0.8,
) -> pd.Series:
    """
    Triple-barrier labeling with ATR-normalized TP/SL per bar.

    TP = atr_proxy * tp_mult (high upside skew for Sortino)
    SL = atr_proxy * sl_mult (tight downside control)

    BUY (1) if price hits entry + TP before entry - SL within horizon.
    HOLD (0) otherwise (SL hit first, or time barrier reached).
    """
    closes = df["close"].values
    atrs = df["atr_proxy"].values
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        entry = closes[i]
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue  # no valid ATR — label stays 0
        tp = entry + atr * tp_mult
        sl = entry - atr * sl_mult
        for j in range(i + 1, i + horizon + 1):
            if closes[j] >= tp:
                labels[i] = 1
                break
            elif closes[j] <= sl:
                break
    return pd.Series(labels, index=df.index)


# ── Feature preparation ──────────────────────────────────────────────────────

def prepare_all_coins(
    data_dir: str = "data",
    tp_mult: float = 2.5,
    sl_mult: float = 0.8,
) -> pd.DataFrame:
    """
    Load all 20 coins, compute features + labels, return pooled DataFrame.
    """
    data_path = Path(data_dir)

    # Load BTC and ETH first (needed for market context)
    btc_raw = pd.read_parquet(data_path / "BTCUSDT_15m.parquet")
    eth_raw = pd.read_parquet(data_path / "ETHUSDT_15m.parquet")
    for df in (btc_raw, eth_raw):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    # Pre-compute all coins' atr_proxy for vol_rank (cross-sectional)
    print("Pre-computing ATR proxies for vol_rank...")
    all_atr_proxies = {}
    for coin in COIN_UNIVERSE:
        coin_df = pd.read_parquet(data_path / f"{coin}USDT_15m.parquet")
        coin_df.index = pd.to_datetime(coin_df.index)
        coin_df.columns = coin_df.columns.str.lower()
        lr = np.log(coin_df["close"] / coin_df["close"].shift(1))
        all_atr_proxies[coin] = lr.rolling(14).std() * coin_df["close"] * 1.25

    # Process each coin
    all_dfs = []
    for coin in COIN_UNIVERSE:
        t0 = time.time()
        coin_path = data_path / f"{coin}USDT_15m.parquet"
        coin_df = pd.read_parquet(coin_path)
        coin_df.index = pd.to_datetime(coin_df.index)
        coin_df.columns = coin_df.columns.str.lower()

        # Per-coin technicals (13 features, shifted 1 bar)
        feat = compute_features(coin_df)

        # Market context (4 features: btc/eth returns at 4H and 1D lags)
        feat = compute_market_context_features(feat, btc_raw, eth_raw)

        # Coin identity (4 features: btc_corr_30d, relative_vol, vol_rank, liquidity_tier)
        feat = compute_coin_identity_features(
            feat, btc_raw,
            liquidity_tier=LIQUIDITY_TIERS[coin],
            all_atr_proxies={**all_atr_proxies, "self": all_atr_proxies[coin]},
        )

        # ATR-normalized labels
        labels = compute_atr_normalized_labels(feat, HORIZON_15M, tp_mult, sl_mult)
        feat["label"] = labels

        # Tag coin for diagnostics (not a model feature)
        feat["_coin"] = coin

        all_dfs.append(feat)
        elapsed = time.time() - t0
        buy_rate = labels.sum() / max(len(labels), 1)
        print(f"  {coin:>5}: {len(feat):>7,} bars | BUY rate: {buy_rate:.1%} | {elapsed:.1f}s")

    # Pool and clean
    pooled = pd.concat(all_dfs)
    pooled = pooled.sort_index()  # temporal order for TimeSeriesSplit
    pooled = pooled.dropna(subset=FEATURE_COLS)

    print(f"\nPooled: {len(pooled):,} bars | {pooled['_coin'].nunique()} coins")
    print(f"Date range: {pooled.index[0]} to {pooled.index[-1]}")
    buy_rate = pooled["label"].mean()
    print(f"Overall BUY rate: {buy_rate:.1%}")

    # Per-coin BUY rate check
    print("\nPer-coin BUY rates:")
    for coin in COIN_UNIVERSE:
        mask = pooled["_coin"] == coin
        if mask.sum() > 0:
            rate = pooled.loc[mask, "label"].mean()
            print(f"  {coin:>5}: {rate:.1%}  ({mask.sum():,} bars)")

    return pooled


def prepare_training_data(
    pooled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X, y from pooled DataFrame. Drops last HORIZON bars per coin."""
    trimmed = pooled.groupby("_coin").apply(
        lambda g: g.iloc[:-HORIZON_15M], include_groups=False
    ).droplevel(0)

    cols = [c for c in FEATURE_COLS if c in trimmed.columns]
    X = trimmed[cols]
    y = trimmed["label"]

    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    print(f"Training data: {len(X):,} bars | Features: {X.shape[1]}")
    n_buy = int(y.sum())
    print(f"Class balance: BUY={n_buy:,} ({n_buy/len(y):.1%}), "
          f"NOT-BUY={len(y)-n_buy:,} ({1-n_buy/len(y):.1%})")

    return X, y


# ── Walk-forward CV ───────────────────────────────────────────────────────────

def run_walk_forward_cv(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """Walk-forward CV with 8 splits and 64-bar gap."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)
    scores = []

    print(f"Walk-forward CV: {CV_SPLITS} splits, gap={CV_GAP} bars ({CV_GAP * 15 // 60}H)")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        n_pos = int(y_tr.sum())
        if n_pos == 0:
            print(f"Fold {fold}: SKIP — no positive labels")
            continue
        spw = (len(y_tr) - n_pos) / n_pos

        model = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": spw})
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        proba = model.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, proba)
        f1 = f1_score(y_va, (proba >= 0.5).astype(int), zero_division=0)
        best_iter = getattr(model, "best_iteration", "N/A")

        scores.append({"fold": fold, "ap": ap, "f1": f1, "best_iter": best_iter,
                       "n_train": len(X_tr), "n_val": len(X_va)})
        print(f"Fold {fold}: AP={ap:.3f}  F1={f1:.3f}  best_iter={best_iter}  "
              f"train={len(X_tr):,}  val={len(X_va):,}")

    if scores:
        mean_ap = sum(s["ap"] for s in scores) / len(scores)
        mean_f1 = sum(s["f1"] for s in scores) / len(scores)
        print(f"\nCV Summary: Mean AP={mean_ap:.3f}  Mean F1={mean_f1:.3f}  ({len(scores)} folds)")

        if mean_ap < 0.45:
            print(f"\n*** WARNING: Mean AP {mean_ap:.3f} < 0.45 gate. "
                  f"Model may not have enough edge. ***")
        else:
            print(f"\n*** PASS: Mean AP {mean_ap:.3f} >= 0.45 gate. ***")

    return scores


# ── Final model training ──────────────────────────────────────────────────────

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_cutoff: str = "2024-01-01",
    n_estimators: int = 500,
):
    """Train on data before cutoff, evaluate on held-out test set."""
    cutoff = pd.Timestamp(test_cutoff, tz="UTC")

    X_train = X[X.index < cutoff]
    y_train = y[y.index < cutoff]
    X_test = X[X.index >= cutoff]
    y_test = y[y.index >= cutoff]

    print(f"Train: {len(X_train):,} bars ({X_train.index[0].date()} to {X_train.index[-1].date()})")
    print(f"Test:  {len(X_test):,} bars ({X_test.index[0].date()} to {X_test.index[-1].date()})")

    n_pos = int(y_train.sum())
    spw = (len(y_train) - n_pos) / n_pos if n_pos > 0 else 1.0
    print(f"Train class balance: BUY={n_pos:,} ({n_pos/len(y_train):.1%}), spw={spw:.2f}")

    final_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = n_estimators
    final_params["scale_pos_weight"] = spw

    print(f"\nTraining final model (n_estimators={n_estimators})...")
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train, verbose=False)

    if len(X_test) > 0 and y_test.sum() > 0:
        proba_test = model.predict_proba(X_test)[:, 1]
        ap_test = average_precision_score(y_test, proba_test)
        f1_test = f1_score(y_test, (proba_test >= 0.5).astype(int), zero_division=0)
        print(f"\nTest set: AP={ap_test:.3f}  F1={f1_test:.3f}  "
              f"({len(X_test):,} bars, {int(y_test.sum())} BUY)")

    importances = pd.Series(
        model.feature_importances_, index=model.feature_names_in_
    ).sort_values(ascending=False)
    print(f"\nFeature importances:")
    for feat, imp in importances.items():
        print(f"  {feat:>20}: {imp:.4f}")

    return model


def save_model(model: xgb.XGBClassifier, output_path: str) -> None:
    """Save model with pickle, verify round-trip."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    with open(output_path, "rb") as f:
        reloaded = pickle.load(f)
    assert list(reloaded.feature_names_in_) == list(model.feature_names_in_)

    print(f"\nSaved: {output_path}")
    print(f"Features ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")
    print("Round-trip pickle verification: OK")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train unified 20-coin XGBoost model (15M)")
    p.add_argument("--data-dir", default="data", help="Directory with parquet files")
    p.add_argument("--tp-mult", type=float, default=2.5,
                   help="TP multiplier for ATR-normalized labels (default 2.5)")
    p.add_argument("--sl-mult", type=float, default=0.8,
                   help="SL multiplier for ATR-normalized labels (default 0.8)")
    p.add_argument("--test-cutoff", default="2024-01-01",
                   help="Train/test split date (default 2024-01-01)")
    p.add_argument("--n-estimators", type=int, default=500,
                   help="n_estimators for final model (default 500)")
    p.add_argument("--cv-only", action="store_true",
                   help="Run CV only, skip final training")
    p.add_argument("--output", default="models/xgb_unified_20coin_15m.pkl")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  UNIFIED 20-COIN XGBOOST TRAINING")
    print("=" * 60)
    print(f"TP mult: {args.tp_mult}  SL mult: {args.sl_mult}  "
          f"Horizon: {HORIZON_15M} bars (4H)")

    print("\nStep 1: Loading and preparing all 20 coins...")
    pooled = prepare_all_coins(args.data_dir, args.tp_mult, args.sl_mult)

    print("\nStep 2: Preparing training data...")
    X, y = prepare_training_data(pooled)

    print("\nStep 3: Walk-forward CV...")
    run_walk_forward_cv(X, y)

    if args.cv_only:
        print("\nCV complete. Re-run without --cv-only to train and save.")
        return

    print("\nStep 4: Training final model...")
    model = train_final_model(X, y, args.test_cutoff, args.n_estimators)

    print("\nStep 5: Saving model...")
    save_model(model, args.output)

    print(f"\nDone. Next: python scripts/backtest_unified_20coin.py "
          f"--model {args.output}")


if __name__ == "__main__":
    main()

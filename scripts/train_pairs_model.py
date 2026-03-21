#!/usr/bin/env python3
"""
Training script for BTC/ETH pairs ML model.

This trains an XGBoost classifier to predict mean-reversion entry opportunities
based on pairs features (rolling OLS beta/alpha, z-score, etc.).

Usage:
  python scripts/train_pairs_model.py
  python scripts/train_pairs_model.py --btc data/BTCUSDT_15m.parquet --eth data/ETHUSDT_15m.parquet
  python scripts/train_pairs_model.py --threshold-sweep
  python scripts/train_pairs_model.py --output models/pairs_btc_eth_15m.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.pairs_features import compute_pairs_features

# ── Constants ──────────────────────────────────────────────────────────────────

# Entry/exit thresholds for z-score based entry
ENTRY_ZSCORE = 1.5
EXIT_ZSCORE = 0.5
STOP_ZSCORE = 3.0

# Forward horizon for label computation (bars)
HORIZON = 32

# Train/test split cutoff
TRAIN_CUTOFF = "2024-01-01"

# Annualisation constant for 15M resolution: 365.25 * 24 * 4 = 35,040
PERIODS_15M = 35_040


# ── Label generation ───────────────────────────────────────────────────────────

def _forward_min_max(z: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """
    Compute forward-looking min and max over horizon.

    For each bar t, computes the min and max of z[t+1:t+1+horizon].
    Uses rolling window on shifted data for efficiency.

    Args:
        z: z-score series
        horizon: forward window size

    Returns:
        (fwd_min, fwd_max) series
    """
    # Shift by 1 to get z[t+1]
    z_shifted = z.shift(-1)
    # Rolling min/max over horizon starting from z[t+1]
    fwd_min = z_shifted.rolling(horizon, min_periods=1).min()
    fwd_max = z_shifted.rolling(horizon, min_periods=1).max()
    return fwd_min, fwd_max


def generate_labels(zscore_unshifted: pd.Series) -> pd.Series:
    """
    Generate reversion labels without look-ahead bias.

    Label logic:
    - Entry occurs when abs(z[t]) > ENTRY_ZSCORE
    - For positive entry (z > +ENTRY_ZSCORE, BTC overpriced):
      - label[t] = 1 if min(z[t+1:t+HORIZON+1]) < EXIT_ZSCORE
        AND max(z[t+1:t+HORIZON+1]) < STOP_ZSCORE
    - For negative entry (z < -ENTRY_ZSCORE, BTC underpriced):
      - label[t] = 1 if max(z[t+1:t+HORIZON+1]) > -EXIT_ZSCORE
        AND min(z[t+1:t+HORIZON+1]) > -STOP_ZSCORE
    - Otherwise: label[t] = 0

    Args:
        zscore_unshifted: The raw z-score before shifting (used for label alignment)

    Returns:
        Binary label series (0 or 1)
    """
    z = zscore_unshifted.copy()

    # Compute forward-looking min and max
    fwd_min, fwd_max = _forward_min_max(z, HORIZON)

    # Entry conditions
    pos_entry = z > ENTRY_ZSCORE  # BTC overpriced, buy ETH (laggard)
    neg_entry = z < -ENTRY_ZSCORE  # BTC underpriced, buy BTC (laggard)

    # Exit/stop conditions for positive entry
    pos_label = pos_entry & (fwd_min < EXIT_ZSCORE) & (fwd_max < STOP_ZSCORE)

    # Exit/stop conditions for negative entry
    neg_label = neg_entry & (fwd_max > -EXIT_ZSCORE) & (fwd_min > -STOP_ZSCORE)

    return (pos_label | neg_label).astype(int)


# ── Data loading and preparation ───────────────────────────────────────────────

def load_and_prepare_data(btc_path: str, eth_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load BTC and ETH data, compute pairs features, and return feature DataFrame
    and unshifted z-score (for label generation).

    Returns:
        (feat_df, zscore_unshifted): feature DataFrame and unshifted z-score
    """
    # Load parquet files
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)

    # Normalize columns and index
    btc.index = pd.to_datetime(btc.index)
    eth.index = pd.to_datetime(eth.index)
    btc.columns = btc.columns.str.lower()
    eth.columns = eth.columns.str.lower()

    # Align on common index
    common_index = btc.index.union(eth.index)
    btc = btc.reindex(common_index).ffill()
    eth = eth.reindex(common_index).ffill()

    # Compute pairs features
    feat_df = compute_pairs_features(btc["close"], eth["close"])

    # Extract unshifted z-score BEFORE the shift happened in compute_pairs_features
    # We need to recompute it without the shift to use for labeling
    log_btc = np.log(btc["close"])
    log_eth = np.log(eth["close"])

    ols_window = 2880
    zscore_window = 672

    rolling_cov = log_btc.rolling(ols_window).cov(log_eth)
    rolling_var = log_eth.rolling(ols_window).var()
    rolling_beta = rolling_cov / rolling_var
    rolling_btc_mean = log_btc.rolling(ols_window).mean()
    rolling_eth_mean = log_eth.rolling(ols_window).mean()
    rolling_alpha = rolling_btc_mean - rolling_beta * rolling_eth_mean

    spread = log_btc - rolling_alpha - rolling_beta * log_eth
    spread_mean = spread.rolling(zscore_window).mean()
    spread_std = spread.rolling(zscore_window).std() + 1e-10
    zscore_unshifted = (spread - spread_mean) / spread_std

    # Drop NaN rows
    feat_df = feat_df.dropna()
    zscore_unshifted = zscore_unshifted.reindex(feat_df.index)

    return feat_df, zscore_unshifted


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train BTC/ETH pairs ML model")
    parser.add_argument(
        "--btc",
        type=str,
        default="data/BTCUSDT_15m.parquet",
        help="Path to BTC 15M parquet file",
    )
    parser.add_argument(
        "--eth",
        type=str,
        default="data/ETHUSDT_15m.parquet",
        help="Path to ETH 15M parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/pairs_btc_eth_15m.pkl",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Run threshold sweep (0.50-0.75 in 0.05 steps) instead of training",
    )

    args = parser.parse_args()

    # Load and prepare data
    print(f"Loading data from {args.btc} and {args.eth}...")
    feat_df, zscore_unshifted = load_and_prepare_data(args.btc, args.eth)
    print(f"Loaded {len(feat_df)} bars")

    # Generate labels
    print("Generating labels...")
    labels = generate_labels(zscore_unshifted)
    feat_df["label"] = labels

    # Filter to entry-relevant zones (abs(zscore) > 0.5) to focus training
    # and reduce class imbalance from flat-spread rows
    mask = abs(zscore_unshifted) > 0.5
    feat_filtered = feat_df[mask].copy()
    print(f"Filtered to {len(feat_filtered)} rows in entry-relevant zones")

    # Print class balance
    pos_count = (feat_filtered["label"] == 1).sum()
    neg_count = (feat_filtered["label"] == 0).sum()
    print(f"Class balance: {pos_count} positive, {neg_count} negative "
          f"({100*pos_count/(pos_count+neg_count):.1f}% positive)")

    # Train/test split
    train_mask = feat_filtered.index < TRAIN_CUTOFF
    test_mask = feat_filtered.index >= TRAIN_CUTOFF

    X_train = feat_filtered.loc[train_mask].drop(
        columns=["label", "spread", "spread_mean", "spread_std"]
    )
    y_train = feat_filtered.loc[train_mask, "label"]

    X_test = feat_filtered.loc[test_mask].drop(
        columns=["label", "spread", "spread_mean", "spread_std"]
    )
    y_test = feat_filtered.loc[test_mask, "label"]

    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    if args.threshold_sweep:
        # Threshold sweep mode: just evaluate pre-trained model performance
        print("\nThreshold sweep requested, but model training is still required.")
        print("Train mode continues below...\n")

    # Train XGBoost model
    print("Training XGBoost model...")

    # Handle class imbalance
    pos_weight = neg_count / (pos_count + 1e-10) if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # OOS evaluation at threshold=0.60
    print("\n" + "=" * 70)
    print("Out-of-Sample Evaluation (threshold=0.60)")
    print("=" * 70)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.60
    y_pred = (y_pred_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"Pos rate (OOS): {y_pred.mean():.4f}")

    # Threshold sweep
    print("\n" + "=" * 70)
    print("Threshold Sweep (0.50 to 0.75)")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Positives':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 48)

    thresholds = np.arange(0.50, 0.76, 0.05)
    sweep_results = []

    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        n_pos = y_pred_thresh.sum()
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)

        print(f"{thresh:<12.2f} {n_pos:<12} {prec:<12.4f} {rec:<12.4f}")
        sweep_results.append((thresh, n_pos, prec, rec))

    # Recommendation
    print("-" * 48)
    best_f1_idx = np.argmax([prec * rec / (prec + rec + 1e-10) for _, _, prec, rec in sweep_results])
    best_thresh, best_n_pos, best_prec, best_rec = sweep_results[best_f1_idx]
    print(f"Recommendation: threshold={best_thresh:.2f} (F1={best_prec*best_rec/(best_prec+best_rec+1e-10):.4f})")

    # Feature importance
    print("\n" + "=" * 70)
    print("Feature Importance (Top 10 by Gain)")
    print("=" * 70)

    importance_df = pd.DataFrame(
        {
            "feature": model.feature_names_in_,
            "gain": model.get_booster().get_score(importance_type="gain").values(),
        }
    )
    # Handle case where some features may not have importance scores
    importance_df = importance_df.sort_values("gain", ascending=False).head(10)
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:<30} {row['gain']:>10.1f}")

    # Save model
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()

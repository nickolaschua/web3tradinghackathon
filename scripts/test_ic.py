"""
Information Coefficient (IC) test for feature validation.

Computes Spearman IC = corr(feature_t, forward_return_{t+horizon}) using
non-overlapping 90-day windows on the DEVELOPMENT split only (before 2023-07-01).
The validation and test sets are never touched here.

Acceptance thresholds (iteration_process.md):
  Mean IC > 0.03          -- feature has predictive power
  Positive windows > 60%  -- signal is consistent, not a single-period fluke
  Concentration < 3x mean -- no single window dominates

Usage:
  python scripts/test_ic.py                        # IC for all features
  python scripts/test_ic.py --sensitivity          # window size sweep for new features
  python scripts/test_ic.py --feature eth_btc_corr # single feature deep-dive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import (
    compute_btc_context_features,
    compute_cross_asset_features,
    compute_features,
)

# -- Data partitions ------------------------------------------------------------
DEV_END = pd.Timestamp("2023-07-01", tz="UTC")   # development split ends here
VAL_END = pd.Timestamp("2024-01-01", tz="UTC")   # = TRAIN_CUTOFF in backtest (NEVER touch test)

HORIZON = 6           # forward bars to compute return label (24H at 4H cadence)
WINDOW_BARS = 540     # 90 days x 6 bars/day = one IC evaluation chunk

# -- Feature groups -------------------------------------------------------------
BASELINE_FEATURES = [
    "atr_proxy", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    "eth_return_lag1", "eth_return_lag2", "sol_return_lag1", "sol_return_lag2",
]
NEW_FEATURES = [
    "eth_btc_corr", "sol_btc_corr",
    "eth_btc_beta", "sol_btc_beta",
    "eth_btc_divergence", "sol_btc_divergence",
]

# Window sizes (in bars) for sensitivity sweep: 14d, 20d, 30d, 45d, 60d at 4H
SENSITIVITY_WINDOWS = [84, 120, 180, 270, 360]


def load_and_compute(data_dir: str = "data", corr_window: int = 180) -> pd.DataFrame:
    """Load parquets, compute full feature set, return dev+val split (no test)."""
    btc = pd.read_parquet(f"{data_dir}/BTCUSDT_4h.parquet")
    eth = pd.read_parquet(f"{data_dir}/ETHUSDT_4h.parquet")
    sol = pd.read_parquet(f"{data_dir}/SOLUSDT_4h.parquet")

    for df in (btc, eth, sol):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    feat = compute_features(btc)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    feat = compute_btc_context_features(feat, eth, sol, window=corr_window)
    feat = feat.dropna()

    # Never expose test set in IC analysis
    return feat[feat.index < VAL_END]


def compute_ic_windows(
    feat_df: pd.DataFrame,
    feature: str,
    horizon: int = HORIZON,
    chunk_size: int = WINDOW_BARS,
) -> pd.Series:
    """
    Compute Spearman IC in non-overlapping chunks of `chunk_size` bars.

    Returns a Series of IC values (one per chunk).
    Using non-overlapping windows prevents inflated window count from autocorrelation.
    """
    if feature not in feat_df.columns:
        return pd.Series(dtype=float)

    fwd_ret = feat_df["close"].shift(-horizon) / feat_df["close"] - 1
    combined = pd.DataFrame({"feature": feat_df[feature], "fwd_ret": fwd_ret}).dropna()

    # Exclude last `horizon` rows (no valid forward return)
    combined = combined.iloc[:-horizon]

    if len(combined) < chunk_size:
        return pd.Series(dtype=float)

    ics = []
    n_chunks = len(combined) // chunk_size
    for i in range(n_chunks):
        chunk = combined.iloc[i * chunk_size : (i + 1) * chunk_size]
        ic, _ = spearmanr(chunk["feature"], chunk["fwd_ret"])
        if not np.isnan(ic):
            ics.append(ic)

    return pd.Series(ics)


def report_ic(feature: str, ic_series: pd.Series) -> dict:
    """Print IC statistics and return a summary dict."""
    if ic_series.empty:
        print(f"  {feature:<32} NO DATA")
        return {"feature": feature, "mean_ic": np.nan, "pct_positive": np.nan,
                "n_windows": 0, "status": "NO DATA"}

    mean_ic = ic_series.mean()
    pct_pos = (ic_series > 0).mean()
    max_abs = ic_series.abs().max()
    n = len(ic_series)

    pass_mean = mean_ic > 0.03
    pass_pct  = pct_pos > 0.60
    pass_conc = max_abs < abs(mean_ic) * 3 if abs(mean_ic) > 1e-6 else True

    if pass_mean and pass_pct:
        status = "PASS"
    elif mean_ic > 0:
        status = "MARGINAL"
    else:
        status = "FAIL"

    conc_flag = " [CONCENTRATED]" if not pass_conc else ""
    print(
        f"  {feature:<32} IC={mean_ic:+.4f}  pos%={pct_pos:.0%}  "
        f"max={max_abs:.4f}  n={n:2d}  [{status}]{conc_flag}"
    )

    return {
        "feature": feature, "mean_ic": mean_ic, "pct_positive": pct_pos,
        "max_ic": max_abs, "n_windows": n, "status": status,
    }


def sensitivity_sweep(feat_df_base, eth_df, sol_df, features: list[str]) -> None:
    """Re-compute features for each window size and report IC stability."""
    print("\n=== Sensitivity Sweep: rolling window size ===")
    print(f"  Testing features: {features}")
    print(f"  Window sizes (bars): {SENSITIVITY_WINDOWS}\n")

    btc = pd.read_parquet("data/BTCUSDT_4h.parquet")
    eth = pd.read_parquet("data/ETHUSDT_4h.parquet")
    sol = pd.read_parquet("data/SOLUSDT_4h.parquet")
    for df in (btc, eth, sol):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    for feat_name in features:
        print(f"Feature: {feat_name}")
        print(f"  {'Window':>8}  {'Days':>6}  {'Mean IC':>10}  {'Pos%':>7}  Status")
        print(f"  {'-'*50}")
        for w in SENSITIVITY_WINDOWS:
            base = compute_features(btc)
            base = compute_cross_asset_features(base, {"ETH/USD": eth, "SOL/USD": sol})
            base = compute_btc_context_features(base, eth, sol, window=w)
            base = base.dropna()
            base = base[base.index < DEV_END]

            ic = compute_ic_windows(base, feat_name)
            if ic.empty:
                print(f"  {w:>8}  {w//6:>6}  {'NO DATA':>10}")
                continue
            mean_ic = ic.mean()
            pct_pos = (ic > 0).mean()
            status = "PASS" if mean_ic > 0.03 and pct_pos > 0.60 else ("POS" if mean_ic > 0 else "FAIL")
            print(f"  {w:>8}  {w//6:>6}  {mean_ic:>+10.4f}  {pct_pos:>6.0%}  [{status}]")
        print()


def main():
    parser = argparse.ArgumentParser(description="IC test for feature validation")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run window-size sensitivity sweep for new features")
    parser.add_argument("--feature", type=str, default=None,
                        help="Test only this feature (name as in DataFrame)")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    print("Loading data and computing features...")
    feat_df = load_and_compute(args.data_dir)
    dev_df  = feat_df[feat_df.index < DEV_END]
    print(f"Development split: {dev_df.index[0].date()} to {dev_df.index[-1].date()} "
          f"({len(dev_df)} bars, {len(dev_df) // WINDOW_BARS} IC windows)\n")

    if args.sensitivity:
        eth = pd.read_parquet(f"{args.data_dir}/ETHUSDT_4h.parquet")
        sol = pd.read_parquet(f"{args.data_dir}/SOLUSDT_4h.parquet")
        sensitivity_sweep(dev_df, eth, sol, NEW_FEATURES)
        return

    features = [args.feature] if args.feature else (BASELINE_FEATURES + NEW_FEATURES)

    print(f"=== IC Test -- Development split (horizon={HORIZON} bars = {HORIZON*4}H) ===")
    print(f"Chunk size: {WINDOW_BARS} bars ~= 90 days  |  Acceptance: IC>0.03, pos%>60%\n")

    results = []

    if not args.feature:
        print("--- Baseline features (already in model) ---")
    for feat in [f for f in features if f in BASELINE_FEATURES]:
        ic = compute_ic_windows(dev_df, feat)
        results.append(report_ic(feat, ic))

    if not args.feature:
        print("\n--- New BTC lead-lag features (this iteration) ---")
    for feat in [f for f in features if f in NEW_FEATURES]:
        ic = compute_ic_windows(dev_df, feat)
        results.append(report_ic(feat, ic))

    if args.feature and args.feature not in BASELINE_FEATURES + NEW_FEATURES:
        ic = compute_ic_windows(dev_df, args.feature)
        results.append(report_ic(args.feature, ic))

    # Summary
    df_res = pd.DataFrame(results).dropna(subset=["mean_ic"])
    if df_res.empty:
        return

    new_res = df_res[df_res["feature"].isin(NEW_FEATURES)]
    pass_new = new_res[new_res["status"] == "PASS"]
    marginal_new = new_res[new_res["status"] == "MARGINAL"]

    print(f"\n=== Summary ===")
    print(f"New features:   {len(new_res)} tested  |  "
          f"{len(pass_new)} PASS  |  {len(marginal_new)} MARGINAL")

    if not pass_new.empty:
        best = pass_new.loc[pass_new["mean_ic"].idxmax()]
        print(f"Best new feature: {best['feature']}  IC={best['mean_ic']:.4f}")

    add_to_model = new_res[new_res["mean_ic"] > 0.02].sort_values("mean_ic", ascending=False)
    if not add_to_model.empty:
        print("\nRecommended additions to FEATURE_COLS (IC > 0.02):")
        for _, row in add_to_model.iterrows():
            print(f"  \"{row['feature']}\"  # IC={row['mean_ic']:.4f}  pos%={row['pct_positive']:.0%}  [{row['status']}]")
    else:
        print("\nNo new features meet the IC > 0.02 threshold -- review thresholds or feature design.")


if __name__ == "__main__":
    main()

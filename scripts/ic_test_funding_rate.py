"""
Iteration 3 — Funding Rate Sentiment: IC test script.

Tests candidate features on the dev split ONLY (< 2023-07-01).
NEVER touches the test set (>= 2024-01-01).

Candidate features (all BTC-specific):
  btc_funding_latest       : settled funding rate, fwd-filled to 4H, shifted 1
  btc_funding_ma_24h       : 3-settlement (24h) rolling mean
  btc_funding_change_24h   : latest - 3-settlements-ago (sentiment momentum)
  btc_funding_self_zscore  : z-score vs own 90-day history
  btc_funding_extreme      : |self_zscore| > 2 (binary)

Acceptance thresholds (same across all iterations):
  mean IC > 0.03
  positive IC in > 60% of non-overlapping windows
  max IC < 3× |mean IC|  (CONCENTRATED flag if violated)

Window size: 540 bars = ~90 days at 4H cadence.
Forward return horizon: 6 bars = 24H.
"""
import json
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
from bot.data.funding_features import compute_btc_funding_features
from bot.data.funding_fetcher import load_or_fetch_funding

# ── Constants ────────────────────────────────────────────────────────────────
DEV_CUTOFF    = pd.Timestamp("2023-07-01", tz="UTC")
HORIZON       = 6      # forward bars for label
WINDOW_BARS   = 540    # IC test window (≈90 days at 4H)
MIN_IC_MEAN   = 0.03
MIN_POS_PCT   = 0.60
CONC_RATIO    = 3.0    # max_ic / |mean_ic| threshold for CONCENTRATED flag

CANDIDATE_FEATURES = [
    "btc_funding_latest",
    "btc_funding_ma_24h",
    "btc_funding_change_24h",
    "btc_funding_self_zscore",
    "btc_funding_extreme",
]


# ── IC helpers ────────────────────────────────────────────────────────────────

def ic_test_single(feature: pd.Series, target: pd.Series) -> dict:
    """
    Non-overlapping window IC test.

    Divides the aligned (feature, target) series into WINDOW_BARS-sized chunks,
    computes Spearman correlation per chunk, and reports summary statistics.
    """
    aligned = pd.concat([feature.rename("feat"), target.rename("tgt")], axis=1).dropna()
    n = len(aligned)
    ics = []

    for start in range(0, n - WINDOW_BARS + 1, WINDOW_BARS):
        chunk = aligned.iloc[start : start + WINDOW_BARS]
        if len(chunk) < 30:
            continue
        ic, _ = spearmanr(chunk["feat"], chunk["tgt"])
        if not np.isnan(ic):
            ics.append(float(ic))

    if not ics:
        return {
            "mean_ic": float("nan"), "pos_pct": float("nan"),
            "max_ic": float("nan"), "n_windows": 0, "status": "FAIL",
        }

    mean_ic = float(np.mean(ics))
    pos_pct = float(np.mean([ic > 0 for ic in ics]))
    max_ic  = float(max(abs(ic) for ic in ics))

    pass_mean = mean_ic > MIN_IC_MEAN
    pass_pos  = pos_pct > MIN_POS_PCT
    concentrated = (abs(mean_ic) > 0) and (max_ic > CONC_RATIO * abs(mean_ic))

    if pass_mean and pass_pos:
        status = "PASS+CONCENTRATED" if concentrated else "PASS"
    elif 0 < mean_ic <= MIN_IC_MEAN and pass_pos:
        status = "MARGINAL+CONCENTRATED" if concentrated else "MARGINAL"
    else:
        status = "FAIL"

    return {
        "mean_ic": mean_ic,
        "pos_pct": pos_pct,
        "max_ic":  max_ic,
        "n_windows": len(ics),
        "status": status,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Iteration 3: Funding Rate Sentiment — IC Test")
    print(f"Dev split: < {DEV_CUTOFF.date()}  |  Horizon: {HORIZON} bars  |  Window: {WINDOW_BARS} bars")
    print("=" * 60)

    # Step 1: Load OHLCV
    print("\nStep 1: Loading OHLCV data...")
    btc = pd.read_parquet(project_root / "data" / "BTCUSDT_4h.parquet")
    eth = pd.read_parquet(project_root / "data" / "ETHUSDT_4h.parquet")
    sol = pd.read_parquet(project_root / "data" / "SOLUSDT_4h.parquet")
    for df in (btc, eth, sol):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()
    print(f"  BTC: {len(btc)} bars ({btc.index[0].date()} — {btc.index[-1].date()})")

    # Step 2: Fetch / load funding rates
    print("\nStep 2: Loading BTC funding rates...")
    funding_df = load_or_fetch_funding(
        symbol="BTCUSDT",
        start_date="2021-03-01",
        end_date="2026-04-01",
        cache_dir=project_root / "data" / "funding",
    )
    if funding_df.empty:
        print("ERROR: No funding rate data available. Cannot run IC test.")
        return
    print(f"  {len(funding_df)} records  ({funding_df['fundingTime'].min().date()} — {funding_df['fundingTime'].max().date()})")

    # Step 3: Compute all features
    print("\nStep 3: Computing features...")
    feat = compute_features(btc)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    feat = compute_btc_context_features(feat, eth, sol)

    funding_feat = compute_btc_funding_features(funding_df, feat.index)
    for col in funding_feat.columns:
        feat[col] = funding_feat[col]

    feat = feat.dropna(subset=["RSI_14", "eth_btc_corr"])  # warmup only
    print(f"  Feature matrix: {feat.shape[0]} bars × {feat.shape[1]} columns")

    # Step 4: Build forward return (not filtered to dev yet)
    fwd_ret = btc["close"].shift(-HORIZON) / btc["close"] - 1
    fwd_ret = fwd_ret.reindex(feat.index)

    # Step 5: Filter to dev split, drop last HORIZON rows (no label)
    dev_feat = feat[feat.index < DEV_CUTOFF].iloc[:-HORIZON].copy()
    dev_fwd  = fwd_ret[fwd_ret.index < DEV_CUTOFF].iloc[:-HORIZON].copy()

    # Align
    valid_idx = dev_feat.index.intersection(dev_fwd.dropna().index)
    dev_feat = dev_feat.loc[valid_idx]
    dev_fwd  = dev_fwd.loc[valid_idx]
    print(f"\n  Dev split: {len(dev_feat)} bars ({dev_feat.index[0].date()} — {dev_feat.index[-1].date()})")

    # Step 6: IC tests
    print(f"\nStep 4: Running IC tests (window={WINDOW_BARS} bars, non-overlapping)...\n")
    results: dict[str, dict] = {}

    for feat_name in CANDIDATE_FEATURES:
        if feat_name not in dev_feat.columns:
            print(f"  {feat_name}: MISSING — skipping")
            continue
        r = ic_test_single(dev_feat[feat_name], dev_fwd)
        results[feat_name] = r

    # Print summary table
    print(f"\n{'Feature':<35} | {'Mean IC':>8} | {'Pos%':>6} | {'Max IC':>8} | {'n':>3} | Status")
    print("-" * 82)
    for feat_name, r in results.items():
        print(
            f"{feat_name:<35} | {r['mean_ic']:>+8.4f} | {r['pos_pct']:>6.0%} | "
            f"{r['max_ic']:>8.4f} | {r['n_windows']:>3} | {r['status']}"
        )

    # Save JSON for iteration_log update
    out_path = project_root / "research" / "iter3_ic_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path.relative_to(project_root)}")
    print("\nNext: review results, run CV comparison if any feature passes, update iteration_log.md")


if __name__ == "__main__":
    main()

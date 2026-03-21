#!/usr/bin/env python3
"""
Test Advanced Funding Signals - Integrated Backtest
Extends alpha_research.py with new funding signals
"""
import sys
sys.path.insert(0, 'scripts')

import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from funding_alpha_signals import compute_all_funding_signals
warnings.filterwarnings("ignore")

# Import from alpha_research
from alpha_research import (
    DATA_DIR, RESULTS_DIR, OHLCV_PAIRS, FUNDING_PAIRS,
    XGB_PARAMS, LABEL_HORIZON, LABEL_THRESHOLD, BASE_FEATURE_COLS,
    download_all_data, compute_base_features, compute_cross_asset_lags,
    compute_funding_features, compute_macro_features, compute_paxg_features,
    compute_volume_features, compute_cross_sectional_momentum,
    run_cv, test_feature_set
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_splits = 3 if args.quick else 5

    print("="*70)
    print("  ADVANCED FUNDING SIGNALS BACKTEST")
    print("="*70)

    # [1] Load data (same as alpha_research.py)
    print("\n[1/5] Loading data...")
    if not args.skip_download:
        download_all_data(DATA_DIR)

    ohlcv = {}
    for s in OHLCV_PAIRS:
        p = DATA_DIR / f"{s}_4h.parquet"
        if p.exists():
            ohlcv[s] = pd.read_parquet(p)
            print(f"  {s}: {len(ohlcv[s])} bars")

    if "BTCUSDT" not in ohlcv:
        print("FATAL: BTC data missing")
        return

    funding = {}
    for s in FUNDING_PAIRS:
        p = DATA_DIR / f"{s}_funding.parquet"
        if p.exists():
            funding[s] = pd.read_parquet(p)
            print(f"  {s} funding: {len(funding[s])} rows")

    macro = {}
    for n in ["oil","dxy"]:
        p = DATA_DIR / f"{n}_daily.parquet"
        if p.exists():
            macro[n] = pd.read_parquet(p)
            print(f"  {n}: {len(macro[n])} daily bars")

    # [2] Compute base features
    print("\n[2/5] Computing base features...")
    btc = ohlcv["BTCUSDT"]
    feat = compute_base_features(btc)
    feat = compute_cross_asset_lags(feat, {s: ohlcv[s] for s in ["ETHUSDT","SOLUSDT"] if s in ohlcv})

    # OLD funding features
    if funding:
        feat = compute_funding_features(feat, funding)

    if macro:
        feat = compute_macro_features(feat, macro)
    if "PAXGUSDT" in ohlcv:
        feat = compute_paxg_features(feat, ohlcv["PAXGUSDT"])
    feat = compute_volume_features(feat)
    feat = compute_cross_sectional_momentum(feat, ohlcv)

    print(f"  Base features: {feat.shape}")

    # [3] Compute ADVANCED FUNDING SIGNALS
    print("\n[3/5] Computing advanced funding signals...")
    advanced_signals = compute_all_funding_signals(feat, funding, ohlcv)
    print(f"  Advanced signals: {advanced_signals.shape}")
    print(f"  Signal columns: {list(advanced_signals.columns)}")

    # Join with features
    feat = feat.join(advanced_signals)
    print(f"  Combined features: {feat.shape}")

    # [4] Labels
    print(f"\n[4/5] Computing labels (horizon={LABEL_HORIZON}, threshold={LABEL_THRESHOLD})...")
    fwd_ret = feat["close"].shift(-LABEL_HORIZON) / feat["close"] - 1
    labels = (fwd_ret >= LABEL_THRESHOLD).astype(int)
    X_all = feat.iloc[:-LABEL_HORIZON].copy()
    y_all = labels.iloc[:-LABEL_HORIZON].copy()

    bv = X_all[BASE_FEATURE_COLS].dropna().index
    X_all = X_all.loc[bv]
    y_all = y_all.loc[bv]
    br = y_all.mean()
    print(f"  Bars: {len(X_all)} | BUY rate: {br:.1%}")

    # [5] Test advanced signal groups
    print(f"\n[5/5] Testing advanced funding signal groups ({n_splits}-fold CV)...\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # Define signal groups
    signal_groups = {
        "Signal 1: Funding Leadership": [
            "funding_leadership_raw",
            "funding_leadership_aligned",
        ],
        "Signal 2: Leverage Persistence": [
            "funding_persistence_regime",
            "funding_persistence_vol_adjusted",
            "funding_zscore",
        ],
        "Signal 3: Funding Divergence": [
            "funding_divergence_raw",
            "funding_divergence_strong",
            "funding_cum24h_zscore",
        ],
        "Signal 4: Vol-Adjusted Funding": [
            "funding_vol_amplified",
            "funding_vol_volume_amplified",
        ],
        "Signal 5: Composite Sentiment": [
            "funding_composite",
            "funding_composite_extreme",
            "funding_consensus",
        ],
        "All Advanced Signals": [c for c in advanced_signals.columns if c in X_all.columns],
        "OLD Funding (baseline)": [c for c in X_all.columns if "funding" in c and c not in advanced_signals.columns],
    }

    # Add interaction groups
    signal_groups["Leadership + Divergence"] = (
        signal_groups["Signal 1: Funding Leadership"] +
        signal_groups["Signal 3: Funding Divergence"]
    )
    signal_groups["Composite + Vol-Adjusted"] = (
        signal_groups["Signal 5: Composite Sentiment"] +
        signal_groups["Signal 4: Vol-Adjusted Funding"]
    )

    for gn, cols in signal_groups.items():
        cols_in_x = [c for c in cols if c in X_all.columns]
        if not cols_in_x:
            print(f"  {gn}: no features, skipping")
            continue

        print(f"  Testing: {gn} ({len(cols_in_x)} feat)...", end=" ", flush=True)
        r = test_feature_set(X_all, y_all, cols_in_x, gn, n_splits)
        results.append(r)
        print(f"dAP: {r['delta_ap']:+.4f} - {r['verdict']}")

    # [6] Results summary
    print("\n" + "="*70)
    print("  RESULTS - Advanced Funding Signals Performance")
    print("="*70)
    results.sort(key=lambda r: r["delta_ap"], reverse=True)
    print(f"\n  {'Group':<35} {'dAP':>8} {'Base':>8} {'Aug':>8} {'Verdict'}")
    print("  " + "-"*75)
    for r in results:
        m = " ***" if r["delta_ap"] > 0.005 else ""
        print(f"  {r['label']:<35} {r['delta_ap']:>+8.4f} {r['baseline_ap']:>8.4f} {r['augmented_ap']:>8.4f} {r['verdict']}{m}")

    winners = [r for r in results if r["delta_ap"] > 0.005]
    if winners:
        print(f"\n  🏆 STRONG PERFORMERS (dAP > 0.005):")
        for w in winners:
            print(f"    {w['label']}: dAP = {w['delta_ap']:+.4f}")
    else:
        print(f"\n  ⚠️  No signal improved AP by > 0.005")

    # Save results
    with open(RESULTS_DIR / "advanced_funding_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_DIR / 'advanced_funding_results.json'}")

    # [7] Feature importance analysis for best signal
    if winners:
        print(f"\n" + "="*70)
        print("  FEATURE IMPORTANCE - Top Performing Signal")
        print("="*70)

        best = winners[0]
        bc = [c for c in BASE_FEATURE_COLS if c in X_all.columns]
        ac = bc + [c for c in best['candidate_cols'] if c in X_all.columns]

        valid = X_all.index
        for col in ac:
            if col in X_all.columns:
                valid = valid[X_all[col].reindex(valid).notna()]

        Xv, yv = X_all.loc[valid][ac], y_all.loc[valid]

        np_, nn = int(yv.sum()), len(yv) - int(yv.sum())
        if np_ > 0 and nn > 0:
            m = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": nn/np_})
            m.fit(Xv, yv, verbose=False)

            importances = pd.DataFrame({
                "feature": Xv.columns,
                "importance": m.feature_importances_
            }).sort_values("importance", ascending=False)

            print(f"\n  Top 15 features for '{best['label']}':")
            print(f"  {'Feature':<40} {'Importance':>12}")
            print("  " + "-"*55)
            for _, row in importances.head(15).iterrows():
                marker = "🔥" if row['feature'] in best['candidate_cols'] else "  "
                print(f"  {marker} {row['feature']:<38} {row['importance']:>12.4f}")

    print("\n" + "="*70)
    print("Done.")
    print("="*70)

if __name__ == "__main__":
    main()

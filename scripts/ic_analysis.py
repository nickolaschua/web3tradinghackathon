#!/usr/bin/env python3
"""
IC analysis for momentum signal components.

Computes Spearman rank IC for each signal component across 20 coins
on formation period (2024-01 to 2025-06). Determines which components
survive and their weights for the composite score.

Usage:
  python scripts/ic_analysis.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.strategy.momentum_signals import (
    resample_to_4h, sharpe_momentum, nearness_to_high, residual_momentum,
)

COINS = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","LINK","DOT",
         "LTC","UNI","NEAR","SUI","APT","PEPE","ARB","SHIB","FIL","HBAR"]

FORMATION_START = pd.Timestamp("2024-01-01", tz="UTC")
FORMATION_END = pd.Timestamp("2025-06-01", tz="UTC")


def main():
    data_dir = Path("data")

    # Load and resample all coins to 4H
    print("Loading and resampling to 4H...", flush=True)
    coin_4h = {}
    for coin in COINS:
        df = pd.read_parquet(data_dir / f"{coin}USDT_15m.parquet")
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()
        coin_4h[coin] = resample_to_4h(df)

    btc_4h = coin_4h["BTC"]
    btc_ret = np.log(btc_4h["close"] / btc_4h["close"].shift(1)).dropna()

    # For each rebalancing point in formation period, compute scores and forward returns
    # Rebalancing every 4H = every bar in the 4H data
    formation_idx = btc_4h.index[
        (btc_4h.index >= FORMATION_START) & (btc_4h.index < FORMATION_END)
    ]

    print(f"Formation period: {FORMATION_START.date()} to {FORMATION_END.date()}")
    print(f"Rebalancing points: {len(formation_idx)}", flush=True)

    # Collect cross-sectional scores at each rebalancing point
    records = []
    for i, ts in enumerate(formation_idx):
        if i % 100 == 0:
            print(f"  Processing bar {i}/{len(formation_idx)}...", flush=True)

        for coin in COINS:
            c4h = coin_4h[coin]
            if ts not in c4h.index:
                continue

            loc = c4h.index.get_loc(ts)
            if loc < 200:  # need enough history
                continue

            closes = c4h["close"].iloc[:loc + 1]

            # Forward 4H return (the thing we're predicting)
            if loc + 1 >= len(c4h):
                continue
            fwd_ret = np.log(c4h["close"].iloc[loc + 1] / c4h["close"].iloc[loc])

            # Compute each signal component
            sm_48h = sharpe_momentum(closes, lookback=12, skip=1)
            sm_168h = sharpe_momentum(closes, lookback=42, skip=1)
            near = nearness_to_high(closes, window=180)

            # Residual momentum
            coin_ret = np.log(c4h["close"].iloc[:loc + 1] / c4h["close"].iloc[:loc + 1].shift(1)).dropna()
            common = coin_ret.index.intersection(btc_ret.index)
            common = common[common <= ts]
            if len(common) > 60:
                res_mom = residual_momentum(
                    coin_ret.reindex(common), btc_ret.reindex(common)
                )
            else:
                res_mom = np.nan

            records.append({
                "ts": ts, "coin": coin,
                "sharpe_48h": sm_48h, "nearness": near,
                "sharpe_168h": sm_168h, "residual": res_mom,
                "fwd_return": fwd_ret,
            })

    df = pd.DataFrame(records)
    print(f"\nTotal observations: {len(df):,}", flush=True)

    # Compute IC per component (cross-sectional Spearman at each timestamp, then average)
    components = ["sharpe_48h", "nearness", "sharpe_168h", "residual"]

    print("\n" + "=" * 70)
    print("IC ANALYSIS (Spearman rank correlation with forward 4H return)")
    print("=" * 70)

    ics = {}
    for comp in components:
        # Per-timestamp cross-sectional IC
        per_ts_ic = []
        for ts, group in df.groupby("ts"):
            valid = group[[comp, "fwd_return"]].dropna()
            if len(valid) < 5:
                continue
            corr, _ = stats.spearmanr(valid[comp], valid["fwd_return"])
            if not np.isnan(corr):
                per_ts_ic.append(corr)

        if per_ts_ic:
            mean_ic = np.mean(per_ts_ic)
            std_ic = np.std(per_ts_ic)
            t_stat = mean_ic / (std_ic / np.sqrt(len(per_ts_ic))) if std_ic > 0 else 0
            pct_positive = sum(1 for x in per_ts_ic if x > 0) / len(per_ts_ic) * 100
        else:
            mean_ic = std_ic = t_stat = 0.0
            pct_positive = 0.0

        ics[comp] = mean_ic
        status = "KEEP" if mean_ic > 0 else "DROP"
        print(f"  {comp:>15}: IC={mean_ic:>+.4f}  std={std_ic:.4f}  "
              f"t={t_stat:>5.2f}  %pos={pct_positive:>4.0f}%  [{status}]", flush=True)

    # Derive weights from positive-IC components
    positive = {k: v for k, v in ics.items() if v > 0}
    if not positive:
        print("\n*** ALL COMPONENTS HAVE IC <= 0. Strategy has no edge. ***")
        return

    total_ic = sum(positive.values())
    weights = {k: round(v / total_ic, 3) for k, v in positive.items()}

    print(f"\n  Surviving components: {list(positive.keys())}")
    print(f"  IC-derived weights: {weights}")
    print(f"  Dropped components: {[k for k in ics if k not in positive]}")

    # Output as Python dict for copy-paste into config
    print(f"\n  # Copy into config or strategy:")
    print(f"  SIGNAL_WEIGHTS = {weights}")

    print("\n" + "=" * 70)
    print("Done. Use these weights in the backtest and strategy.")


if __name__ == "__main__":
    main()

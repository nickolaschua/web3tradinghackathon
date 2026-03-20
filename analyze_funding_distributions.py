#!/usr/bin/env python3
"""
Empirical Funding Rate Analysis - Data-Driven Signal Design
Analyzes actual BTC/ETH/SOL funding distributions from 2022-2025 to derive signal thresholds
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("research_data")

def load_and_analyze_funding():
    """Load funding data and compute empirical distributions"""
    results = {}

    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        path = DATA_DIR / f"{symbol}_funding.parquet"
        if not path.exists():
            print(f"⚠️  {symbol} funding data not found")
            continue

        df = pd.read_parquet(path)
        fr = df["funding_rate"]

        # Basic stats
        stats = {
            "symbol": symbol,
            "n_obs": len(fr),
            "mean": fr.mean(),
            "std": fr.std(),
            "median": fr.median(),
            "min": fr.min(),
            "max": fr.max(),
        }

        # Percentiles (for extreme thresholds)
        percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
        for p in percentiles:
            stats[f"p{p}"] = fr.quantile(p/100)

        # Z-score distribution (using 90-period rolling stats)
        rolling_mean = fr.rolling(90, min_periods=30).mean()
        rolling_std = fr.rolling(90, min_periods=30).std()
        zscore = (fr - rolling_mean) / rolling_std.replace(0, np.nan)

        stats["zscore_mean"] = zscore.mean()
        stats["zscore_std"] = zscore.std()
        stats["zscore_p95"] = zscore.quantile(0.95)
        stats["zscore_p99"] = zscore.quantile(0.99)
        stats["zscore_p05"] = zscore.quantile(0.05)
        stats["zscore_p01"] = zscore.quantile(0.01)

        # Extreme regime analysis (abs z-score > 2)
        extreme_mask = (np.abs(zscore) > 2.0)
        stats["pct_extreme"] = extreme_mask.sum() / len(zscore.dropna()) * 100

        # Persistence analysis
        extreme_flag = extreme_mask.astype(float)
        persistence = extreme_flag.rolling(3).sum()
        stats["pct_3bar_persistence"] = (persistence >= 2).sum() / len(persistence.dropna()) * 100

        # Reversion analysis (does extreme funding predict reversal?)
        # Align with 4H bars
        fr_4h = fr.resample("4h").ffill()
        zscore_4h = zscore.resample("4h").ffill()

        # Future 24H funding change after extreme events
        fwd_24h_change = fr_4h.shift(-6) - fr_4h  # 6 bars = 24H
        extreme_4h = (np.abs(zscore_4h) > 2.0)

        if extreme_4h.sum() > 10:
            stats["extreme_fwd_24h_mean_reversion"] = fwd_24h_change[extreme_4h].mean()
            stats["extreme_fwd_24h_std"] = fwd_24h_change[extreme_4h].std()
        else:
            stats["extreme_fwd_24h_mean_reversion"] = np.nan
            stats["extreme_fwd_24h_std"] = np.nan

        # Cross-sectional comparison: cumulative funding over 24H
        cum_24h = fr_4h.rolling(6).sum()
        stats["cum24h_mean"] = cum_24h.mean()
        stats["cum24h_std"] = cum_24h.std()
        stats["cum24h_p95"] = cum_24h.quantile(0.95)
        stats["cum24h_p05"] = cum_24h.quantile(0.05)

        results[symbol] = stats

        # Print summary
        prefix = symbol.replace("USDT", "")
        print(f"\n{'='*70}")
        print(f"{prefix} FUNDING RATE ANALYSIS (n={stats['n_obs']:,})")
        print(f"{'='*70}")
        print(f"  Mean: {stats['mean']*10000:.2f} bps | Std: {stats['std']*10000:.2f} bps")
        print(f"  Range: [{stats['min']*10000:.2f}, {stats['max']*10000:.2f}] bps")
        print(f"  P1-P99: [{stats['p1']*10000:.2f}, {stats['p99']*10000:.2f}] bps")
        print(f"\n  Z-score (90-period rolling):")
        print(f"    95% of data within: [{stats['zscore_p05']:.2f}, {stats['zscore_p95']:.2f}]σ")
        print(f"    99% of data within: [{stats['zscore_p01']:.2f}, {stats['zscore_p99']:.2f}]σ")
        print(f"    % observations |z| > 2: {stats['pct_extreme']:.2f}%")
        print(f"    % with 2+ consecutive extreme: {stats['pct_3bar_persistence']:.2f}%")
        print(f"\n  24H forward dynamics after extreme events:")
        if not np.isnan(stats['extreme_fwd_24h_mean_reversion']):
            print(f"    Mean funding change: {stats['extreme_fwd_24h_mean_reversion']*10000:.2f} bps (negative = mean reversion)")
        else:
            print(f"    Insufficient extreme events")
        print(f"\n  Cumulative 24H funding:")
        print(f"    Mean: {stats['cum24h_mean']*10000:.2f} bps | Std: {stats['cum24h_std']*10000:.2f} bps")
        print(f"    P5-P95: [{stats['cum24h_p05']*10000:.2f}, {stats['cum24h_p95']*10000:.2f}] bps")

    return results


def cross_asset_analysis(results):
    """Analyze cross-asset funding dynamics"""
    print(f"\n{'='*70}")
    print("CROSS-ASSET FUNDING DYNAMICS")
    print(f"{'='*70}")

    # Load 4H funding for correlation analysis
    funding_4h = {}
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        path = DATA_DIR / f"{symbol}_funding.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            funding_4h[symbol] = df["funding_rate"].resample("4h").ffill()

    if len(funding_4h) < 2:
        print("  Insufficient data for cross-asset analysis")
        return

    # Create aligned dataframe
    aligned = pd.DataFrame(funding_4h)

    # Correlation matrix
    print("\n  Funding Rate Correlation Matrix:")
    corr = aligned.corr()
    print(corr.to_string())

    # Spread analysis: ETH - BTC
    if "ETHUSDT" in aligned.columns and "BTCUSDT" in aligned.columns:
        spread = aligned["ETHUSDT"] - aligned["BTCUSDT"]
        print(f"\n  ETH - BTC Funding Spread:")
        print(f"    Mean: {spread.mean()*10000:.2f} bps | Std: {spread.std()*10000:.2f} bps")
        print(f"    P5-P95: [{spread.quantile(0.05)*10000:.2f}, {spread.quantile(0.95)*10000:.2f}] bps")

        # When does ETH funding lead BTC?
        btc_ret = aligned["BTCUSDT"].pct_change()
        eth_spread_lag = spread.shift(1)

        # Split into quintiles
        quintiles = pd.qcut(eth_spread_lag.dropna(), q=5, labels=False, duplicates='drop')
        quintiles_aligned = quintiles.reindex(btc_ret.index)

        print(f"\n  24H BTC Funding Change by ETH-BTC Spread Quintile:")
        for q in range(5):
            mask = (quintiles_aligned == q)
            if mask.sum() > 10:
                fwd_change = aligned["BTCUSDT"].shift(-6) - aligned["BTCUSDT"]
                mean_change = fwd_change[mask].mean()
                print(f"    Q{q+1} (lowest to highest spread): {mean_change*10000:.2f} bps")


def recommend_thresholds(results):
    """Recommend empirically-derived thresholds for signal design"""
    print(f"\n{'='*70}")
    print("RECOMMENDED SIGNAL THRESHOLDS (DATA-DRIVEN)")
    print(f"{'='*70}")

    for symbol, stats in results.items():
        prefix = symbol.replace("USDT", "")
        print(f"\n{prefix}:")

        # Extreme regime threshold
        z_threshold = max(1.5, min(2.5, stats['zscore_p95']))  # Adaptive but bounded
        print(f"  Z-score extreme threshold: {z_threshold:.2f}σ ({stats['pct_extreme']:.1f}% of data)")

        # Cumulative funding threshold
        cum_extreme = stats['cum24h_p95']
        print(f"  Cumulative 24H extreme: {cum_extreme*10000:.2f} bps (95th percentile)")

        # Persistence requirement
        if stats['pct_3bar_persistence'] < 2.0:
            print(f"  ⚠️  3-bar persistence rare ({stats['pct_3bar_persistence']:.2f}%), use 2-bar instead")
        else:
            print(f"  ✓ 3-bar persistence occurs {stats['pct_3bar_persistence']:.2f}% of time")


if __name__ == "__main__":
    print("="*70)
    print("EMPIRICAL FUNDING RATE DISTRIBUTION ANALYSIS")
    print("="*70)

    results = load_and_analyze_funding()
    cross_asset_analysis(results)
    recommend_thresholds(results)

    print("\n" + "="*70)
    print("Analysis complete. Use these thresholds for signal design.")
    print("="*70)

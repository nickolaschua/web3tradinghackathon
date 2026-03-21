#!/usr/bin/env python3
"""
Backtest the Oil + DXY XGBoost model.

Usage:
    python scripts/backtest_oil_dxy.py --threshold 0.6 --fee-bps 10
    python scripts/backtest_oil_dxy.py --sweep  # threshold sweep
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs

# Configuration
DATA_DIR = Path("research_data")
MODEL_PATH = Path("models/xgb_oil_dxy.pkl")

OIL_DXY_FEATURES = [
    "oil_return_1d",
    "oil_return_5d",
    "oil_vol_5d",
    "oil_acceleration",
    "dxy_return_1d",
    "dxy_return_5d",
    "dxy_vol_5d",
    "dxy_acceleration",
]

PERIODS_4H = 2190  # Annualization for 4H crypto bars


def prepare_features(start=None, end=None):
    """
    Load data and compute Oil + DXY features.

    Args:
        start: Optional start date filter, e.g. "2024-01-01"
        end: Optional end date filter

    Returns:
        pd.DataFrame: Feature matrix with UTC DatetimeIndex
    """
    # Load BTC OHLCV
    btc_path = DATA_DIR / "BTCUSDT_4h.parquet"
    btc = pd.read_parquet(btc_path)
    btc.columns = btc.columns.str.lower()

    # Load macro data
    oil = pd.read_parquet(DATA_DIR / "oil_daily.parquet")
    dxy = pd.read_parquet(DATA_DIR / "dxy_daily.parquet")

    # Compute macro features
    feat = btc[["open", "high", "low", "close", "volume"]].copy()

    for name, mdf in [("oil", oil), ("dxy", dxy)]:
        close = mdf["close"].copy()
        close_4h = close.resample("4h").ffill()
        aligned = close_4h.reindex(feat.index, method="ffill")

        feat[f"{name}_return_1d"] = aligned.pct_change(6).shift(1)
        feat[f"{name}_return_5d"] = aligned.pct_change(30).shift(1)
        feat[f"{name}_vol_5d"] = aligned.pct_change().rolling(30).std().shift(1)
        r1d = aligned.pct_change(6)
        feat[f"{name}_acceleration"] = (r1d - r1d.shift(6)).shift(1)

    # Drop NaN rows
    feat = feat.dropna()

    # Filter by date
    if start:
        feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]

    return feat


def make_signal_fn(model, threshold: float):
    """Create signal function from model."""
    feature_cols = list(model.feature_names_in_)
    validated = [False]

    def signal_fn(row: pd.Series) -> str:
        if not validated[0]:
            missing = [c for c in feature_cols if c not in row.index]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            validated[0] = True

        X = pd.DataFrame([row[feature_cols]])
        proba_buy = model.predict_proba(X)[0][1]

        if proba_buy >= threshold:
            return "BUY"
        elif proba_buy <= (1.0 - threshold):
            return "SELL"
        return "HOLD"

    return signal_fn


def run_backtest(feat_df: pd.DataFrame, signal_fn, fee_bps: int = 0):
    """
    Bar-by-bar backtest simulation.

    Args:
        feat_df: Feature matrix with 'close' column
        signal_fn: callable(row: pd.Series) -> "BUY" | "SELL" | "HOLD"
        fee_bps: Transaction fee in basis points (each side)

    Returns:
        (returns_series, closed_trades)
    """
    fee_rate = fee_bps / 10_000

    position = 0
    entry_price = None
    entry_bar = None
    prev_close = None

    returns = []
    timestamps = []
    closed_trades = []

    for idx, row in feat_df.iterrows():
        close = row["close"]

        # Mark-to-market return
        bar_return = 0.0
        if position == 1 and prev_close is not None:
            bar_return = (close - prev_close) / prev_close

        signal = signal_fn(row)

        if signal == "BUY" and position == 0:
            position = 1
            entry_price = close * (1 + fee_rate)
            entry_bar = idx
            bar_return -= fee_rate
        elif signal == "SELL" and position == 1:
            exit_price = close * (1 - fee_rate)
            bar_return -= fee_rate
            pnl_pct = (exit_price - entry_price) / entry_price
            closed_trades.append({
                "entry_bar": entry_bar,
                "exit_bar": idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
            })
            position = 0
            entry_price = None
            entry_bar = None

        returns.append(bar_return)
        timestamps.append(idx)
        prev_close = close

    returns_series = pd.Series(returns, index=pd.DatetimeIndex(timestamps))
    return returns_series, closed_trades


def compute_stats(returns: pd.Series, closed_trades: list) -> dict:
    """Compute backtest statistics."""
    returns = returns[returns.index.notna()]

    n_trades = len(closed_trades)
    if n_trades > 0:
        winners = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
        trade_win_rate = winners / n_trades
        avg_trade_pnl = sum(t["pnl_pct"] for t in closed_trades) / n_trades
        best_trade = max(t["pnl_pct"] for t in closed_trades)
        worst_trade = min(t["pnl_pct"] for t in closed_trades)
    else:
        trade_win_rate = avg_trade_pnl = best_trade = worst_trade = 0.0

    return {
        "total_return_pct": float((1 + returns).prod() - 1) * 100,
        "cagr_pct": float(qs.stats.cagr(returns, periods=PERIODS_4H)) * 100,
        "sharpe": float(qs.stats.sharpe(returns, periods=PERIODS_4H)),
        "sortino": float(qs.stats.sortino(returns, periods=PERIODS_4H)),
        "calmar": float(qs.stats.calmar(returns)),
        "max_drawdown_pct": float(qs.stats.max_drawdown(returns)) * 100,
        "volatility_ann_pct": float(qs.stats.volatility(returns, periods=PERIODS_4H)) * 100,
        "n_trades": n_trades,
        "trade_win_rate_pct": trade_win_rate * 100,
        "avg_trade_pnl_pct": avg_trade_pnl * 100,
        "best_trade_pct": best_trade * 100,
        "worst_trade_pct": worst_trade * 100,
        "n_bars": len(returns),
    }


def print_stats(stats: dict, label: str = "") -> None:
    """Print formatted stats report."""
    sep = "=" * 60
    header = f"  BACKTEST RESULTS{' — ' + label if label else ''}"
    print(sep)
    print(header)
    print(sep)
    print(f"  Period bars      : {stats['n_bars']}")
    print(f"  Total return     : {stats['total_return_pct']:+.2f}%")
    print(f"  CAGR             : {stats['cagr_pct']:+.2f}%")
    print(f"  Sharpe (4H ann.) : {stats['sharpe']:.3f}")
    print(f"  Sortino (4H ann.): {stats['sortino']:.3f}")
    print(f"  Calmar           : {stats['calmar']:.3f}")
    print(f"  Max drawdown     : {stats['max_drawdown_pct']:.2f}%")
    print(f"  Volatility (ann.): {stats['volatility_ann_pct']:.2f}%")
    print(sep)
    print(f"  # Trades         : {stats['n_trades']}")
    print(f"  Trade win rate   : {stats['trade_win_rate_pct']:.1f}%")
    print(f"  Avg trade PnL    : {stats['avg_trade_pnl_pct']:+.2f}%")
    print(f"  Best trade       : {stats['best_trade_pct']:+.2f}%")
    print(f"  Worst trade      : {stats['worst_trade_pct']:+.2f}%")
    print(sep)


def run_threshold_sweep(feat_df, model, fee_bps):
    """Sweep thresholds and report best Sharpe."""
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    results = []

    for t in thresholds:
        signal_fn = make_signal_fn(model, t)
        returns, trades = run_backtest(feat_df, signal_fn, fee_bps=fee_bps)
        stats = compute_stats(returns, trades)
        results.append({
            "threshold": t,
            "sharpe": round(stats["sharpe"], 3),
            "sortino": round(stats["sortino"], 3),
            "calmar": round(stats["calmar"], 3),
            "n_trades": stats["n_trades"],
            "total_return_pct": round(stats["total_return_pct"], 2),
            "trade_win_rate_pct": round(stats["trade_win_rate_pct"], 1),
        })

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest Oil + DXY XGBoost model")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")
    parser.add_argument("--threshold", type=float, default=0.6, help="P(BUY) threshold")
    parser.add_argument("--fee-bps", type=int, default=10, help="Fee in basis points")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Features: {len(model.feature_names_in_)}")

    # Prepare features
    print("\nPreparing features...")
    feat = prepare_features(args.start, args.end)
    print(f"  Feature matrix: {feat.shape[0]} bars x {feat.shape[1]} columns")
    print(f"  Date range: {feat.index[0]} to {feat.index[-1]}")

    if args.sweep:
        # Threshold sweep
        print(f"\nRunning threshold sweep (fee={args.fee_bps} bps)...")
        results = run_threshold_sweep(feat, model, args.fee_bps)
        print(f"\n{'Threshold':>10}  {'Sharpe':>8}  {'Sortino':>8}  {'Calmar':>8}  "
              f"{'Trades':>7}  {'Return%':>9}  {'WinRate%':>9}")
        print("-" * 75)
        for r in results:
            marker = " <-- best" if r is results[0] else ""
            print(f"  {r['threshold']:>8.2f}  {r['sharpe']:>8.3f}  {r['sortino']:>8.3f}  "
                  f"{r['calmar']:>8.3f}  {r['n_trades']:>7}  "
                  f"{r['total_return_pct']:>9.2f}  {r['trade_win_rate_pct']:>8.1f}%"
                  f"{marker}")
        best = results[0]
        print(f"\nBest threshold: {best['threshold']} (Sharpe={best['sharpe']:.3f})")
    else:
        # Single threshold backtest
        print(f"\nRunning backtest (threshold={args.threshold}, fee={args.fee_bps}bps)...")
        signal_fn = make_signal_fn(model, args.threshold)
        returns, trades = run_backtest(feat, signal_fn, fee_bps=args.fee_bps)
        print(f"  Simulation complete: {len(trades)} closed trades")

        stats = compute_stats(returns, trades)
        label = f"Oil+DXY threshold={args.threshold} fee={args.fee_bps}bps"
        print()
        print_stats(stats, label)


if __name__ == "__main__":
    main()

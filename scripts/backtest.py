"""
Backtest runner for XGBoost model and rule-based strategies.

Usage:
  python scripts/backtest.py --model models/xgb_btc_4h.pkl --fee-bps 10
  python scripts/backtest.py --model models/xgb_btc_4h.pkl --sweep  # threshold sweep
  python scripts/backtest.py --strategy momentum --fee-bps 10
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import quantstats as qs
import xgboost

# Add project root to path so bot package can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_features, compute_cross_asset_features


def prepare_features(btc_path, eth_path, sol_path, start=None, end=None):
    """
    Load Parquet files, run feature pipeline, and return date-filtered feature matrix.

    Pipeline order (CRITICAL — from Phase 04-03):
    1. compute_features(btc) on BTC data
    2. compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    3. dropna() to remove warmup rows

    Args:
        btc_path: Path to BTCUSDT_4h.parquet
        eth_path: Path to ETHUSDT_4h.parquet
        sol_path: Path to SOLUSDT_4h.parquet
        start: Optional start date filter, e.g. "2024-01-01"
        end: Optional end date filter, e.g. "2024-12-31"

    Returns:
        pd.DataFrame: Feature matrix with UTC DatetimeIndex, ready for model inference
    """
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)
    sol = pd.read_parquet(sol_path)

    for df in (btc, eth, sol):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    feat = compute_features(btc)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    feat = feat.dropna()

    if start:
        feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]

    return feat


def load_model(model_path: str):
    """Load a pre-trained XGBoost model from a .pkl file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model at {model_path} does not have predict_proba(). "
            "Expected XGBClassifier."
        )
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            f"Model at {model_path} has no feature_names_in_. "
            "Retrain with a named DataFrame X."
        )
    return model


def make_model_signal_fn(model, threshold: float):
    """
    Return a signal_fn wrapping model.predict_proba at the given threshold.

    signal_fn(row: pd.Series) -> "BUY" | "SELL" | "HOLD"
    """
    feature_cols = list(model.feature_names_in_)
    validated = [False]

    def signal_fn(row: pd.Series) -> str:
        if not validated[0]:
            missing = [c for c in feature_cols if c not in row.index]
            if missing:
                raise ValueError(f"Feature matrix missing model columns: {missing[:5]}")
            validated[0] = True

        X = pd.DataFrame([row[feature_cols]])
        proba_buy = model.predict_proba(X)[0][1]
        if proba_buy >= threshold:
            return "BUY"
        elif proba_buy <= (1.0 - threshold):
            return "SELL"
        return "HOLD"

    return signal_fn


def make_strategy_signal_fn(strategy_name: str, pair: str):
    """
    Return a signal_fn wrapping a BaseStrategy.generate_signal call.

    Maintains a growing history of feature rows so strategies can use
    features.iloc[-1] as usual.

    signal_fn(row: pd.Series) -> "BUY" | "SELL" | "HOLD"
    """
    from bot.strategy.momentum import MomentumStrategy

    registry = {
        "momentum": MomentumStrategy,
    }
    if strategy_name not in registry:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Available: {list(registry.keys())}"
        )

    strategy = registry[strategy_name]()
    history: list[pd.Series] = []

    def signal_fn(row: pd.Series) -> str:
        history.append(row)
        feat_window = pd.DataFrame(history)
        sig = strategy.generate_signal(pair, feat_window)
        return sig.direction.name  # "BUY", "SELL", or "HOLD"

    return signal_fn


def run_backtest(
    feat_df: pd.DataFrame,
    signal_fn,
    fee_bps: int = 0,
) -> tuple:
    """
    Bar-by-bar backtest simulation.

    Args:
        feat_df: Feature matrix from prepare_features() — UTC DatetimeIndex,
                 must include a 'close' column.
        signal_fn: callable(row: pd.Series) -> "BUY" | "SELL" | "HOLD"
                   Use make_model_signal_fn() or make_strategy_signal_fn().
        fee_bps: Transaction fee in basis points (applied each side: entry + exit).

    Returns:
        (returns_series, closed_trades):
        - returns_series: pd.Series with DatetimeIndex, bar-by-bar returns
        - closed_trades: list of dicts {'entry_bar', 'exit_bar', 'entry_price',
                         'exit_price', 'pnl_pct'}
    """
    fee_rate = fee_bps / 10_000

    position = 0        # 0 = flat, 1 = long
    entry_price = None
    entry_bar = None
    prev_close = None

    returns = []
    timestamps = []
    closed_trades = []

    for idx, row in feat_df.iterrows():
        close = row["close"]

        # Mark-to-market bar return for current position
        bar_return = 0.0
        if position == 1 and prev_close is not None:
            bar_return = (close - prev_close) / prev_close

        signal = signal_fn(row)

        if signal == "BUY" and position == 0:
            position = 1
            entry_price = close * (1 + fee_rate)
            entry_bar = idx
            bar_return -= fee_rate  # entry fee deducted from this bar's return
        elif signal == "SELL" and position == 1:
            exit_price = close * (1 - fee_rate)
            bar_return -= fee_rate  # exit fee deducted from this bar's return
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


PERIODS_4H = 2190  # 365.25 * 24 / 4 — correct annualization for 4H crypto bars


def compute_stats_report(returns: pd.Series, closed_trades: list[dict]) -> dict:
    """
    Compute comprehensive backtest statistics.

    Args:
        returns: Bar-by-bar returns Series with DatetimeIndex (from run_backtest).
        closed_trades: List of trade dicts with pnl_pct (from run_backtest).

    Returns:
        Dict of stats suitable for printing or CSV export.
    """
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

    stats = {
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
    return stats


def run_threshold_sweep(feat_df: pd.DataFrame, model, fee_bps: int = 10) -> list[dict]:
    """
    Sweep BUY probability thresholds and report Sharpe/Sortino/trades for each.

    Args:
        feat_df: Feature matrix from prepare_features().
        model: Loaded XGBClassifier from load_model().
        fee_bps: Fee in basis points applied to each side.

    Returns:
        List of dicts sorted by descending Sharpe, one row per threshold.
    """
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    results = []

    for t in thresholds:
        signal_fn = make_model_signal_fn(model, t)
        returns, trades = run_backtest(feat_df, signal_fn, fee_bps=fee_bps)
        stats = compute_stats_report(returns, trades)
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Backtest XGBoost model or rule-based strategy on historical data"
    )
    # --- Signal source (one of --model or --strategy required) ---
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        type=str,
        metavar="PATH",
        help="Path to pre-trained XGBoost .pkl file",
    )
    source.add_argument(
        "--strategy",
        type=str,
        choices=["momentum"],
        metavar="NAME",
        help="Rule-based strategy name (e.g. momentum)",
    )
    # --- Data paths ---
    parser.add_argument("--btc", default="data/BTCUSDT_4h.parquet")
    parser.add_argument("--eth", default="data/ETHUSDT_4h.parquet")
    parser.add_argument("--sol", default="data/SOLUSDT_4h.parquet")
    # --- Date range ---
    parser.add_argument("--start", default=None, help='Start date, e.g. "2024-01-01"')
    parser.add_argument("--end", default=None, help='End date, e.g. "2024-12-31"')
    # --- Model options ---
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="[--model only] P(BUY) threshold (default 0.6); SELL = 1 - threshold",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="[--model only] Sweep thresholds 0.50–0.85 and report best Sharpe",
    )
    # --- Shared options ---
    parser.add_argument(
        "--fee-bps", type=int, default=10,
        help="Transaction cost in basis points per side (default 10 = 0.1%%)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write stats report as CSV to this path",
    )
    return parser.parse_args()


def print_stats_report(stats: dict, label: str = "") -> None:
    """Print formatted stats report to stdout."""
    sep = "=" * 55
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


def main():
    args = parse_args()

    # Step 1: Feature prep
    print("Loading and preparing features...")
    feat = prepare_features(args.btc, args.eth, args.sol, args.start, args.end)
    print(f"  Feature matrix: {feat.shape[0]} bars x {feat.shape[1]} columns")
    print(f"  Date range: {feat.index[0]} to {feat.index[-1]}")

    # --- Model path ---
    if args.model:
        print(f"Loading model from {args.model}...")
        model = load_model(args.model)
        print(f"  Model expects {len(model.feature_names_in_)} features")

        # Threshold sweep mode
        if args.sweep:
            print(f"\nRunning threshold sweep (fee={args.fee_bps} bps)...")
            results = run_threshold_sweep(feat, model, fee_bps=args.fee_bps)
            print(f"\n{'Threshold':>10}  {'Sharpe':>8}  {'Sortino':>8}  {'Calmar':>8}  "
                  f"{'Trades':>7}  {'Return%':>9}  {'WinRate%':>9}")
            print("-" * 72)
            for r in results:
                marker = " <-- best" if r is results[0] else ""
                print(f"  {r['threshold']:>8.2f}  {r['sharpe']:>8.3f}  {r['sortino']:>8.3f}  "
                      f"{r['calmar']:>8.3f}  {r['n_trades']:>7}  "
                      f"{r['total_return_pct']:>9.2f}  {r['trade_win_rate_pct']:>8.1f}%"
                      f"{marker}")
            best = results[0]
            print(f"\nBest threshold: {best['threshold']} (Sharpe={best['sharpe']:.3f})")
            return

        # Single-threshold backtest
        print("Running bar-by-bar simulation...")
        signal_fn = make_model_signal_fn(model, args.threshold)
        returns, trades = run_backtest(feat, signal_fn, fee_bps=args.fee_bps)
        print(f"  Simulation complete: {len(trades)} closed trades")

        stats = compute_stats_report(returns, trades)
        label = f"XGBoost threshold={args.threshold} fee={args.fee_bps}bps"
        print_stats_report(stats, label)

    # --- Strategy path ---
    else:
        print(f"Running strategy: {args.strategy}...")
        signal_fn = make_strategy_signal_fn(args.strategy, "BTC/USD")
        returns, trades = run_backtest(feat, signal_fn, fee_bps=args.fee_bps)
        print(f"  Simulation complete: {len(trades)} closed trades")

        stats = compute_stats_report(returns, trades)
        label = f"{args.strategy} fee={args.fee_bps}bps"
        print_stats_report(stats, label)

    # Optional CSV output
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        print(f"\nStats saved to {args.output}")


if __name__ == "__main__":
    main()

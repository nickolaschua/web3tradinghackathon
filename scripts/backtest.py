"""
Backtest runner for XGBoost trading model.

Loads pre-trained model, runs feature pipeline matching live bot,
and generates trading signals + performance metrics.
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
    # Load Parquet files
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)
    sol = pd.read_parquet(sol_path)

    # Ensure DatetimeIndex
    if not isinstance(btc.index, pd.DatetimeIndex):
        btc.index = pd.to_datetime(btc.index)
    if not isinstance(eth.index, pd.DatetimeIndex):
        eth.index = pd.to_datetime(eth.index)
    if not isinstance(sol.index, pd.DatetimeIndex):
        sol.index = pd.to_datetime(sol.index)

    # Ensure lowercase columns
    btc.columns = btc.columns.str.lower()
    eth.columns = eth.columns.str.lower()
    sol.columns = sol.columns.str.lower()

    # Run feature pipeline in exact order
    feat = compute_features(btc)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    feat = feat.dropna()

    # Apply date range filter if provided
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


def run_backtest(
    feat_df: pd.DataFrame,
    model,
    threshold: float = 0.6,
    fee_bps: int = 0,
) -> tuple:
    """
    Bar-by-bar backtest simulation.

    Args:
        feat_df: Feature matrix from prepare_features() — UTC DatetimeIndex, includes 'close' column.
        model: Loaded XGBClassifier from load_model().
        threshold: Probability >= threshold → BUY; <= (1-threshold) → SELL; else HOLD.
        fee_bps: Transaction fee in basis points (applied each side: entry + exit).

    Returns:
        (returns_series, closed_trades):
        - returns_series: pd.Series with pd.DatetimeIndex, bar-by-bar returns (0.0 when flat)
        - closed_trades: list of dicts {'entry_bar', 'exit_bar', 'entry_price', 'exit_price', 'pnl_pct'}
    """
    feature_cols = list(model.feature_names_in_)  # safe column order from training
    fee_rate = fee_bps / 10_000  # e.g. 10 bps → 0.001

    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        raise ValueError(
            f"Feature matrix missing columns expected by model: {missing[:5]}..."
        )

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

        # Generate signal — pass named DataFrame to preserve column validation
        X = pd.DataFrame([row[feature_cols]])
        proba_buy = model.predict_proba(X)[0][1]  # P(class=1 = BUY)

        if proba_buy >= threshold:
            signal = "BUY"
        elif proba_buy <= (1.0 - threshold):
            signal = "SELL"
        else:
            signal = "HOLD"

        # Position state machine — explicit three-way, HOLD does nothing
        if signal == "BUY" and position == 0:
            position = 1
            entry_price = close * (1 + fee_rate)
            entry_bar = idx
        elif signal == "SELL" and position == 1:
            exit_price = close * (1 - fee_rate)
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
    # Guard against NaT index entries (quantstats requirement)
    returns = returns[returns.index.notna()]

    n_trades = len(closed_trades)

    # Trade-based win rate — quantstats win_rate() is bar-based, not trade-based
    if n_trades > 0:
        winners = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
        trade_win_rate = winners / n_trades
        avg_trade_pnl = sum(t["pnl_pct"] for t in closed_trades) / n_trades
        best_trade = max(t["pnl_pct"] for t in closed_trades)
        worst_trade = min(t["pnl_pct"] for t in closed_trades)
    else:
        trade_win_rate = 0.0
        avg_trade_pnl = 0.0
        best_trade = 0.0
        worst_trade = 0.0

    # Financial metrics via quantstats — always pass periods=PERIODS_4H
    # All handle edge cases (zero std, single period) correctly
    stats = {
        "total_return_pct": float((1 + returns).prod() - 1) * 100,
        "cagr_pct": float(qs.stats.cagr(returns, periods=PERIODS_4H)) * 100,
        "sharpe": float(qs.stats.sharpe(returns, periods=PERIODS_4H)),
        "sortino": float(qs.stats.sortino(returns, periods=PERIODS_4H)),
        "max_drawdown_pct": float(qs.stats.max_drawdown(returns)) * 100,  # negative number
        "volatility_ann_pct": float(qs.stats.volatility(returns, periods=PERIODS_4H)) * 100,
        "n_trades": n_trades,
        "trade_win_rate_pct": trade_win_rate * 100,
        "avg_trade_pnl_pct": avg_trade_pnl * 100,
        "best_trade_pct": best_trade * 100,
        "worst_trade_pct": worst_trade * 100,
        "n_bars": len(returns),
    }
    return stats


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Backtest XGBoost trading model on historical data"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to pre-trained XGBoost .pkl file",
    )
    parser.add_argument(
        "--btc",
        type=str,
        default="data/BTCUSDT_4h.parquet",
        help="BTC Parquet path",
    )
    parser.add_argument(
        "--eth",
        type=str,
        default="data/ETHUSDT_4h.parquet",
        help="ETH Parquet path",
    )
    parser.add_argument(
        "--sol",
        type=str,
        default="data/SOLUSDT_4h.parquet",
        help="SOL Parquet path",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help='Start date filter, e.g. "2024-01-01"',
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help='End date filter, e.g. "2024-12-31"',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Probability threshold for BUY signal; SELL threshold = 1.0 - threshold",
    )
    parser.add_argument(
        "--fee-bps",
        type=int,
        default=0,
        help="Transaction cost in basis points (0 = free)",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        default=False,
        help="If set, ignore SELL signals (exit on SELL, don't go short)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, write stats report as CSV to this path",
    )
    return parser.parse_args()


def print_stats_report(stats: dict, args) -> None:
    """Print formatted stats report to stdout."""
    sep = "=" * 55
    print(sep)
    print("  BACKTEST RESULTS")
    print(sep)
    print(f"  Period bars      : {stats['n_bars']}")
    print(f"  Total return     : {stats['total_return_pct']:+.2f}%")
    print(f"  CAGR             : {stats['cagr_pct']:+.2f}%")
    print(f"  Sharpe (4H ann.) : {stats['sharpe']:.3f}")
    print(f"  Sortino (4H ann.): {stats['sortino']:.3f}")
    print(f"  Max drawdown     : {stats['max_drawdown_pct']:.2f}%")
    print(f"  Volatility (ann.): {stats['volatility_ann_pct']:.2f}%")
    print(sep)
    print(f"  # Trades         : {stats['n_trades']}")
    print(f"  Trade win rate   : {stats['trade_win_rate_pct']:.1f}%")
    print(f"  Avg trade PnL    : {stats['avg_trade_pnl_pct']:+.2f}%")
    print(f"  Best trade       : {stats['best_trade_pct']:+.2f}%")
    print(f"  Worst trade      : {stats['worst_trade_pct']:+.2f}%")
    print(sep)
    print(f"  Threshold        : {args.threshold}")
    print(f"  Fee (bps)        : {args.fee_bps}")
    print(sep)


def main():
    """Load features, run model inference, and print simulation summary."""
    args = parse_args()

    # Step 1: Feature prep
    print("Loading and preparing features...")
    feat = prepare_features(args.btc, args.eth, args.sol, args.start, args.end)
    print(f"  Feature matrix: {feat.shape[0]} bars × {feat.shape[1]} columns")
    print(f"  Date range: {feat.index[0]} to {feat.index[-1]}")

    # Step 2: Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    print(f"  Model expects {len(model.feature_names_in_)} features")

    # Step 3: Run simulation
    print("Running bar-by-bar simulation...")
    returns, trades = run_backtest(feat, model, threshold=args.threshold,
                                   fee_bps=args.fee_bps)
    print(f"  Simulation complete: {len(trades)} closed trades")

    # Step 4: Compute and print stats
    stats = compute_stats_report(returns, trades)
    print_stats_report(stats, args)

    # Step 5: Optional CSV output
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        print(f"\nStats saved to {args.output}")


if __name__ == "__main__":
    main()

"""
Backtest runner for XGBoost trading model.

Loads pre-trained model, runs feature pipeline matching live bot,
and generates trading signals + performance metrics.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

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


def main():
    """Load features and print shape for validation."""
    args = parse_args()
    feat = prepare_features(args.btc, args.eth, args.sol, args.start, args.end)
    print(f"Feature matrix: {feat.shape[0]} bars x {feat.shape[1]} columns")
    print(f"Date range: {feat.index[0]} to {feat.index[-1]}")


if __name__ == "__main__":
    main()

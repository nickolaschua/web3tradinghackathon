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
    """Load features, run model inference, and print simulation summary."""
    args = parse_args()
    feat = prepare_features(args.btc, args.eth, args.sol, args.start, args.end)
    print(f"Feature matrix: {feat.shape[0]} bars × {feat.shape[1]} columns")
    print(f"Date range: {feat.index[0]} to {feat.index[-1]}")
    model = load_model(args.model)
    print(f"Model loaded. Expects {len(model.feature_names_in_)} features.")
    returns, trades = run_backtest(
        feat, model, threshold=args.threshold, fee_bps=args.fee_bps
    )
    print(
        f"Simulation complete. {len(trades)} closed trades. "
        f"Total return: {(1 + returns).prod() - 1:.2%}"
    )


if __name__ == "__main__":
    main()

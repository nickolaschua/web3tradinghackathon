#!/usr/bin/env python3
"""
Multi-asset XGBoost portfolio backtest for high intraday frequency.

Uses 3 XGBoost models (BTC, ETH, SOL) with lower thresholds for more trades.

Target: Achieve higher daily coverage through multiple assets + lower thresholds.

Usage:
  python scripts/backtest_multi_xgboost.py --threshold 0.55
  python scripts/backtest_multi_xgboost.py --threshold 0.50 --exit-threshold 0.15
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_features, compute_cross_asset_features, compute_btc_context_features
from bot.execution.risk import RiskManager, RiskDecision
from bot.strategy.base import TradingSignal, SignalDirection

MODELS_DIR = Path("models")
RESULTS_DIR = Path("research_results")
RESULTS_DIR.mkdir(exist_ok=True)


class MultiAssetXGBoostStrategy:
    """
    Run multiple XGBoost models simultaneously for higher trade frequency.

    Each model (BTC, ETH, SOL) can generate independent signals.
    """

    def __init__(self, threshold: float = 0.55, exit_threshold: float = 0.15):
        self.threshold = threshold
        self.exit_threshold = exit_threshold

        # Load models
        self.models = {}
        for asset in ["btc", "eth", "sol"]:
            model_path = MODELS_DIR / f"xgb_{asset}_15m.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.models[asset.upper()] = pickle.load(f)
                print(f"✅ Loaded {asset.upper()} model: {model_path}")
            else:
                print(f"⚠️  {asset.upper()} model not found: {model_path}")

        print(f"Loaded {len(self.models)} models with threshold={threshold}")

    def generate_signals(self, features_dict: dict) -> dict:
        """
        Generate signals for all assets.

        Args:
            features_dict: {"BTC": df, "ETH": df, "SOL": df}

        Returns:
            {"BTC": signal, "ETH": signal, "SOL": signal}
        """
        signals = {}

        for asset, model in self.models.items():
            if asset not in features_dict:
                signals[asset] = TradingSignal(pair=f"{asset}/USD", direction=SignalDirection.HOLD)
                continue

            features = features_dict[asset]
            if features.empty:
                signals[asset] = TradingSignal(pair=f"{asset}/USD", direction=SignalDirection.HOLD)
                continue

            # Get latest row
            last = features.iloc[[-1]].copy()

            # Extract features model expects
            feature_cols = list(model.feature_names_in_)
            missing = [c for c in feature_cols if c not in last.columns]

            if missing:
                # Fill missing with NaN (XGBoost handles this)
                for col in missing:
                    last[col] = float("nan")

            row = last[feature_cols]

            # Predict
            try:
                proba = float(model.predict_proba(row)[0, 1])
            except Exception as e:
                print(f"⚠️  {asset} prediction failed: {e}")
                signals[asset] = TradingSignal(pair=f"{asset}/USD", direction=SignalDirection.HOLD)
                continue

            # Generate signal
            if proba >= self.threshold:
                signals[asset] = TradingSignal(
                    pair=f"{asset}/USD",
                    direction=SignalDirection.BUY,
                    size=1.0,
                    confidence=min(proba, 1.0)
                )
            elif proba <= self.exit_threshold:
                signals[asset] = TradingSignal(
                    pair=f"{asset}/USD",
                    direction=SignalDirection.SELL,
                    size=1.0,
                    confidence=min(proba, 1.0)
                )
            else:
                signals[asset] = TradingSignal(pair=f"{asset}/USD", direction=SignalDirection.HOLD)

        return signals


def prepare_multi_asset_features(btc_path, eth_path, sol_path, start, end):
    """Prepare features for BTC, ETH, SOL."""

    # Load data
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)
    sol = pd.read_parquet(sol_path)

    for df in [btc, eth, sol]:
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    # Filter by date
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    btc = btc[(btc.index >= start_ts) & (btc.index < end_ts)]
    eth = eth[(eth.index >= start_ts) & (eth.index < end_ts)]
    sol = sol[(sol.index >= start_ts) & (sol.index < end_ts)]

    # Compute features for each asset
    print("Computing BTC features...")
    btc_feat = compute_features(btc)
    btc_feat = compute_cross_asset_features(btc_feat, {"ETH/USD": eth, "SOL/USD": sol})

    # Add cross-asset lags
    for asset, df in [("eth", eth), ("sol", sol)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        btc_feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(btc_feat.index)
        btc_feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(btc_feat.index)

    btc_feat = compute_btc_context_features(btc_feat, eth, sol, window=2880)
    btc_feat = btc_feat.dropna()

    print("Computing ETH features...")
    eth_feat = compute_features(eth)
    eth_feat = compute_cross_asset_features(eth_feat, {"BTC/USD": btc, "SOL/USD": sol})

    # Add cross-asset lags for ETH
    for asset, df in [("btc", btc), ("sol", sol)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        eth_feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(eth_feat.index)
        eth_feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(eth_feat.index)

    # ETH-BTC correlation and beta
    eth_ret = np.log(eth["close"] / eth["close"].shift(1)).reindex(eth_feat.index)
    btc_ret = np.log(btc["close"] / btc["close"].shift(1)).reindex(eth_feat.index)
    corr = eth_ret.rolling(2880).corr(btc_ret)
    cov = eth_ret.rolling(2880).cov(btc_ret)
    var_btc = btc_ret.rolling(2880).var()
    eth_feat["eth_btc_corr"] = corr.shift(1)
    eth_feat["eth_btc_beta"] = (cov / (var_btc + 1e-10)).shift(1)
    eth_feat = eth_feat.dropna()

    print("Computing SOL features...")
    sol_feat = compute_features(sol)
    sol_feat = compute_cross_asset_features(sol_feat, {"BTC/USD": btc, "ETH/USD": eth})

    # Add cross-asset lags for SOL
    for asset, df in [("btc", btc), ("eth", eth)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        sol_feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(sol_feat.index)
        sol_feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(sol_feat.index)

    # SOL-BTC correlation and beta
    sol_ret = np.log(sol["close"] / sol["close"].shift(1)).reindex(sol_feat.index)
    btc_ret = np.log(btc["close"] / btc["close"].shift(1)).reindex(sol_feat.index)
    corr = sol_ret.rolling(2880).corr(btc_ret)
    cov = sol_ret.rolling(2880).cov(btc_ret)
    var_btc = btc_ret.rolling(2880).var()
    sol_feat["sol_btc_corr"] = corr.shift(1)
    sol_feat["sol_btc_beta"] = (cov / (var_btc + 1e-10)).shift(1)
    sol_feat = sol_feat.dropna()

    return {
        "BTC": btc_feat,
        "ETH": eth_feat,
        "SOL": sol_feat
    }


def run_multi_asset_backtest(features_dict, strategy, fee_bps=10.0):
    """
    Run backtest trading multiple assets simultaneously.

    Returns aggregate portfolio metrics.
    """
    risk_mgr = RiskManager({})

    # Track state per asset
    cash = 100_000.0
    positions = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}
    entry_prices = {"BTC": None, "ETH": None, "SOL": None}

    closed_trades = []
    equity_curve = []
    days_with_trades = set()

    risk_mgr.initialize_hwm(cash)
    fee_rate = fee_bps / 10000

    # Align all timeframes to common index
    common_index = features_dict["BTC"].index

    print(f"\nBacktesting {len(common_index)} bars...")

    for idx in common_index:
        # Get current prices
        prices = {}
        for asset in ["BTC", "ETH", "SOL"]:
            if idx in features_dict[asset].index:
                prices[asset] = features_dict[asset].loc[idx, "close"]
            else:
                prices[asset] = None

        # Calculate portfolio value
        position_value = sum(
            positions[asset] * prices[asset]
            for asset in ["BTC", "ETH", "SOL"]
            if prices[asset] is not None
        )
        portfolio_value = cash + position_value
        equity_curve.append(portfolio_value)

        date = idx.date()

        # Check stops for each asset
        for asset in ["BTC", "ETH", "SOL"]:
            if positions[asset] > 0 and prices[asset] is not None:
                atr = features_dict[asset].loc[idx, "atr_proxy"] if "atr_proxy" in features_dict[asset].columns else prices[asset] * 0.02

                stop_result = risk_mgr.check_stops(f"{asset}/USD", prices[asset], atr)
                if stop_result.should_exit:
                    # Exit position
                    exit_value = positions[asset] * prices[asset] * (1 - fee_rate)
                    pnl_pct = (prices[asset] - entry_prices[asset]) / entry_prices[asset] if entry_prices[asset] else 0

                    closed_trades.append({
                        "asset": asset,
                        "ts": idx,
                        "entry_price": entry_prices[asset],
                        "exit_price": prices[asset],
                        "pnl_pct": pnl_pct,
                        "exit_reason": stop_result.exit_reason,
                    })

                    cash += exit_value
                    positions[asset] = 0.0
                    entry_prices[asset] = None
                    risk_mgr.record_exit(f"{asset}/USD")
                    days_with_trades.add(date)

        # Generate signals for all assets
        current_features = {
            asset: features_dict[asset].loc[:idx]
            for asset in ["BTC", "ETH", "SOL"]
        }
        signals = strategy.generate_signals(current_features)

        # Handle signals for each asset
        for asset, signal in signals.items():
            if prices[asset] is None:
                continue

            pair = f"{asset}/USD"

            # BUY signal
            if signal.direction == SignalDirection.BUY and positions[asset] == 0:
                atr = features_dict[asset].loc[idx, "atr_proxy"] if "atr_proxy" in features_dict[asset].columns else prices[asset] * 0.02

                # Size position through RiskManager
                open_positions = {
                    f"{a}/USD": positions[a] * prices[a]
                    for a in ["BTC", "ETH", "SOL"]
                    if positions[a] > 0 and prices[a] is not None
                }

                sizing = risk_mgr.size_new_position(
                    pair=pair,
                    current_price=prices[asset],
                    current_atr=atr,
                    free_balance_usd=cash,
                    open_positions=open_positions,
                    regime_multiplier=1.0,
                    confidence=signal.confidence,
                    portfolio_weight=1.0,
                )

                if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity > 0:
                    # Enter position
                    positions[asset] = sizing.approved_quantity
                    entry_cost = positions[asset] * prices[asset] * (1 + fee_rate)

                    if entry_cost <= cash:
                        cash -= entry_cost
                        entry_prices[asset] = prices[asset] * (1 + fee_rate)
                        risk_mgr.record_entry(pair, entry_prices[asset], sizing.trailing_stop_price)
                        days_with_trades.add(date)
                    else:
                        positions[asset] = 0.0

            # SELL signal
            elif signal.direction == SignalDirection.SELL and positions[asset] > 0:
                exit_value = positions[asset] * prices[asset] * (1 - fee_rate)
                pnl_pct = (prices[asset] - entry_prices[asset]) / entry_prices[asset] if entry_prices[asset] else 0

                closed_trades.append({
                    "asset": asset,
                    "ts": idx,
                    "entry_price": entry_prices[asset],
                    "exit_price": prices[asset],
                    "pnl_pct": pnl_pct,
                    "exit_reason": "signal_exit",
                })

                cash += exit_value
                positions[asset] = 0.0
                entry_prices[asset] = None
                risk_mgr.record_exit(pair)
                days_with_trades.add(date)

    # Calculate metrics
    final_capital = cash + sum(
        positions[asset] * features_dict[asset].iloc[-1]["close"]
        for asset in ["BTC", "ETH", "SOL"]
        if positions[asset] > 0
    )
    total_return = (final_capital / 100_000.0) - 1.0

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 96) if len(returns) > 0 and returns.std() > 0 else 0.0

    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std()) * np.sqrt(252 * 96) if len(downside) > 0 and downside.std() > 0 else 0.0

    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()

    calmar = total_return / abs(max_dd) if max_dd < 0 else 0.0
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    # Win rate
    if len(closed_trades) > 0:
        wins = [t for t in closed_trades if t["pnl_pct"] > 0]
        win_rate = len(wins) / len(closed_trades)
    else:
        win_rate = 0.0

    # Daily coverage
    total_days = (common_index[-1].date() - common_index[0].date()).days
    daily_coverage_pct = len(days_with_trades) / total_days if total_days > 0 else 0.0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "composite_score": composite,
        "num_trades": len(closed_trades),
        "win_rate": win_rate,
        "daily_coverage_pct": daily_coverage_pct,
        "trades": closed_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-asset XGBoost backtest")
    parser.add_argument("--threshold", type=float, default=0.55, help="Buy threshold (default 0.55)")
    parser.add_argument("--exit-threshold", type=float, default=0.15, help="Exit threshold (default 0.15)")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2026-03-01", help="End date")
    args = parser.parse_args()

    print("="*70)
    print("  Multi-Asset XGBoost Portfolio Backtest")
    print("="*70)
    print(f"Threshold: {args.threshold} (lower = more trades)")
    print(f"Exit Threshold: {args.exit_threshold}")
    print()

    # Load features
    features_dict = prepare_multi_asset_features(
        "research_data/BTCUSDT_15m.parquet",
        "research_data/ETHUSDT_15m.parquet",
        "research_data/SOLUSDT_15m.parquet",
        args.start,
        args.end
    )

    # Create strategy
    strategy = MultiAssetXGBoostStrategy(
        threshold=args.threshold,
        exit_threshold=args.exit_threshold
    )

    # Run backtest
    result = run_multi_asset_backtest(features_dict, strategy)

    # Print results
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    print(f"Total Return: {result['total_return']*100:+.2f}%")
    print(f"Sharpe: {result['sharpe']:.3f}")
    print(f"Sortino: {result['sortino']:.3f}")
    print(f"Calmar: {result['calmar']:.3f}")
    print(f"Composite Score: {result['composite_score']:.3f}")
    print(f"Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"Trades: {result['num_trades']}")
    print(f"Win Rate: {result['win_rate']*100:.1f}%")
    print(f"Daily Coverage: {result['daily_coverage_pct']*100:.1f}%")
    print()

    # Save results
    output_path = RESULTS_DIR / f"multi_xgb_t{args.threshold:.2f}_15m.json"
    with open(output_path, "w") as f:
        result_copy = {k: v for k, v in result.items() if k != "trades"}
        result_copy["trades"] = [
            {"asset": t["asset"], "ts": str(t["ts"]), "pnl_pct": t["pnl_pct"], "reason": t["exit_reason"]}
            for t in result["trades"][:100]  # Save first 100 trades only
        ]
        json.dump(result_copy, f, indent=2, default=str)

    print(f"✅ Results saved: {output_path}")


if __name__ == "__main__":
    main()

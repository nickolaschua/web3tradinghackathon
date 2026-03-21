#!/usr/bin/env python3
"""
Backtest XGBoost trading models.

Uses trained XGBoost models to generate trading signals, then backtests with RiskManager framework.

Usage:
  python scripts/backtest_xgboost.py --strategy all
  python scripts/backtest_xgboost.py --strategy mean_reversion
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

from bot.data.features_15m import prepare_15m_features
from bot.execution.risk import RiskManager, RiskDecision
from bot.strategy.base import TradingSignal, SignalDirection

MODELS_DIR = Path("models")
RESULTS_DIR = Path("research_results")
RESULTS_DIR.mkdir(exist_ok=True)


class XGBoostStrategy:
    """
    Wrapper for XGBoost models to use with backtest framework.

    Loads trained model and generates TradingSignal based on predictions.
    """

    def __init__(self, model_path: Path, threshold: float = 0.5):
        """
        Args:
            model_path: Path to pickled model dict
            threshold: Probability threshold for BUY signal (default 0.5)
        """
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.strategy_name = data["strategy"]
        self.threshold = threshold

        print(f"Loaded XGBoost model: {self.strategy_name}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Threshold: {self.threshold}")

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal using XGBoost model prediction.

        Args:
            pair: Trading pair
            features: DataFrame with feature columns

        Returns:
            TradingSignal (BUY, SELL, or HOLD)
        """
        if features.empty:
            return TradingSignal(pair=pair, direction=SignalDirection.HOLD)

        # Get latest features
        latest = features.iloc[-1]

        # Extract features for model
        X = []
        for feat in self.feature_names:
            if feat in features.columns:
                X.append(latest.get(feat, 0.0))
            else:
                X.append(0.0)

        X = np.array(X).reshape(1, -1)

        # Check for NaN/Inf
        if np.isnan(X).any() or np.isinf(X).any():
            return TradingSignal(pair=pair, direction=SignalDirection.HOLD)

        # Predict probability
        proba = self.model.predict_proba(X)[0, 1]  # Probability of BUY (class 1)

        # Generate signal based on threshold
        if proba >= self.threshold:
            # Confidence = how far above threshold (0.5 to 1.0 scale)
            confidence = 0.5 + (proba - self.threshold) * (0.5 / (1.0 - self.threshold))
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=0.4,  # Base size, RiskManager will adjust
                confidence=min(confidence, 1.0)
            )
        else:
            # If well below threshold, exit position
            if proba < 0.3:
                return TradingSignal(
                    pair=pair,
                    direction=SignalDirection.SELL,
                    size=1.0,
                    confidence=0.7
                )
            else:
                return TradingSignal(pair=pair, direction=SignalDirection.HOLD)


def run_backtest_with_risk_manager(
    feat_df: pd.DataFrame,
    strategy: XGBoostStrategy,
    pair: str = "BTCUSDT",
    fee_bps: float = 10.0,
    risk_config: dict = None
):
    """
    Run backtest with RiskManager.

    Returns:
        dict with metrics
    """
    if risk_config is None:
        risk_config = {}

    risk_mgr = RiskManager(risk_config)

    # Track state
    cash = 100_000.0
    position = 0.0
    entry_price = None
    fee_rate = fee_bps / 10000
    closed_trades = []
    utilization = []
    equity_curve = []

    # Track daily coverage
    days_with_trades = set()

    # Initialize HWM
    risk_mgr.initialize_hwm(cash)

    print(f"\n  Running backtest on {len(feat_df)} bars...")

    for idx, row in feat_df.iterrows():
        close = row["close"]
        atr = row.get("atr_proxy", 0.0)

        # Calculate portfolio value
        position_value = position * close
        portfolio_value = cash + position_value

        # Track equity curve
        equity_curve.append(portfolio_value)

        # Track utilization
        util = position_value / portfolio_value if portfolio_value > 0 else 0.0
        utilization.append(util)

        # Track date for coverage
        date = idx.date()

        # Check stops for existing position
        if position > 0:
            stop_result = risk_mgr.check_stops(pair, close, atr)
            if stop_result.should_exit:
                # Exit due to stop
                exit_value = position * close * (1 - fee_rate)
                pnl_pct = (close - entry_price) / entry_price if entry_price else 0
                closed_trades.append({
                    "ts": idx,
                    "entry_price": entry_price,
                    "exit_price": close,
                    "pnl_pct": pnl_pct,
                    "exit_reason": stop_result.exit_reason,
                })
                cash += exit_value
                position = 0.0
                entry_price = None
                risk_mgr.record_exit(pair)

                # Track daily coverage
                days_with_trades.add(date)

        # Generate signal
        signal = strategy.generate_signal(pair, feat_df.loc[:idx])

        # Handle BUY signal
        if signal.direction == SignalDirection.BUY and position == 0:
            # Size position through RiskManager
            open_positions = {pair: position_value} if position > 0 else {}
            sizing = risk_mgr.size_new_position(
                pair=pair,
                current_price=close,
                current_atr=atr,
                free_balance_usd=cash,
                open_positions=open_positions,
                regime_multiplier=1.0,  # Assume BULL regime for backtest
                confidence=signal.confidence,
                portfolio_weight=1.0,  # Single asset for now
            )

            if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity > 0:
                # Enter position
                position = sizing.approved_quantity
                entry_cost = position * close * (1 + fee_rate)

                if entry_cost <= cash:
                    cash -= entry_cost
                    entry_price = close * (1 + fee_rate)
                    risk_mgr.record_entry(pair, entry_price, sizing.trailing_stop_price)

                    # Track daily coverage
                    days_with_trades.add(date)
                else:
                    # Insufficient funds - skip trade
                    position = 0.0

        # Handle SELL signal
        elif signal.direction == SignalDirection.SELL and position > 0:
            # Exit position
            exit_value = position * close * (1 - fee_rate)
            pnl_pct = (close - entry_price) / entry_price if entry_price else 0
            closed_trades.append({
                "ts": idx,
                "entry_price": entry_price,
                "exit_price": close,
                "pnl_pct": pnl_pct,
                "exit_reason": "signal_exit",
            })
            cash += exit_value
            position = 0.0
            entry_price = None
            risk_mgr.record_exit(pair)

            # Track daily coverage
            days_with_trades.add(date)

    # Calculate metrics
    start_capital = 100_000.0
    final_capital = cash + (position * feat_df.iloc[-1]["close"])
    total_return = (final_capital / start_capital) - 1.0

    # Use tracked equity curve
    equity_series = pd.Series(equity_curve)

    # Sharpe ratio
    returns = equity_series.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 96)  # 96 bars per day (15m)
    else:
        sharpe = 0.0

    # Sortino ratio
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(252 * 96)
    else:
        sortino = 0.0

    # Max drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()

    # Calmar ratio
    if max_dd < 0:
        calmar = total_return / abs(max_dd)
    else:
        calmar = 0.0

    # Composite score
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    # Win rate
    if len(closed_trades) > 0:
        wins = [t for t in closed_trades if t["pnl_pct"] > 0]
        win_rate = len(wins) / len(closed_trades)
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
        losses = [t for t in closed_trades if t["pnl_pct"] <= 0]
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0.0
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0

    # Daily coverage
    total_days = (feat_df.index[-1].date() - feat_df.index[0].date()).days
    daily_coverage_pct = len(days_with_trades) / total_days if total_days > 0 else 0.0

    # Average utilization
    avg_util = np.mean(utilization) if utilization else 0.0

    return {
        "strategy": strategy.strategy_name,
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "composite_score": composite,
        "num_trades": len(closed_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "daily_coverage_pct": daily_coverage_pct,
        "avg_utilization": avg_util,
        "trades": [{"ts": str(t["ts"]), "entry_price": t["entry_price"], "exit_price": t["exit_price"],
                    "pnl_pct": t["pnl_pct"], "exit_reason": t["exit_reason"]} for t in closed_trades]
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest XGBoost trading models")
    parser.add_argument(
        "--strategy",
        choices=["mean_reversion", "momentum", "volatility_breakout", "rsi_divergence", "trend_following", "all"],
        required=True,
        help="Strategy to backtest (or 'all')"
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2026-03-01", help="End date")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for BUY (default 0.5)")
    args = parser.parse_args()

    print("="*70)
    print("  XGBoost Backtest Pipeline")
    print("="*70)

    # Load features
    print("\nLoading 15m features...")
    df = prepare_15m_features(
        btc_path="research_data/BTCUSDT_15m.parquet",
        eth_path="research_data/ETHUSDT_15m.parquet",
        sol_path="research_data/SOLUSDT_15m.parquet",
        funding_path="research_data/BTCUSDT_funding.parquet",
        start=args.start,
        end=args.end,
    )

    print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Define strategies
    strategies = []
    if args.strategy == "all":
        strategies = ["mean_reversion", "momentum", "volatility_breakout", "rsi_divergence", "trend_following"]
    else:
        strategies = [args.strategy]

    # Backtest each strategy
    results = []

    for strategy_name in strategies:
        print(f"\n{'='*70}")
        print(f"  Testing: {strategy_name}")
        print(f"{'='*70}")

        model_path = MODELS_DIR / f"xgb_{strategy_name}_15m.pkl"

        if not model_path.exists():
            print(f"  ❌ Model not found: {model_path}")
            continue

        # Load model
        strategy = XGBoostStrategy(model_path, threshold=args.threshold)

        # Run backtest
        result = run_backtest_with_risk_manager(
            feat_df=df,
            strategy=strategy,
            pair="BTCUSDT",
            fee_bps=10.0
        )

        # Print results
        print(f"\n  Results:")
        print(f"    Total Return: {result['total_return']*100:+.2f}%")
        print(f"    Sharpe: {result['sharpe']:.3f}")
        print(f"    Sortino: {result['sortino']:.3f}")
        print(f"    Calmar: {result['calmar']:.3f}")
        print(f"    Composite Score: {result['composite_score']:.3f}")
        print(f"    Max Drawdown: {result['max_drawdown']*100:.2f}%")
        print(f"    Trades: {result['num_trades']}")
        print(f"    Win Rate: {result['win_rate']*100:.1f}%")
        print(f"    Daily Coverage: {result['daily_coverage_pct']*100:.1f}%")

        # Save results
        output_path = RESULTS_DIR / f"xgb_{strategy_name}_15m.json"
        with open(output_path, "w") as f:
            # Save results (trades already serialized in result dict)
            json.dump(result, f, indent=2, default=str)

        print(f"    ✅ Results saved: {output_path}")

        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("  Summary: XGBoost Models")
    print(f"{'='*70}\n")

    # Sort by composite score
    results_sorted = sorted(results, key=lambda x: x["composite_score"], reverse=True)

    print(f"{'Strategy':<25} {'Composite':<12} {'Return':<12} {'Sharpe':<10} {'Coverage':<12}")
    print("-" * 70)

    for r in results_sorted:
        print(f"{r['strategy']:<25} {r['composite_score']:<12.3f} {r['total_return']*100:+10.2f}% {r['sharpe']:<10.3f} {r['daily_coverage_pct']*100:>10.1f}%")

    print("\n")


if __name__ == "__main__":
    main()

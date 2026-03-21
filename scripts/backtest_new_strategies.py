#!/usr/bin/env python3
"""
Backtest runner for new correlation/mean-reversion/rotation strategies.

Tests:
1. BTC Correlation Divergence
2. Multi-Factor Mean Reversion
3. Relative Strength Rotation

Uses shared RiskManager + PortfolioAllocator framework with composite scoring:
  composite_score = 0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar

Supports both 15m and 4h timeframes for comparison.

Usage:
  python scripts/backtest_new_strategies.py --strategy correlation --timeframe 15m
  python scripts/backtest_new_strategies.py --strategy mean_reversion --timeframe 15m
  python scripts/backtest_new_strategies.py --strategy rotation --timeframe 15m
  python scripts/backtest_new_strategies.py --strategy all --timeframe 15m  # Run all strategies
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features_15m import prepare_15m_features
from bot.execution.portfolio import PortfolioAllocator
from bot.execution.risk import RiskDecision, RiskManager
from bot.strategy.base import SignalDirection
from bot.strategy.btc_correlation_divergence import BTCCorrelationDivergenceStrategy
from bot.strategy.multifactor_mean_reversion import MultifactorMeanReversionStrategy
from bot.strategy.relative_strength_rotation import RelativeStrengthRotationStrategy
from bot.strategy.volatility_breakout import VolatilityBreakoutStrategy
from bot.strategy.rsi_divergence import RSIDivergenceStrategy
from bot.strategy.ema_crossover_aggressive import EMACrossoverAggressiveStrategy
from bot.strategy.momentum_aggressive import MomentumAggressiveStrategy
from bot.strategy.always_in_market import AlwaysInMarketStrategy

PERIODS_15M = 35040  # 365.25 * 24 * 60 / 15 - annualization for 15m bars
PERIODS_4H = 2190    # 365.25 * 24 / 4 - annualization for 4h bars


def run_backtest_with_risk_manager(
    feat_df: pd.DataFrame,
    strategy,
    pair: str = "BTC/USD",
    fee_bps: int = 10,
    risk_config: dict | None = None,
) -> tuple:
    """
    Bar-by-bar backtest with RiskManager and PortfolioAllocator integration.

    Args:
        feat_df: Feature matrix with UTC DatetimeIndex
        strategy: Strategy instance (generates signals)
        pair: Trading pair (default "BTC/USD")
        fee_bps: Transaction fee in basis points
        risk_config: RiskManager config dict

    Returns:
        (returns_series, closed_trades, utilization_series):
        - returns_series: pd.Series with bar-by-bar returns
        - closed_trades: list of trade dicts with timestamps
        - utilization_series: pd.Series with position utilization over time
    """
    if risk_config is None:
        risk_config = {
            "hard_stop_pct": 0.08,
            "atr_stop_multiplier": 2.0,
            "circuit_breaker_drawdown": 0.30,
            "max_positions": 1,
            "max_single_position_pct": 0.40,
            "risk_per_trade_pct": 0.02,
            "expected_win_loss_ratio": 1.5,
        }

    risk_mgr = RiskManager(risk_config)
    fee_rate = fee_bps / 10_000

    # Initialize portfolio
    initial_capital = 10000.0
    cash = initial_capital
    position = 0.0  # BTC quantity
    entry_price = None

    risk_mgr.initialize_hwm(initial_capital)

    returns = []
    timestamps = []
    closed_trades = []
    utilization = []

    for idx, row in feat_df.iterrows():
        close = row["close"]
        atr = row.get("atr_proxy", 0.0)

        # Calculate portfolio value
        position_value = position * close
        portfolio_value = cash + position_value

        # Track utilization
        util = position_value / portfolio_value if portfolio_value > 0 else 0.0
        utilization.append(util)

        # Check stops for existing position
        bar_return = 0.0
        if position > 0:
            stop_result = risk_mgr.check_stops(pair, close, atr)
            if stop_result.should_exit:
                # Exit due to stop
                exit_value = position * close * (1 - fee_rate)
                bar_return = -fee_rate  # Exit fee
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

            else:
                # Mark-to-market return for existing position
                prev_close = feat_df.iloc[feat_df.index.get_loc(idx) - 1]["close"] if feat_df.index.get_loc(idx) > 0 else close
                bar_return = (close - prev_close) / prev_close if prev_close > 0 else 0.0

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
                    bar_return -= fee_rate  # Entry fee
                    risk_mgr.record_entry(pair, entry_price, sizing.trailing_stop_price)
                else:
                    # Insufficient funds - skip trade
                    position = 0.0

        # Handle SELL signal
        elif signal.direction == SignalDirection.SELL and position > 0:
            # Exit position
            exit_value = position * close * (1 - fee_rate)
            bar_return -= fee_rate  # Exit fee
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

        # Update circuit breaker
        risk_mgr.check_circuit_breaker(portfolio_value)

        returns.append(bar_return)
        timestamps.append(idx)

    returns_series = pd.Series(returns, index=pd.DatetimeIndex(timestamps))
    utilization_series = pd.Series(utilization, index=pd.DatetimeIndex(timestamps))

    return returns_series, closed_trades, utilization_series


def compute_metrics(
    returns: pd.Series,
    closed_trades: list[dict],
    utilization: pd.Series,
    periods: int,
) -> dict:
    """
    Compute comprehensive backtest metrics with composite score.

    Composite score = 0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar

    Args:
        returns: Bar-by-bar returns series
        closed_trades: List of closed trade dicts
        utilization: Position utilization series
        periods: Annualization periods (35040 for 15m, 2190 for 4h)

    Returns:
        Dict of metrics including composite_score
    """
    returns = returns[returns.index.notna()]
    n_trades = len(closed_trades)

    # Trade statistics
    if n_trades > 0:
        winners = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
        win_rate = winners / n_trades
        avg_pnl = sum(t["pnl_pct"] for t in closed_trades) / n_trades
        best_trade = max(t["pnl_pct"] for t in closed_trades)
        worst_trade = min(t["pnl_pct"] for t in closed_trades)

        # Daily coverage: how many unique days had trades
        trade_days = {pd.Timestamp(t["ts"]).normalize() for t in closed_trades}
        cal_days = int(returns.index.normalize().nunique()) if len(returns) > 0 else 0
        daily_coverage = len(trade_days) / cal_days if cal_days > 0 else 0.0
        trades_per_day = n_trades / cal_days if cal_days > 0 else 0.0
    else:
        win_rate = avg_pnl = best_trade = worst_trade = 0.0
        daily_coverage = trades_per_day = 0.0

    # Risk-adjusted metrics
    sharpe = float(qs.stats.sharpe(returns, periods=periods))
    sortino = float(qs.stats.sortino(returns, periods=periods))
    calmar = float(qs.stats.calmar(returns))

    # Composite score (same as intraday_hybrid)
    composite_score = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    return {
        "total_return_pct": float((1 + returns).prod() - 1) * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "composite_score": composite_score,
        "max_drawdown_pct": float(qs.stats.max_drawdown(returns)) * 100,
        "n_trades": n_trades,
        "daily_coverage": daily_coverage,
        "trades_per_day": trades_per_day,
        "avg_utilization": float(utilization.mean()),
        "n_bars": len(returns),
        "win_rate_pct": win_rate * 100,
        "avg_trade_pnl_pct": avg_pnl * 100,
        "best_trade_pct": best_trade * 100,
        "worst_trade_pct": worst_trade * 100,
    }


def print_metrics(metrics: dict, strategy_name: str) -> None:
    """Print formatted metrics report."""
    print("\n" + "=" * 70)
    print(f"  BACKTEST RESULTS: {strategy_name}")
    print("=" * 70)
    print(f"  Total return     : {metrics['total_return_pct']:+.2f}%")
    print(f"  Sharpe           : {metrics['sharpe']:.3f}")
    print(f"  Sortino          : {metrics['sortino']:.3f}")
    print(f"  Calmar           : {metrics['calmar']:.3f}")
    print(f"  COMPOSITE SCORE  : {metrics['composite_score']:.3f}")
    print(f"  Max drawdown     : {metrics['max_drawdown_pct']:.2f}%")
    print("-" * 70)
    print(f"  # Trades         : {metrics['n_trades']}")
    print(f"  Trades/day       : {metrics['trades_per_day']:.2f}")
    print(f"  Daily coverage   : {metrics['daily_coverage']*100:.1f}%")
    print(f"  Win rate         : {metrics['win_rate_pct']:.1f}%")
    print(f"  Avg trade PnL    : {metrics['avg_trade_pnl_pct']:+.2f}%")
    print(f"  Avg utilization  : {metrics['avg_utilization']*100:.1f}%")
    print(f"  # Bars           : {metrics['n_bars']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Backtest new strategies with composite scoring")
    parser.add_argument(
        "--strategy",
        choices=["correlation", "mean_reversion", "rotation", "volatility", "rsi_div",
                 "ema_cross", "momentum_agg", "always_in", "all", "aggressive_only"],
        required=True,
        help="Strategy to test (or 'all' for all, 'aggressive_only' for high-frequency)",
    )
    parser.add_argument(
        "--timeframe",
        choices=["15m", "4h"],
        default="15m",
        help="Timeframe to test (default: 15m)",
    )
    parser.add_argument("--start", default=None, help="Start date (e.g., '2024-01-01')")
    parser.add_argument("--end", default=None, help="End date (e.g., '2024-12-31')")
    parser.add_argument("--fee-bps", type=int, default=10, help="Fee in basis points (default 10)")
    parser.add_argument("--output-dir", default="research_results", help="Output directory for results")
    args = parser.parse_args()

    print(f"\nLoading {args.timeframe} data...")

    # Load features based on timeframe
    if args.timeframe == "15m":
        feat = prepare_15m_features(
            btc_path="research_data/BTCUSDT_15m.parquet",
            eth_path="research_data/ETHUSDT_4h.parquet",  # Will be resampled if needed
            sol_path="research_data/SOLUSDT_4h.parquet",
            funding_path="research_data/BTCUSDT_funding.parquet",
            start=args.start,
            end=args.end,
        )
        periods = PERIODS_15M
    else:
        # For 4h, use existing feature pipeline
        from bot.data.features import compute_features, compute_cross_asset_features
        btc = pd.read_parquet("research_data/BTCUSDT_4h.parquet")
        eth = pd.read_parquet("research_data/ETHUSDT_4h.parquet")
        sol = pd.read_parquet("research_data/SOLUSDT_4h.parquet")

        feat = compute_features(btc)
        feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
        feat = feat.dropna()

        # Normalize column names to lowercase for consistency with 15m
        feat.columns = feat.columns.str.lower()

        # Rename specific columns to match 15m naming convention
        rename_map = {
            "rsi_14": "rsi",
            "ema_20": "ema_20",
            "ema_50": "ema_50",
            "macd_12_26_9": "macd",
            "macds_12_26_9": "macd_signal",
            "macdh_12_26_9": "macd_hist",
        }
        feat.rename(columns=rename_map, inplace=True)

        if args.start:
            feat = feat[feat.index >= pd.Timestamp(args.start, tz="UTC")]
        if args.end:
            feat = feat[feat.index <= pd.Timestamp(args.end, tz="UTC")]

        periods = PERIODS_4H

    print(f"  Feature matrix: {feat.shape[0]} bars x {feat.shape[1]} columns")
    print(f"  Date range: {feat.index[0]} to {feat.index[-1]}")

    # Define strategies
    strategies = {}
    if args.strategy in ["correlation", "all"]:
        strategies["correlation_divergence"] = BTCCorrelationDivergenceStrategy()
    if args.strategy in ["mean_reversion", "all"]:
        strategies["multifactor_mean_reversion"] = MultifactorMeanReversionStrategy()
    if args.strategy in ["rotation", "all"]:
        strategies["relative_strength_rotation"] = RelativeStrengthRotationStrategy()
    if args.strategy in ["volatility", "all"]:
        strategies["volatility_breakout"] = VolatilityBreakoutStrategy()
    if args.strategy in ["rsi_div", "all"]:
        strategies["rsi_divergence"] = RSIDivergenceStrategy()

    # Aggressive high-frequency strategies
    if args.strategy in ["ema_cross", "all", "aggressive_only"]:
        strategies["ema_crossover_aggressive"] = EMACrossoverAggressiveStrategy()
    if args.strategy in ["momentum_agg", "all", "aggressive_only"]:
        strategies["momentum_aggressive"] = MomentumAggressiveStrategy()
    if args.strategy in ["always_in", "all", "aggressive_only"]:
        strategies["always_in_market"] = AlwaysInMarketStrategy()

    # Run backtests
    results = {}
    for name, strategy in strategies.items():
        print(f"\nRunning backtest: {name}...")
        returns, trades, util = run_backtest_with_risk_manager(
            feat, strategy, pair="BTC/USD", fee_bps=args.fee_bps
        )

        metrics = compute_metrics(returns, trades, util, periods)
        print_metrics(metrics, name)

        results[name] = metrics

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for name, metrics in results.items():
        output_file = output_dir / f"{name}_{args.timeframe}.json"
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {output_file}")

    # Summary comparison if multiple strategies
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("  STRATEGY COMPARISON (sorted by composite score)")
        print("=" * 70)
        sorted_results = sorted(results.items(), key=lambda x: x[1]["composite_score"], reverse=True)
        print(f"{'Strategy':<30} {'Composite':>10} {'Sharpe':>8} {'Sortino':>8} {'Trades':>8}")
        print("-" * 70)
        for name, m in sorted_results:
            print(f"{name:<30} {m['composite_score']:>10.3f} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['n_trades']:>8}")
        print("=" * 70)


if __name__ == "__main__":
    main()

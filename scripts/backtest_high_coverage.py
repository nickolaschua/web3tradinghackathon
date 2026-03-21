#!/usr/bin/env python3
"""
Backtest with High Coverage Risk Manager

Tests strategies using looser risk management to maximize daily coverage.

Usage:
  python scripts/backtest_high_coverage.py --strategy always_in --timeframe 15m
  python scripts/backtest_high_coverage.py --strategy all --timeframe 15m
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import quantstats as qs

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features_15m import prepare_15m_features
from bot.execution.risk_high_coverage import RiskManagerHighCoverage, RiskDecision
from bot.strategy.base import SignalDirection
from bot.strategy.always_in_market import AlwaysInMarketStrategy
from bot.strategy.rsi_divergence import RSIDivergenceStrategy
from bot.strategy.multifactor_mean_reversion import MultifactorMeanReversionStrategy
from bot.strategy.volatility_breakout import VolatilityBreakoutStrategy
from bot.strategy.ema_crossover_aggressive import EMACrossoverAggressiveStrategy

PERIODS_15M = 35040


def run_backtest_high_coverage(
    feat_df: pd.DataFrame,
    strategy,
    pair: str = "BTC/USD",
    fee_bps: int = 10,
) -> tuple:
    """
    Backtest with high-coverage risk manager.
    """
    # High coverage config
    risk_config = {
        "hard_stop_pct": 0.15,           # 15% vs 8%
        "atr_stop_multiplier": 0.8,      # 0.8x vs 2.0x
        "min_hold_bars": 16,             # 4 hours minimum hold
        "circuit_breaker_drawdown": 0.40, # 40% vs 30%
        "max_positions": 1,
        "max_single_position_pct": 0.50,  # 50% vs 40%
        "risk_per_trade_pct": 0.03,       # 3% vs 2%
        "expected_win_loss_ratio": 1.5,
    }

    risk_mgr = RiskManagerHighCoverage(risk_config)
    fee_rate = fee_bps / 10_000

    initial_capital = 10000.0
    cash = initial_capital
    position = 0.0
    entry_price = None

    risk_mgr.initialize_hwm(initial_capital)

    returns = []
    timestamps = []
    closed_trades = []
    utilization = []
    trading_days = set()

    for idx, row in feat_df.iterrows():
        close = row["close"]
        atr = row.get("atr_proxy", 0.0)

        position_value = position * close
        portfolio_value = cash + position_value
        util = position_value / portfolio_value if portfolio_value > 0 else 0.0
        utilization.append(util)

        # Check stops
        bar_return = 0.0
        if position > 0:
            stop_result = risk_mgr.check_stops(pair, close, atr)
            if stop_result.should_exit:
                exit_value = position * close * (1 - fee_rate)
                bar_return = -fee_rate
                pnl_pct = (close - entry_price) / entry_price if entry_price else 0
                closed_trades.append({
                    "ts": idx,
                    "pnl_pct": pnl_pct,
                    "exit_reason": stop_result.exit_reason,
                })
                trading_days.add(idx.normalize())
                cash += exit_value
                position = 0.0
                entry_price = None
                risk_mgr.record_exit(pair)
            else:
                prev_close = feat_df.iloc[feat_df.index.get_loc(idx) - 1]["close"] if feat_df.index.get_loc(idx) > 0 else close
                bar_return = (close - prev_close) / prev_close if prev_close > 0 else 0.0

        # Generate signal
        signal = strategy.generate_signal(pair, feat_df.loc[:idx])

        # Handle BUY
        if signal.direction == SignalDirection.BUY and position == 0:
            open_positions = {pair: position_value} if position > 0 else {}
            sizing = risk_mgr.size_new_position(
                pair=pair,
                current_price=close,
                current_atr=atr,
                free_balance_usd=cash,
                open_positions=open_positions,
                regime_multiplier=1.0,
                confidence=signal.confidence,
                portfolio_weight=1.0,
            )

            if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity > 0:
                position = sizing.approved_quantity
                entry_cost = position * close * (1 + fee_rate)

                if entry_cost <= cash:
                    cash -= entry_cost
                    entry_price = close * (1 + fee_rate)
                    bar_return -= fee_rate
                    trading_days.add(idx.normalize())
                    risk_mgr.record_entry(pair, entry_price, sizing.trailing_stop_price)
                else:
                    position = 0.0

        # Handle SELL
        elif signal.direction == SignalDirection.SELL and position > 0:
            exit_value = position * close * (1 - fee_rate)
            bar_return -= fee_rate
            pnl_pct = (close - entry_price) / entry_price if entry_price else 0
            closed_trades.append({
                "ts": idx,
                "pnl_pct": pnl_pct,
                "exit_reason": "signal_exit",
            })
            trading_days.add(idx.normalize())
            cash += exit_value
            position = 0.0
            entry_price = None
            risk_mgr.record_exit(pair)

        risk_mgr.check_circuit_breaker(portfolio_value)
        returns.append(bar_return)
        timestamps.append(idx)

    returns_series = pd.Series(returns, index=pd.DatetimeIndex(timestamps))
    utilization_series = pd.Series(utilization, index=pd.DatetimeIndex(timestamps))

    return returns_series, closed_trades, utilization_series, trading_days


def compute_metrics(returns, trades, utilization, trading_days, periods):
    """Compute metrics including daily coverage."""
    n_trades = len(trades)
    cal_days = int(returns.index.normalize().nunique()) if len(returns) > 0 else 0
    daily_coverage = len(trading_days) / cal_days if cal_days > 0 else 0.0

    sharpe = float(qs.stats.sharpe(returns, periods=periods))
    sortino = float(qs.stats.sortino(returns, periods=periods))
    calmar = float(qs.stats.calmar(returns))
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
        "trades_per_day": n_trades / cal_days if cal_days > 0 else 0,
        "avg_utilization": float(utilization.mean()),
        "n_bars": len(returns),
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest with high-coverage risk manager")
    parser.add_argument(
        "--strategy",
        choices=["always_in", "rsi_div", "mean_reversion", "volatility", "ema_cross", "all"],
        required=True,
    )
    parser.add_argument("--timeframe", default="15m", choices=["15m"])
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--fee-bps", type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  HIGH COVERAGE MODE - Looser Risk Management")
    print(f"{'='*70}")
    print(f"  Hard stop: 15% (vs 8% standard)")
    print(f"  ATR multiplier: 0.8x (vs 2.0x standard)")
    print(f"  Min hold time: 16 bars = 4 hours")
    print(f"  Max position: 50% (vs 40% standard)")
    print(f"  Risk per trade: 3% (vs 2% standard)")
    print(f"{'='*70}\n")

    print(f"Loading {args.timeframe} data...")
    feat = prepare_15m_features(
        btc_path="research_data/BTCUSDT_15m.parquet",
        eth_path="research_data/ETHUSDT_15m.parquet",
        sol_path="research_data/SOLUSDT_15m.parquet",
        funding_path="research_data/BTCUSDT_funding.parquet",
        start=args.start,
        end=args.end,
    )

    print(f"  Feature matrix: {feat.shape[0]} bars x {feat.shape[1]} columns")
    print(f"  Date range: {feat.index[0]} to {feat.index[-1]}")

    # Define strategies
    strategies = {}
    if args.strategy in ["always_in", "all"]:
        strategies["always_in_market"] = AlwaysInMarketStrategy()
    if args.strategy in ["rsi_div", "all"]:
        strategies["rsi_divergence"] = RSIDivergenceStrategy()
    if args.strategy in ["mean_reversion", "all"]:
        strategies["mean_reversion"] = MultifactorMeanReversionStrategy()
    if args.strategy in ["volatility", "all"]:
        strategies["volatility_breakout"] = VolatilityBreakoutStrategy()
    if args.strategy in ["ema_cross", "all"]:
        strategies["ema_crossover_aggressive"] = EMACrossoverAggressiveStrategy()

    # Run backtests
    results = {}
    for name, strategy in strategies.items():
        print(f"\nRunning HIGH COVERAGE backtest: {name}...")
        returns, trades, util, trading_days = run_backtest_high_coverage(
            feat, strategy, pair="BTC/USD", fee_bps=args.fee_bps
        )

        metrics = compute_metrics(returns, trades, util, trading_days, PERIODS_15M)

        print("\n" + "=" * 70)
        print(f"  {name.upper()} - HIGH COVERAGE RESULTS")
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
        print(f"  DAILY COVERAGE   : {metrics['daily_coverage']*100:.1f}%  ⭐️")
        print(f"  Avg utilization  : {metrics['avg_utilization']*100:.1f}%")
        print("=" * 70)

        results[name] = metrics

        # Save individual result
        output = Path(f"research_results/{name}_high_coverage_15m.json")
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)

    # Comparison
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("  STRATEGY COMPARISON - HIGH COVERAGE MODE")
        print("=" * 70)
        sorted_results = sorted(results.items(), key=lambda x: x[1]["composite_score"], reverse=True)
        print(f"{'Strategy':<25} {'Coverage':>10} {'Composite':>10} {'Return':>10} {'MaxDD':>10}")
        print("-" * 70)
        for name, m in sorted_results:
            print(f"{name:<25} {m['daily_coverage']*100:>9.1f}% {m['composite_score']:>10.3f} "
                  f"{m['total_return_pct']:>9.1f}% {m['max_drawdown_pct']:>9.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Portfolio Backtest for Maximum Daily Coverage

Runs multiple strategies simultaneously to maximize daily trading coverage.
Target: 95%+ daily coverage while maintaining positive returns.

Usage:
  python scripts/backtest_portfolio_coverage.py --timeframe 15m --start 2024-01-01
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
from bot.execution.risk import RiskDecision, RiskManager
from bot.strategy.base import SignalDirection
from bot.strategy.rsi_divergence import RSIDivergenceStrategy
from bot.strategy.multifactor_mean_reversion import MultifactorMeanReversionStrategy
from bot.strategy.volatility_breakout import VolatilityBreakoutStrategy
from bot.strategy.always_in_market import AlwaysInMarketStrategy

PERIODS_15M = 35040


def run_portfolio_backtest(
    feat_df: pd.DataFrame,
    strategies: dict,
    fee_bps: int = 10,
) -> tuple:
    """
    Run multiple strategies in parallel, taking ANY signal that fires.

    This maximizes daily coverage by trading whenever ANY strategy signals.
    """
    risk_config = {
        "hard_stop_pct": 0.08,
        "atr_stop_multiplier": 2.0,
        "circuit_breaker_drawdown": 0.30,
        "max_positions": 1,  # Still only 1 position at a time
        "max_single_position_pct": 0.40,
        "risk_per_trade_pct": 0.02,
        "expected_win_loss_ratio": 1.5,
    }

    risk_mgr = RiskManager(risk_config)
    fee_rate = fee_bps / 10_000

    initial_capital = 10000.0
    cash = initial_capital
    position = 0.0
    entry_price = None
    current_strategy = None

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
            stop_result = risk_mgr.check_stops("BTC/USD", close, atr)
            if stop_result.should_exit:
                exit_value = position * close * (1 - fee_rate)
                bar_return = -fee_rate
                pnl_pct = (close - entry_price) / entry_price if entry_price else 0
                closed_trades.append({
                    "ts": idx,
                    "strategy": current_strategy,
                    "pnl_pct": pnl_pct,
                })
                trading_days.add(idx.normalize())
                cash += exit_value
                position = 0.0
                entry_price = None
                current_strategy = None
                risk_mgr.record_exit("BTC/USD")
            else:
                prev_close = feat_df.iloc[feat_df.index.get_loc(idx) - 1]["close"] if feat_df.index.get_loc(idx) > 0 else close
                bar_return = (close - prev_close) / prev_close if prev_close > 0 else 0.0

        # Try all strategies, take first BUY signal if no position
        if position == 0:
            for strat_name, strategy in strategies.items():
                signal = strategy.generate_signal("BTC/USD", feat_df.loc[:idx])

                if signal.direction == SignalDirection.BUY:
                    # Take this signal
                    open_positions = {}
                    sizing = risk_mgr.size_new_position(
                        pair="BTC/USD",
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
                            current_strategy = strat_name
                            trading_days.add(idx.normalize())
                            risk_mgr.record_entry("BTC/USD", entry_price, sizing.trailing_stop_price)
                            break  # Only take one signal per bar
                        else:
                            position = 0.0

        # Check for SELL signals if we have a position
        elif position > 0:
            signal = strategies[current_strategy].generate_signal("BTC/USD", feat_df.loc[:idx])

            if signal.direction == SignalDirection.SELL:
                exit_value = position * close * (1 - fee_rate)
                bar_return -= fee_rate
                pnl_pct = (close - entry_price) / entry_price if entry_price else 0
                closed_trades.append({
                    "ts": idx,
                    "strategy": current_strategy,
                    "pnl_pct": pnl_pct,
                })
                trading_days.add(idx.normalize())
                cash += exit_value
                position = 0.0
                entry_price = None
                current_strategy = None
                risk_mgr.record_exit("BTC/USD")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="15m", choices=["15m", "4h"])
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--fee-bps", type=int, default=10)
    args = parser.parse_args()

    print(f"\nLoading {args.timeframe} data...")
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

    # Portfolio of best strategies
    strategies = {
        "always_in_market": AlwaysInMarketStrategy(),
        "rsi_divergence": RSIDivergenceStrategy(),
        "mean_reversion": MultifactorMeanReversionStrategy(),
        "volatility_breakout": VolatilityBreakoutStrategy(),
    }

    print(f"\nRunning portfolio with {len(strategies)} strategies...")
    returns, trades, util, trading_days = run_portfolio_backtest(
        feat, strategies, fee_bps=args.fee_bps
    )

    metrics = compute_metrics(returns, trades, util, trading_days, PERIODS_15M)

    print("\n" + "=" * 70)
    print("  PORTFOLIO BACKTEST RESULTS")
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
    print(f"  DAILY COVERAGE   : {metrics['daily_coverage']*100:.1f}%")
    print(f"  Avg utilization  : {metrics['avg_utilization']*100:.1f}%")
    print("=" * 70)

    # Save
    output = Path("research_results/portfolio_coverage.json")
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Layered multi-timeframe backtest: XGBoost + intraday scalps.

Runs multiple strategies simultaneously on different "layers":
- Layer 1 (Base): XGBoost directional positions (4-hour horizon, longer holds)
- Layer 2 (Scalp): Intraday mean reversion scalps (15min-2h holds)
- Layer 3 (Momentum): Momentum breakout scalps (30min-3h holds)

Each layer can have independent positions. Target: 80%+ daily coverage.

Usage:
  python scripts/backtest_layered_strategies.py
  python scripts/backtest_layered_strategies.py --xgb-threshold 0.65 --assets BTC,ETH,SOL
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
from bot.strategy.intraday_scalp import ExtremeMeanReversionScalp, MomentumBreakoutScalp, VWAPReversionScalp

MODELS_DIR = Path("models")
RESULTS_DIR = Path("research_results")


class LayeredPortfolioManager:
    """
    Manages multiple strategy layers per asset.

    Each asset has 3 independent position "slots":
    1. xgb_long: XGBoost long-term position
    2. scalp_mr: Mean reversion scalp
    3. scalp_mom: Momentum scalp
    """

    def __init__(self, xgb_threshold=0.65, xgb_exit=0.12):
        self.xgb_threshold = xgb_threshold
        self.xgb_exit = xgb_exit

        # Load XGBoost models
        self.xgb_models = {}
        for asset in ["btc", "eth", "sol"]:
            model_path = MODELS_DIR / f"xgb_{asset}_15m.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.xgb_models[asset.upper()] = pickle.load(f)

        # Initialize scalp strategies
        self.mr_scalp = ExtremeMeanReversionScalp(rsi_oversold=15, rsi_overbought=85)
        self.mom_scalp = MomentumBreakoutScalp(bb_squeeze_pct=0.20, volume_threshold=2.0)
        self.vwap_scalp = VWAPReversionScalp(vwap_threshold=1.5)

        print(f"Initialized: {len(self.xgb_models)} XGBoost + 3 scalp strategies")

    def generate_layered_signals(self, features_dict):
        """
        Generate signals for all layers and all assets.

        Returns: {(asset, layer): signal}
        """
        signals = {}

        for asset in features_dict.keys():
            features = features_dict[asset]

            if features.empty:
                continue

            pair = f"{asset}/USD"

            # Layer 1: XGBoost (if model exists)
            if asset in self.xgb_models:
                model = self.xgb_models[asset]
                last = features.iloc[[-1]].copy()

                feature_cols = list(model.feature_names_in_)
                missing = [c for c in feature_cols if c not in last.columns]
                for col in missing:
                    last[col] = float("nan")

                row = last[feature_cols]

                try:
                    proba = float(model.predict_proba(row)[0, 1])

                    if proba >= self.xgb_threshold:
                        signals[(asset, "xgb")] = TradingSignal(
                            pair=pair,
                            direction=SignalDirection.BUY,
                            size=1.0,
                            confidence=proba
                        )
                    elif proba <= self.xgb_exit:
                        signals[(asset, "xgb")] = TradingSignal(
                            pair=pair,
                            direction=SignalDirection.SELL,
                            size=1.0,
                            confidence=proba
                        )
                    else:
                        signals[(asset, "xgb")] = TradingSignal(pair=pair)
                except:
                    signals[(asset, "xgb")] = TradingSignal(pair=pair)

            # Layer 2: Mean Reversion Scalp
            signals[(asset, "mr_scalp")] = self.mr_scalp.generate_signal(pair, features)

            # Layer 3: Momentum Scalp
            signals[(asset, "mom_scalp")] = self.mom_scalp.generate_signal(pair, features)

            # Layer 4: VWAP Scalp
            signals[(asset, "vwap_scalp")] = self.vwap_scalp.generate_signal(pair, features)

        return signals


def prepare_features(asset, btc_df, eth_df, sol_df):
    """Prepare features for a single asset."""

    if asset == "BTC":
        target_df = btc_df
        cross_dfs = {"ETH/USD": eth_df, "SOL/USD": sol_df}
        cross_assets = [("eth", eth_df), ("sol", sol_df)]
        corr_name = "eth_btc"
    elif asset == "ETH":
        target_df = eth_df
        cross_dfs = {"BTC/USD": btc_df, "SOL/USD": sol_df}
        cross_assets = [("btc", btc_df), ("sol", sol_df)]
        corr_name = "eth_btc"
    else:  # SOL
        target_df = sol_df
        cross_dfs = {"BTC/USD": btc_df, "ETH/USD": eth_df}
        cross_assets = [("btc", btc_df), ("eth", eth_df)]
        corr_name = "sol_btc"

    feat = compute_features(target_df)
    feat = compute_cross_asset_features(feat, cross_dfs)

    # Add cross-asset lags
    for name, df in cross_assets:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{name}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{name}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    # Add correlation/beta
    if asset != "BTC":
        target_ret = np.log(target_df["close"] / target_df["close"].shift(1)).reindex(feat.index)
        btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1)).reindex(feat.index)
        corr = target_ret.rolling(2880).corr(btc_ret)
        cov = target_ret.rolling(2880).cov(btc_ret)
        var_btc = btc_ret.rolling(2880).var()
        feat[f"{corr_name}_corr"] = corr.shift(1)
        feat[f"{corr_name}_beta"] = (cov / (var_btc + 1e-10)).shift(1)
    else:
        feat = compute_btc_context_features(feat, eth_df, sol_df, window=2880)

    # Add VWAP (approximation)
    feat["vwap"] = (feat["close"] * feat["volume"]).rolling(20).sum() / feat["volume"].rolling(20).sum()

    feat = feat.dropna()

    return feat


def run_layered_backtest(features_dict, portfolio_mgr, assets):
    """Run backtest with layered positions."""

    risk_mgr = RiskManager({})
    cash = 100_000.0

    # Track positions per layer per asset
    positions = {}
    entry_prices = {}
    entry_times = {}

    for asset in assets:
        positions[asset] = {"xgb": 0.0, "mr_scalp": 0.0, "mom_scalp": 0.0, "vwap_scalp": 0.0}
        entry_prices[asset] = {"xgb": None, "mr_scalp": None, "mom_scalp": None, "vwap_scalp": None}
        entry_times[asset] = {"xgb": None, "mr_scalp": None, "mom_scalp": None, "vwap_scalp": None}

    closed_trades = []
    equity_curve = []
    days_with_trades = set()

    risk_mgr.initialize_hwm(cash)
    fee_rate = 10 / 10000

    # Use BTC index as common timeline
    common_index = features_dict["BTC"].index

    print(f"\nBacktesting {len(common_index)} bars on {len(assets)} assets...")

    for idx in common_index:
        # Get current prices
        prices = {}
        for asset in assets:
            if idx in features_dict[asset].index:
                prices[asset] = features_dict[asset].loc[idx, "close"]

        # Calculate portfolio value
        position_value = sum(
            sum(positions[asset].values()) * prices.get(asset, 0)
            for asset in assets
        )
        portfolio_value = cash + position_value
        equity_curve.append(portfolio_value)

        date = idx.date()

        # Check stops for ALL layers
        for asset in assets:
            if not prices.get(asset):
                continue

            for layer in ["xgb", "mr_scalp", "mom_scalp", "vwap_scalp"]:
                if positions[asset][layer] > 0:
                    pair = f"{asset}/USD-{layer}"  # Unique pair ID per layer

                    atr = features_dict[asset].loc[idx, "atr_proxy"] if "atr_proxy" in features_dict[asset].columns else prices[asset] * 0.02

                    stop_result = risk_mgr.check_stops(pair, prices[asset], atr)

                    if stop_result.should_exit:
                        # Exit this layer
                        exit_value = positions[asset][layer] * prices[asset] * (1 - fee_rate)
                        pnl_pct = (prices[asset] - entry_prices[asset][layer]) / entry_prices[asset][layer] if entry_prices[asset][layer] else 0

                        hold_bars = (idx - entry_times[asset][layer]).total_seconds() / 900 if entry_times[asset][layer] else 0

                        closed_trades.append({
                            "asset": asset,
                            "layer": layer,
                            "ts": idx,
                            "pnl_pct": pnl_pct,
                            "hold_bars": hold_bars,
                            "exit_reason": stop_result.exit_reason,
                        })

                        cash += exit_value
                        positions[asset][layer] = 0.0
                        entry_prices[asset][layer] = None
                        entry_times[asset][layer] = None
                        risk_mgr.record_exit(pair)
                        days_with_trades.add(date)

        # Generate signals for all layers
        current_features = {
            asset: features_dict[asset].loc[:idx]
            for asset in assets
        }

        all_signals = portfolio_mgr.generate_layered_signals(current_features)

        # Execute signals per layer
        for (asset, layer), signal in all_signals.items():
            if not prices.get(asset):
                continue

            pair = f"{asset}/USD-{layer}"

            # BUY
            if signal.direction == SignalDirection.BUY and positions[asset][layer] == 0:
                atr = features_dict[asset].loc[idx, "atr_proxy"] if "atr_proxy" in features_dict[asset].columns else prices[asset] * 0.02

                # Get all open positions across all assets and layers
                open_positions = {}
                for a in assets:
                    for l in ["xgb", "mr_scalp", "mom_scalp", "vwap_scalp"]:
                        if positions[a][l] > 0 and prices.get(a):
                            open_positions[f"{a}/USD-{l}"] = positions[a][l] * prices[a]

                # Adjust confidence for scalps (lower priority)
                conf = signal.confidence
                if layer in ["mr_scalp", "mom_scalp", "vwap_scalp"]:
                    conf = conf * 0.8  # Reduce confidence for scalps

                sizing = risk_mgr.size_new_position(
                    pair=pair,
                    current_price=prices[asset],
                    current_atr=atr,
                    free_balance_usd=cash,
                    open_positions=open_positions,
                    regime_multiplier=1.0,
                    confidence=conf,
                    portfolio_weight=1.0,
                )

                if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity > 0:
                    positions[asset][layer] = sizing.approved_quantity
                    entry_cost = positions[asset][layer] * prices[asset] * (1 + fee_rate)

                    if entry_cost <= cash:
                        cash -= entry_cost
                        entry_prices[asset][layer] = prices[asset] * (1 + fee_rate)
                        entry_times[asset][layer] = idx
                        risk_mgr.record_entry(pair, entry_prices[asset][layer], sizing.trailing_stop_price)
                        days_with_trades.add(date)
                    else:
                        positions[asset][layer] = 0.0

            # SELL
            elif signal.direction == SignalDirection.SELL and positions[asset][layer] > 0:
                exit_value = positions[asset][layer] * prices[asset] * (1 - fee_rate)
                pnl_pct = (prices[asset] - entry_prices[asset][layer]) / entry_prices[asset][layer] if entry_prices[asset][layer] else 0

                hold_bars = (idx - entry_times[asset][layer]).total_seconds() / 900 if entry_times[asset][layer] else 0

                closed_trades.append({
                    "asset": asset,
                    "layer": layer,
                    "ts": idx,
                    "pnl_pct": pnl_pct,
                    "hold_bars": hold_bars,
                    "exit_reason": "signal_exit",
                })

                cash += exit_value
                positions[asset][layer] = 0.0
                entry_prices[asset][layer] = None
                entry_times[asset][layer] = None
                risk_mgr.record_exit(pair)
                days_with_trades.add(date)

    # Calculate metrics
    final_capital = cash
    for asset in assets:
        for layer in ["xgb", "mr_scalp", "mom_scalp", "vwap_scalp"]:
            if positions[asset][layer] > 0:
                final_capital += positions[asset][layer] * features_dict[asset].iloc[-1]["close"]

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

    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0

    total_days = (common_index[-1].date() - common_index[0].date()).days
    daily_coverage_pct = len(days_with_trades) / total_days if total_days > 0 else 0.0

    # Calculate 10-day rolling coverage
    all_dates = pd.date_range(common_index[0].date(), common_index[-1].date(), freq='D')
    coverage_10day = []
    for i in range(len(all_dates) - 9):
        window_start = all_dates[i]
        window_end = all_dates[i + 9]
        days_in_window = sum(1 for d in days_with_trades if window_start <= d <= window_end)
        coverage_10day.append(days_in_window)

    avg_10day = np.mean(coverage_10day) if coverage_10day else 0
    pct_8plus = sum(1 for x in coverage_10day if x >= 8) / len(coverage_10day) if coverage_10day else 0

    # Layer breakdown
    trades_by_layer = {}
    for layer in ["xgb", "mr_scalp", "mom_scalp", "vwap_scalp"]:
        layer_trades = [t for t in closed_trades if t["layer"] == layer]
        if layer_trades:
            layer_wins = [t for t in layer_trades if t["pnl_pct"] > 0]
            avg_hold = np.mean([t["hold_bars"] for t in layer_trades])
            trades_by_layer[layer] = {
                "count": len(layer_trades),
                "win_rate": len(layer_wins) / len(layer_trades),
                "avg_hold_bars": avg_hold,
                "avg_hold_hours": avg_hold * 0.25,
            }

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
        "avg_days_per_10day_window": avg_10day,
        "pct_windows_8plus_days": pct_8plus,
        "trades_by_layer": trades_by_layer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-threshold", type=float, default=0.65)
    parser.add_argument("--xgb-exit", type=float, default=0.12)
    parser.add_argument("--assets", default="BTC,ETH,SOL")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-03-01")
    args = parser.parse_args()

    assets = args.assets.split(",")

    print("="*70)
    print("  LAYERED MULTI-TIMEFRAME BACKTEST")
    print("="*70)
    print(f"Assets: {', '.join(assets)}")
    print(f"XGBoost threshold: {args.xgb_threshold}")
    print(f"Layers: XGBoost (4h) + MR Scalp + Momentum Scalp + VWAP Scalp")
    print()

    # Load data
    btc = pd.read_parquet("research_data/BTCUSDT_15m.parquet")
    eth = pd.read_parquet("research_data/ETHUSDT_15m.parquet")
    sol = pd.read_parquet("research_data/SOLUSDT_15m.parquet")

    for df in [btc, eth, sol]:
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")

    btc = btc[(btc.index >= start_ts) & (btc.index < end_ts)]
    eth = eth[(eth.index >= start_ts) & (eth.index < end_ts)]
    sol = sol[(sol.index >= start_ts) & (sol.index < end_ts)]

    # Prepare features
    features_dict = {}
    for asset in assets:
        print(f"Computing {asset} features...")
        features_dict[asset] = prepare_features(asset, btc, eth, sol)

    portfolio_mgr = LayeredPortfolioManager(
        xgb_threshold=args.xgb_threshold,
        xgb_exit=args.xgb_exit
    )

    result = run_layered_backtest(features_dict, portfolio_mgr, assets)

    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    print(f"Total Return: {result['total_return']*100:+.2f}%")
    print(f"Sharpe: {result['sharpe']:.3f}")
    print(f"Composite Score: {result['composite_score']:.3f}")
    print(f"Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"Total Trades: {result['num_trades']}")
    print(f"Win Rate: {result['win_rate']*100:.1f}%")
    print()
    print(f"📊 COVERAGE METRICS:")
    print(f"  Daily Coverage: {result['daily_coverage_pct']*100:.1f}%")
    print(f"  Avg Days Per 10-Day Window: {result['avg_days_per_10day_window']:.1f}")
    print(f"  % of Windows with 8+ Days: {result['pct_windows_8plus_days']*100:.1f}%")
    print()

    print("📈 LAYER BREAKDOWN:")
    for layer, stats in result['trades_by_layer'].items():
        print(f"  {layer:12s}: {stats['count']:4d} trades, {stats['win_rate']*100:5.1f}% win, {stats['avg_hold_hours']:5.1f}h avg hold")

    if result['avg_days_per_10day_window'] >= 8.0:
        print("\n✅ TARGET MET: Trading 8+ days per 10-day window!")
    else:
        print(f"\n❌ Below target: {result['avg_days_per_10day_window']:.1f}/8 days per 10-day window")

    output_path = RESULTS_DIR / f"layered_t{args.xgb_threshold:.2f}_{len(assets)}assets.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✅ Results saved: {output_path}")


if __name__ == "__main__":
    main()

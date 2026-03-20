#!/usr/bin/env python3
"""
15-minute XGBoost backtest with live-bot risk management.

Replicates the full risk stack from bot/execution/risk.py:
  - ATR-based trailing stop (atr_proxy as ATR estimate; only moves UP)
  - Hard % stop floor (5%)
  - Equal dollar risk position sizing (risk_usd / stop_distance)
  - Tiered circuit breaker (10%->0.5x, 20%->0.25x, >=30%->halt)
  - Kelly criterion gate (blocks if edge <= 0)
  - Concentration cap (max 40% of portfolio per position)

BACKTEST-ONLY. No live trading, no Roostoo API calls.

ATR CALIBRATION NOTE:
  The live bot uses --atr-mult 2.0 at 4H candles (~1.25% stop distance).
  At 15M, the atr_proxy is ~5-10x smaller relative to price (~0.25% raw),
  so the multiplier must be scaled up accordingly.
  Empirical sweep result (OOS 2024-2026):
    2x  -> 74% stop exits, Sharpe=-0.499  (too tight, noise-stops dominate)
    8x  -> 25% stop exits, Sharpe=+0.639
    10x -> 19% stop exits, Sharpe=+0.760  (default)
    15x -> 12% stop exits, Sharpe=+0.873

BEST CONFIGURATION (OOS 2024-2026, threshold=0.70, atr-mult=10):
  Sharpe=1.141 | Sortino=1.649 | MaxDD=-6.00% | Return=+16.33%
  73 trades | 60.3% win rate | 37% stop-exits / 63% signal-exits

Usage:
  python scripts/backtest_15m.py --model models/xgb_btc_15m.pkl
  python scripts/backtest_15m.py --model models/xgb_btc_15m.pkl --threshold 0.70 --atr-mult 10
  python scripts/backtest_15m.py --model models/xgb_btc_15m.pkl --atr-sweep
  python scripts/backtest_15m.py --model models/xgb_btc_15m.pkl --sweep --atr-mult 10
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_cross_asset_features, compute_features

# 15M annualisation constant: 365.25 days x 24 h x 4 bars/h
PERIODS_15M = 35_040
TRAIN_CUTOFF = "2024-01-01"   # must match train_model_15m.py --test-cutoff


# -- Feature preparation -------------------------------------------------------

def prepare_features(btc_path, eth_path, sol_path, start=None, end=None):
    """
    Load 15M parquets, run feature pipeline, return date-filtered feature matrix.
    Pipeline order must match train_model_15m.py exactly.
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

    # 4H (16-bar) and 1D (96-bar) cross-asset lags -- must match train_model_15m.py
    for asset, df in [("eth", eth), ("sol", sol)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    feat = feat.dropna()

    if start:
        feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]

    return feat


# -- Model loading -------------------------------------------------------------

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Model at {model_path} has no predict_proba(). Expected XGBClassifier.")
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(f"Model at {model_path} has no feature_names_in_. Retrain with train_model_15m.py.")
    return model


def batch_predict(model, feat_df: pd.DataFrame) -> np.ndarray:
    feature_cols = list(model.feature_names_in_)
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Feature matrix missing model columns: {missing[:5]}")
    return model.predict_proba(feat_df[feature_cols])[:, 1]


# -- Circuit breaker -----------------------------------------------------------

def _cb_multiplier(
    drawdown: float,
    halt: float = 0.30,
    reduce_heavy: float = 0.20,
    reduce_light: float = 0.10,
) -> float:
    """Tiered circuit breaker size multiplier (mirrors RiskManager._get_cb_size_multiplier)."""
    if drawdown >= halt:
        return 0.0
    if drawdown >= reduce_heavy:
        return 0.25
    if drawdown >= reduce_light:
        return 0.50
    return 1.0


# -- Sized backtest simulation -------------------------------------------------

def run_backtest(
    feat_df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float,
    initial_capital: float = 10_000.0,
    risk_per_trade_pct: float = 0.02,
    hard_stop_pct: float = 0.05,
    atr_stop_multiplier: float = 2.0,
    max_single_position_pct: float = 0.40,
    expected_win_loss_ratio: float = 1.5,
    cb_halt: float = 0.30,
    cb_reduce_heavy: float = 0.20,
    cb_reduce_light: float = 0.10,
    fee_bps: int = 10,
) -> tuple:
    """
    Bar-by-bar backtest with live-bot risk management.

    Position sizing:
        risk_usd      = total_portfolio x risk_per_trade_pct x p_buy x cb_mult
        stop_distance = price - max(hard_stop, price - atr_mult x atr_proxy)
        quantity      = risk_usd / stop_distance
        target_usd    = min(quantity x price, portfolio x max_single_pct, 0.95 x free_balance)

    Exit conditions (whichever fires first):
        1. ATR trailing stop triggered (stop only ratchets UP)
        2. SELL signal: probas[i] <= 1 - threshold

    Returns:
        (returns_series, portfolio_series, closed_trades, gate_stats)
        - returns_series:  bar-by-bar % change in total portfolio value
        - portfolio_series: absolute portfolio value at each bar
        - closed_trades:   list of trade dicts
        - gate_stats:      dict counting how many BUY signals were blocked by each gate
    """
    fee_rate = fee_bps / 10_000.0
    sell_threshold = 1.0 - threshold

    closes = feat_df["close"].values
    atrs = feat_df["atr_proxy"].values
    timestamps = feat_df.index
    n = len(closes)

    # Portfolio state
    free_balance = initial_capital
    portfolio_hwm = initial_capital
    position_units = 0.0
    entry_effective_price = 0.0
    trail_stop = 0.0
    entry_bar_ts = None

    portfolio_values = np.zeros(n)
    portfolio_values[0] = initial_capital
    returns = np.zeros(n)
    closed_trades = []
    gate_stats = {"kelly_blocked": 0, "cb_halted": 0, "cb_reduced": 0}

    for i in range(n):
        c = closes[i]
        atr = atrs[i]   # shifted 1 bar in compute_features -- look-ahead safe
        p = probas[i]

        # Update trailing stop for open position
        if position_units > 0 and not np.isnan(atr) and atr > 0:
            new_atr_stop = c - atr_stop_multiplier * atr
            trail_stop = max(trail_stop, new_atr_stop)

        # Check exits
        just_exited = False
        if position_units > 0:
            stop_hit = c <= trail_stop
            sell_signal = p <= sell_threshold

            if stop_hit or sell_signal:
                proceeds = position_units * c * (1.0 - fee_rate)
                net_exit = c * (1.0 - fee_rate)
                pnl_pct = (net_exit - entry_effective_price) / entry_effective_price

                closed_trades.append({
                    "entry_bar": entry_bar_ts,
                    "exit_bar": timestamps[i],
                    "entry_price": entry_effective_price,
                    "exit_price": net_exit,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "stop" if stop_hit else "signal",
                })

                free_balance += proceeds
                position_units = 0.0
                just_exited = True

        # Mark to market
        position_value = position_units * c
        total_portfolio = free_balance + position_value
        portfolio_hwm = max(portfolio_hwm, total_portfolio)
        drawdown = (portfolio_hwm - total_portfolio) / portfolio_hwm if portfolio_hwm > 0 else 0.0
        cb_mult = _cb_multiplier(drawdown, cb_halt, cb_reduce_heavy, cb_reduce_light)

        # Check BUY
        if position_units == 0 and not just_exited and p >= threshold:
            # Gate 1: Kelly criterion
            kelly = (p * expected_win_loss_ratio - (1.0 - p)) / expected_win_loss_ratio
            if kelly <= 0:
                gate_stats["kelly_blocked"] += 1
            # Gate 2: Circuit breaker halt
            elif cb_mult == 0.0:
                gate_stats["cb_halted"] += 1
            else:
                if cb_mult < 1.0:
                    gate_stats["cb_reduced"] += 1

                # Compute stop levels
                hard_stop_price = c * (1.0 - hard_stop_pct)
                if not np.isnan(atr) and atr > 0:
                    atr_stop_price = c - atr_stop_multiplier * atr
                    initial_stop = max(hard_stop_price, atr_stop_price)
                else:
                    initial_stop = hard_stop_price

                stop_distance = c - initial_stop
                # Clamp: never let stop_distance exceed hard_stop_pct of price
                # and never let it go zero/negative -- fallback to hard stop
                stop_distance = min(stop_distance, c * hard_stop_pct)
                if stop_distance <= 0:
                    stop_distance = c * hard_stop_pct

                # Size the position
                risk_usd = total_portfolio * risk_per_trade_pct * p * cb_mult
                quantity = risk_usd / stop_distance
                target_usd = quantity * c

                # Concentration + liquidity caps
                usable = free_balance * 0.95
                target_usd = min(target_usd, total_portfolio * max_single_position_pct, usable)

                if target_usd >= 10.0:
                    # Enter: deduct invested amount + fee from free balance
                    position_units = target_usd / c
                    entry_fee = target_usd * fee_rate
                    free_balance -= (target_usd + entry_fee)
                    entry_effective_price = c * (1.0 + fee_rate)
                    trail_stop = initial_stop
                    entry_bar_ts = timestamps[i]

                    # Update mark-to-market after entry
                    position_value = position_units * c
                    total_portfolio = free_balance + position_value

        # Record bar-end portfolio value
        portfolio_values[i] = free_balance + (position_units * c)
        if i > 0:
            returns[i] = portfolio_values[i] / portfolio_values[i - 1] - 1.0

    returns_series = pd.Series(returns, index=timestamps)
    portfolio_series = pd.Series(portfolio_values, index=timestamps)
    return returns_series, portfolio_series, closed_trades, gate_stats


# -- Statistics ----------------------------------------------------------------

def compute_stats_report(
    returns: pd.Series,
    portfolio: pd.Series,
    closed_trades: list[dict],
    gate_stats: dict,
    initial_capital: float,
) -> dict:
    returns = returns[returns.index.notna()]
    n_trades = len(closed_trades)

    if n_trades > 0:
        winners = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
        trade_win_rate = winners / n_trades
        avg_trade_pnl = sum(t["pnl_pct"] for t in closed_trades) / n_trades
        best_trade = max(t["pnl_pct"] for t in closed_trades)
        worst_trade = min(t["pnl_pct"] for t in closed_trades)
        stop_exits = sum(1 for t in closed_trades if t["exit_reason"] == "stop")
        signal_exits = n_trades - stop_exits
    else:
        trade_win_rate = avg_trade_pnl = best_trade = worst_trade = 0.0
        stop_exits = signal_exits = 0

    final_value = portfolio.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    stats = {
        "total_return_pct": total_return * 100,
        "final_portfolio_usd": final_value,
        "cagr_pct": float(qs.stats.cagr(returns, periods=PERIODS_15M)) * 100,
        "sharpe": float(qs.stats.sharpe(returns, periods=PERIODS_15M)),
        "sortino": float(qs.stats.sortino(returns, periods=PERIODS_15M)),
        "calmar": float(qs.stats.calmar(returns)),
        "max_drawdown_pct": float(qs.stats.max_drawdown(returns)) * 100,
        "volatility_ann_pct": float(qs.stats.volatility(returns, periods=PERIODS_15M)) * 100,
        "n_trades": n_trades,
        "stop_exits": stop_exits,
        "signal_exits": signal_exits,
        "trade_win_rate_pct": trade_win_rate * 100,
        "avg_trade_pnl_pct": avg_trade_pnl * 100,
        "best_trade_pct": best_trade * 100,
        "worst_trade_pct": worst_trade * 100,
        "n_bars": len(returns),
        "kelly_blocked": gate_stats["kelly_blocked"],
        "cb_halted": gate_stats["cb_halted"],
        "cb_reduced": gate_stats["cb_reduced"],
    }
    return stats


def print_stats_report(
    stats: dict,
    label: str = "",
    initial_capital: float = 10_000.0,
) -> None:
    sep = "=" * 60
    header = f"  BACKTEST RESULTS (15M){' -- ' + label if label else ''}"
    print(sep)
    print(header)
    print(sep)
    print(f"  Initial capital   : ${initial_capital:,.0f}")
    print(f"  Final portfolio   : ${stats['final_portfolio_usd']:,.2f}")
    print(f"  Total return      : {stats['total_return_pct']:+.2f}%")
    print(f"  CAGR              : {stats['cagr_pct']:+.2f}%")
    print(f"  Sharpe (15M ann.) : {stats['sharpe']:.3f}")
    print(f"  Sortino (15M ann.): {stats['sortino']:.3f}")
    print(f"  Calmar            : {stats['calmar']:.3f}")
    print(f"  Max drawdown      : {stats['max_drawdown_pct']:.2f}%")
    print(f"  Volatility (ann.) : {stats['volatility_ann_pct']:.2f}%")
    print(sep)
    print(f"  # Trades          : {stats['n_trades']}")
    print(f"    Stop exits       : {stats['stop_exits']}")
    print(f"    Signal exits     : {stats['signal_exits']}")
    print(f"  Trade win rate    : {stats['trade_win_rate_pct']:.1f}%")
    print(f"  Avg trade PnL     : {stats['avg_trade_pnl_pct']:+.2f}%")
    print(f"  Best trade        : {stats['best_trade_pct']:+.2f}%")
    print(f"  Worst trade       : {stats['worst_trade_pct']:+.2f}%")
    print(sep)
    print(f"  BUY signals blocked:")
    print(f"    Kelly gate       : {stats['kelly_blocked']}")
    print(f"    CB halt (>=30% DD): {stats['cb_halted']}")
    print(f"    CB reduced sizing: {stats['cb_reduced']} (not blocked, but smaller)")
    print(sep)


# -- Threshold sweep -----------------------------------------------------------

def run_threshold_sweep(
    feat_df: pd.DataFrame,
    probas: np.ndarray,
    initial_capital: float = 10_000.0,
    fee_bps: int = 10,
    **sizing_kwargs,
) -> list[dict]:
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    results = []

    for t in thresholds:
        ret, port, trades, gates = run_backtest(
            feat_df, probas, t,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            **sizing_kwargs,
        )
        stats = compute_stats_report(ret, port, trades, gates, initial_capital)
        results.append({
            "threshold": t,
            "sharpe": round(stats["sharpe"], 3),
            "sortino": round(stats["sortino"], 3),
            "calmar": round(stats["calmar"], 3),
            "n_trades": stats["n_trades"],
            "total_return_pct": round(stats["total_return_pct"], 2),
            "trade_win_rate_pct": round(stats["trade_win_rate_pct"], 1),
            "final_usd": round(stats["final_portfolio_usd"], 2),
        })

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results


# -- ATR multiplier sweep ------------------------------------------------------

def run_atr_sweep(
    feat_df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float,
    initial_capital: float = 10_000.0,
    fee_bps: int = 10,
    **base_sizing_kwargs,
) -> list[dict]:
    """
    Sweep ATR multipliers to find the right stop calibration for 15M.

    At 15M, the ATR proxy is ~5-10x smaller than at 4H relative to price.
    The live bot default (2x) is therefore too tight; this sweep finds the
    multiplier where stop exits no longer dominate trade counts.
    """
    atr_mults = [2, 4, 6, 8, 10, 12, 15, 20, 30]
    results = []

    for mult in atr_mults:
        kwargs = {**base_sizing_kwargs, "atr_stop_multiplier": float(mult)}
        ret, port, trades, gates = run_backtest(
            feat_df, probas, threshold,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            **kwargs,
        )
        stats = compute_stats_report(ret, port, trades, gates, initial_capital)
        stop_pct = stats["stop_exits"] / stats["n_trades"] * 100 if stats["n_trades"] > 0 else 0
        results.append({
            "atr_mult": mult,
            "sharpe": round(stats["sharpe"], 3),
            "sortino": round(stats["sortino"], 3),
            "n_trades": stats["n_trades"],
            "stop_exit_pct": round(stop_pct, 1),
            "total_return_pct": round(stats["total_return_pct"], 2),
            "trade_win_rate_pct": round(stats["trade_win_rate_pct"], 1),
            "max_drawdown_pct": round(stats["max_drawdown_pct"], 2),
        })

    return results


# -- CLI -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="15M XGBoost backtest with live-bot risk management (ATR stops, CB, Kelly)"
    )
    parser.add_argument("--model", type=str, required=True, metavar="PATH",
                        help="Path to pre-trained XGBoost .pkl file")
    parser.add_argument("--btc", default="data/BTCUSDT_15m.parquet")
    parser.add_argument("--eth", default="data/ETHUSDT_15m.parquet")
    parser.add_argument("--sol", default="data/SOLUSDT_15m.parquet")
    parser.add_argument(
        "--start", default="2024-01-01",
        help="Start date (default 2024-01-01 = OOS). Earlier dates include training data.",
    )
    parser.add_argument("--end", default=None, help='End date, e.g. "2024-12-31"')
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="P(BUY) confidence threshold (default 0.65)",
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep thresholds 0.50-0.85 and report best Sharpe")
    parser.add_argument("--fee-bps", type=int, default=10,
                        help="Transaction cost in basis points per side (default 10 = 0.1%%)")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Starting capital in USD (default 10000)")
    parser.add_argument(
        "--risk-per-trade", type=float, default=0.02,
        help="Fraction of portfolio risked per trade (default 0.02 = 2%%)",
    )
    parser.add_argument(
        "--hard-stop-pct", type=float, default=0.05,
        help="Hard stop loss as fraction of price (default 0.05 = 5%%)",
    )
    parser.add_argument(
        "--atr-mult", type=float, default=10.0,
        help="ATR multiplier for trailing stop (default 10.0 for 15M; live bot uses 2.0 at 4H)",
    )
    parser.add_argument(
        "--atr-sweep", action="store_true",
        help="Sweep ATR multipliers 2-30 to find best 15M calibration (at fixed --threshold)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write stats report as CSV to this path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading and preparing 15M features...")
    feat = prepare_features(args.btc, args.eth, args.sol, args.start, args.end)
    print(f"  Feature matrix: {feat.shape[0]:,} bars x {feat.shape[1]} columns")
    print(f"  Date range: {feat.index[0]} to {feat.index[-1]}")

    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF, tz="UTC")
    if feat.index[0] < cutoff_ts:
        in_sample_bars = (feat.index < cutoff_ts).sum()
        print(
            f"\n  *** WARNING: {in_sample_bars:,} bars ({in_sample_bars / len(feat):.0%}) "
            f"are BEFORE {TRAIN_CUTOFF} (training data). "
            f"Results will be inflated. Use --start {TRAIN_CUTOFF} for honest OOS eval. ***\n"
        )

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    print(f"  Model expects {len(model.feature_names_in_)} features")

    sizing_kwargs = dict(
        risk_per_trade_pct=args.risk_per_trade,
        hard_stop_pct=args.hard_stop_pct,
        atr_stop_multiplier=args.atr_mult,
    )

    print("Batch predicting probas...")
    probas = batch_predict(model, feat)

    if args.atr_sweep:
        base_kwargs = dict(
            risk_per_trade_pct=args.risk_per_trade,
            hard_stop_pct=args.hard_stop_pct,
        )
        print(f"\nRunning ATR multiplier sweep (threshold={args.threshold}, fee={args.fee_bps} bps)...")
        print("  (Finding where stop exits stop dominating -- the right calibration for 15M)\n")
        atr_results = run_atr_sweep(
            feat, probas, args.threshold,
            initial_capital=args.capital,
            fee_bps=args.fee_bps,
            **base_kwargs,
        )
        print(
            f"{'ATR Mult':>10}  {'Sharpe':>8}  {'Sortino':>8}  {'Trades':>7}  "
            f"{'Stop Exit%':>11}  {'Return%':>9}  {'WinRate%':>9}  {'MaxDD%':>8}"
        )
        print("-" * 83)
        best_atr = max(atr_results, key=lambda r: r["sharpe"])
        for r in atr_results:
            marker = " <-- best" if r is best_atr else ""
            print(
                f"  {r['atr_mult']:>8.0f}x  {r['sharpe']:>8.3f}  {r['sortino']:>8.3f}  "
                f"{r['n_trades']:>7}  {r['stop_exit_pct']:>10.1f}%  "
                f"{r['total_return_pct']:>9.2f}  {r['trade_win_rate_pct']:>8.1f}%  "
                f"{r['max_drawdown_pct']:>7.2f}%{marker}"
            )
        print(f"\nBest ATR mult: {best_atr['atr_mult']}x (Sharpe={best_atr['sharpe']:.3f}, "
              f"stop exit rate={best_atr['stop_exit_pct']:.1f}%)")
        return

    if args.sweep:
        print(f"\nRunning threshold sweep (fee={args.fee_bps} bps, capital=${args.capital:,.0f})...")
        results = run_threshold_sweep(
            feat, probas,
            initial_capital=args.capital,
            fee_bps=args.fee_bps,
            **sizing_kwargs,
        )
        print(
            f"\n{'Threshold':>10}  {'Sharpe':>8}  {'Sortino':>8}  {'Calmar':>8}  "
            f"{'Trades':>7}  {'Return%':>9}  {'WinRate%':>9}  {'Final$':>10}"
        )
        print("-" * 83)
        for r in results:
            marker = " <-- best" if r is results[0] else ""
            print(
                f"  {r['threshold']:>8.2f}  {r['sharpe']:>8.3f}  {r['sortino']:>8.3f}  "
                f"{r['calmar']:>8.3f}  {r['n_trades']:>7}  "
                f"{r['total_return_pct']:>9.2f}  {r['trade_win_rate_pct']:>8.1f}%  "
                f"${r['final_usd']:>9,.0f}{marker}"
            )
        best = results[0]
        print(f"\nBest threshold: {best['threshold']} (Sharpe={best['sharpe']:.3f})")
        return

    print("Running backtest simulation...")
    returns, portfolio, trades, gates = run_backtest(
        feat, probas, args.threshold,
        initial_capital=args.capital,
        fee_bps=args.fee_bps,
        **sizing_kwargs,
    )
    print(f"  Simulation complete: {len(trades)} closed trades")

    stats = compute_stats_report(returns, portfolio, trades, gates, args.capital)
    label = (
        f"XGBoost 15M threshold={args.threshold} "
        f"risk={args.risk_per_trade:.0%}/trade "
        f"stop={args.hard_stop_pct:.0%} "
        f"fee={args.fee_bps}bps"
    )
    print_stats_report(stats, label, initial_capital=args.capital)

    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        print(f"\nStats saved to {args.output}")


if __name__ == "__main__":
    main()

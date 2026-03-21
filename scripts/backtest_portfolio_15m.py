#!/usr/bin/env python3
"""
Two-asset portfolio backtest: BTC/USD (XGBoost) + SOL/USD (XGBoost) in parallel.

Both models run simultaneously with shared capital.  Each asset has its own
position slot and ATR trailing stop.  The circuit breaker, Kelly gate, and
sizing logic operate on the total portfolio value.

Usage:
  python scripts/backtest_portfolio_15m.py
  python scripts/backtest_portfolio_15m.py --btc-threshold 0.65 --sol-threshold 0.70
  python scripts/backtest_portfolio_15m.py --sweep
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

from bot.data.features import compute_btc_context_features, compute_cross_asset_features, compute_features

PERIODS_15M = 35_040
TRAIN_CUTOFF = "2024-01-01"
CORR_WINDOW = 2880


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_btc_features(btc_path, eth_path, sol_path) -> pd.DataFrame:
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)
    sol = pd.read_parquet(sol_path)
    for df in (btc, eth, sol):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    feat = compute_features(btc)
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    for asset, df in [("eth", eth), ("sol", sol)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)
    feat = compute_btc_context_features(feat, eth, sol, window=CORR_WINDOW)
    return feat.dropna()


def prepare_sol_features(btc_path, eth_path, sol_path) -> pd.DataFrame:
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)
    sol = pd.read_parquet(sol_path)
    for df in (btc, eth, sol):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    feat = compute_features(sol)
    feat = compute_cross_asset_features(feat, {"BTC/USD": btc, "ETH/USD": eth})
    for asset, df in [("btc", btc), ("eth", eth)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    sol_ret = np.log(sol["close"] / sol["close"].shift(1)).reindex(feat.index)
    btc_ret = np.log(btc["close"] / btc["close"].shift(1)).reindex(feat.index)
    corr = sol_ret.rolling(CORR_WINDOW).corr(btc_ret)
    cov  = sol_ret.rolling(CORR_WINDOW).cov(btc_ret)
    var_btc = btc_ret.rolling(CORR_WINDOW).var()
    feat["sol_btc_corr"] = corr.shift(1)
    feat["sol_btc_beta"] = (cov / (var_btc + 1e-10)).shift(1)
    return feat.dropna()


# ── Risk helpers ───────────────────────────────────────────────────────────────

def _cb_mult(dd, halt=0.30, heavy=0.20, light=0.10):
    if dd >= halt:   return 0.0
    if dd >= heavy:  return 0.25
    if dd >= light:  return 0.50
    return 1.0


# ── Portfolio backtest ─────────────────────────────────────────────────────────

def run_portfolio_backtest(
    btc_feat: pd.DataFrame,
    sol_feat: pd.DataFrame,
    btc_probas: np.ndarray,
    sol_probas: np.ndarray,
    btc_threshold: float,
    sol_threshold: float,
    initial_capital: float = 10_000.0,
    risk_per_trade_pct: float = 0.02,
    hard_stop_pct: float = 0.05,
    atr_stop_multiplier: float = 10.0,
    max_single_position_pct: float = 0.40,
    expected_win_loss_ratio: float = 1.5,
    fee_bps: int = 10,
    exit_threshold: float = 0.10,
) -> tuple:
    """
    Bar-by-bar portfolio simulation with two independent position slots.

    Both assets are evaluated every bar on the shared timestamp union.
    Capital is shared — sizing for each new position uses the current free balance.
    Circuit breaker is portfolio-level (total drawdown from HWM).
    """
    fee_rate = fee_bps / 10_000.0

    # Align both feature sets to a common OOS index
    common_idx = btc_feat.index.intersection(sol_feat.index)
    btc_feat = btc_feat.reindex(common_idx)
    sol_feat = sol_feat.reindex(common_idx)
    btc_p = pd.Series(btc_probas, index=btc_feat.index).reindex(common_idx).fillna(0.0).values
    sol_p = pd.Series(sol_probas, index=sol_feat.index).reindex(common_idx).fillna(0.0).values

    btc_closes = btc_feat["close"].values
    sol_closes = sol_feat["close"].values
    btc_atrs   = btc_feat["atr_proxy"].values
    sol_atrs   = sol_feat["atr_proxy"].values
    timestamps = common_idx
    n = len(timestamps)

    # Portfolio state
    free_balance   = initial_capital
    portfolio_hwm  = initial_capital

    # Per-asset position state
    pos = {
        "btc": dict(units=0.0, entry_price=0.0, trail_stop=0.0, entry_ts=None),
        "sol": dict(units=0.0, entry_price=0.0, trail_stop=0.0, entry_ts=None),
    }

    portfolio_values = np.zeros(n)
    portfolio_values[0] = initial_capital
    returns = np.zeros(n)
    closed_trades = []
    gate_stats = {"kelly_blocked": 0, "cb_halted": 0, "cb_reduced": 0}

    assets = [
        ("btc", btc_closes, btc_atrs, btc_p, btc_threshold),
        ("sol", sol_closes, sol_atrs, sol_p, sol_threshold),
    ]

    for i in range(n):
        ts = timestamps[i]

        # --- Update trailing stops ---
        for name, closes, atrs, _, _ in assets:
            s = pos[name]
            c, atr = closes[i], atrs[i]
            if s["units"] > 0 and not np.isnan(atr) and atr > 0:
                new_stop = c - atr_stop_multiplier * atr
                s["trail_stop"] = max(s["trail_stop"], new_stop)

        # --- Mark to market ---
        btc_val = pos["btc"]["units"] * btc_closes[i]
        sol_val = pos["sol"]["units"] * sol_closes[i]
        total_portfolio = free_balance + btc_val + sol_val
        portfolio_hwm = max(portfolio_hwm, total_portfolio)
        drawdown = (portfolio_hwm - total_portfolio) / portfolio_hwm if portfolio_hwm > 0 else 0.0
        cb = _cb_mult(drawdown)

        # --- Exits ---
        for name, closes, atrs, probas, threshold in assets:
            s = pos[name]
            c = closes[i]
            p = probas[i]
            if s["units"] <= 0:
                continue

            stop_hit   = c <= s["trail_stop"]
            sell_signal = p <= exit_threshold

            if stop_hit or sell_signal:
                proceeds = s["units"] * c * (1.0 - fee_rate)
                net_exit = c * (1.0 - fee_rate)
                pnl_pct  = (net_exit - s["entry_price"]) / s["entry_price"]

                closed_trades.append({
                    "asset": name,
                    "entry_ts": s["entry_ts"],
                    "exit_ts": ts,
                    "entry_price": s["entry_price"],
                    "exit_price": net_exit,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "stop" if stop_hit else "signal",
                })

                free_balance += proceeds
                s["units"] = 0.0
                s["entry_price"] = 0.0
                s["trail_stop"] = 0.0

        # Recalculate portfolio after exits
        total_portfolio = free_balance + pos["btc"]["units"] * btc_closes[i] + pos["sol"]["units"] * sol_closes[i]

        # --- Entries ---
        for name, closes, atrs, probas, threshold in assets:
            s = pos[name]
            c   = closes[i]
            atr = atrs[i]
            p   = probas[i]
            if s["units"] > 0 or p < threshold:
                continue

            kelly = (p * expected_win_loss_ratio - (1.0 - p)) / expected_win_loss_ratio
            if kelly <= 0:
                gate_stats["kelly_blocked"] += 1
                continue
            if cb == 0.0:
                gate_stats["cb_halted"] += 1
                continue
            if cb < 1.0:
                gate_stats["cb_reduced"] += 1

            hard_stop_price = c * (1.0 - hard_stop_pct)
            if not np.isnan(atr) and atr > 0:
                atr_stop_price = c - atr_stop_multiplier * atr
                initial_stop = max(hard_stop_price, atr_stop_price)
            else:
                initial_stop = hard_stop_price

            stop_distance = min(c - initial_stop, c * hard_stop_pct)
            if stop_distance <= 0:
                stop_distance = c * hard_stop_pct

            risk_usd   = total_portfolio * risk_per_trade_pct * p * cb
            quantity   = risk_usd / stop_distance
            target_usd = quantity * c
            usable     = free_balance * 0.95
            target_usd = min(target_usd, total_portfolio * max_single_position_pct, usable)

            if target_usd >= 10.0:
                s["units"]       = target_usd / c
                s["entry_price"] = c * (1.0 + fee_rate)
                s["trail_stop"]  = initial_stop
                s["entry_ts"]    = ts
                free_balance    -= target_usd * (1.0 + fee_rate)

        portfolio_values[i] = free_balance + pos["btc"]["units"] * btc_closes[i] + pos["sol"]["units"] * sol_closes[i]
        if i > 0:
            returns[i] = portfolio_values[i] / portfolio_values[i - 1] - 1.0

    return (
        pd.Series(returns, index=timestamps),
        pd.Series(portfolio_values, index=timestamps),
        closed_trades,
        gate_stats,
    )


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(returns, portfolio, closed_trades, initial_capital):
    n = len(closed_trades)
    if n > 0:
        winners    = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
        win_rate   = winners / n
        avg_pnl    = sum(t["pnl_pct"] for t in closed_trades) / n
        stop_exits = sum(1 for t in closed_trades if t["exit_reason"] == "stop")
        btc_trades = [t for t in closed_trades if t["asset"] == "btc"]
        sol_trades = [t for t in closed_trades if t["asset"] == "sol"]
    else:
        win_rate = avg_pnl = 0.0
        stop_exits = 0
        btc_trades = sol_trades = []

    return {
        "sharpe":           float(qs.stats.sharpe(returns, periods=PERIODS_15M)),
        "sortino":          float(qs.stats.sortino(returns, periods=PERIODS_15M)),
        "max_drawdown_pct": float(qs.stats.max_drawdown(returns)) * 100,
        "total_return_pct": (portfolio.iloc[-1] - initial_capital) / initial_capital * 100,
        "n_trades":         n,
        "btc_trades":       len(btc_trades),
        "sol_trades":       len(sol_trades),
        "stop_exits":       stop_exits,
        "win_rate_pct":     win_rate * 100,
        "avg_pnl_pct":      avg_pnl * 100,
    }


def print_report(stats, btc_threshold, sol_threshold, initial_capital):
    sep = "=" * 60
    print(sep)
    print(f"  PORTFOLIO BACKTEST: BTC(t={btc_threshold}) + SOL(t={sol_threshold})")
    print(sep)
    print(f"  Sharpe (15M ann.)  : {stats['sharpe']:.3f}")
    print(f"  Sortino            : {stats['sortino']:.3f}")
    print(f"  Max drawdown       : {stats['max_drawdown_pct']:.2f}%")
    print(f"  Total return       : {stats['total_return_pct']:+.2f}%")
    print(sep)
    print(f"  Total trades       : {stats['n_trades']}")
    print(f"    BTC trades       : {stats['btc_trades']}")
    print(f"    SOL trades       : {stats['sol_trades']}")
    print(f"    Stop exits       : {stats['stop_exits']}")
    print(f"  Win rate           : {stats['win_rate_pct']:.1f}%")
    print(f"  Avg trade PnL      : {stats['avg_pnl_pct']:+.2f}%")
    print(sep)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BTC + SOL portfolio backtest")
    p.add_argument("--btc-model",      default="models/xgb_btc_15m_iter5.pkl")
    p.add_argument("--sol-model",      default="models/xgb_sol_15m.pkl")
    p.add_argument("--btc",            default="data/BTCUSDT_15m.parquet")
    p.add_argument("--eth",            default="data/ETHUSDT_15m.parquet")
    p.add_argument("--sol",            default="data/SOLUSDT_15m.parquet")
    p.add_argument("--btc-threshold",  type=float, default=0.65)
    p.add_argument("--sol-threshold",  type=float, default=0.70)
    p.add_argument("--capital",        type=float, default=10_000.0)
    p.add_argument("--start",          default=TRAIN_CUTOFF)
    p.add_argument("--sweep",          action="store_true",
                   help="Sweep BTC and SOL thresholds independently and print grid")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading features...")
    btc_feat_full = prepare_btc_features(args.btc, args.eth, args.sol)
    sol_feat_full = prepare_sol_features(args.btc, args.eth, args.sol)

    cutoff = pd.Timestamp(args.start, tz="UTC")
    btc_feat = btc_feat_full[btc_feat_full.index >= cutoff]
    sol_feat = sol_feat_full[sol_feat_full.index >= cutoff]

    print(f"  BTC OOS bars: {len(btc_feat):,}  ({btc_feat.index[0].date()} to {btc_feat.index[-1].date()})")
    print(f"  SOL OOS bars: {len(sol_feat):,}  ({sol_feat.index[0].date()} to {sol_feat.index[-1].date()})")

    print("Loading models...")
    with open(args.btc_model, "rb") as f:
        btc_model = pickle.load(f)
    with open(args.sol_model, "rb") as f:
        sol_model = pickle.load(f)
    print(f"  BTC model: {len(btc_model.feature_names_in_)} features")
    print(f"  SOL model: {len(sol_model.feature_names_in_)} features")

    btc_probas = btc_model.predict_proba(btc_feat[list(btc_model.feature_names_in_)])[:, 1]
    sol_probas = sol_model.predict_proba(sol_feat[list(sol_model.feature_names_in_)])[:, 1]

    if args.sweep:
        btc_thresholds = [0.60, 0.65, 0.70, 0.75]
        sol_thresholds = [0.65, 0.70, 0.75, 0.80]
        print("\nThreshold grid sweep (BTC threshold × SOL threshold):\n")
        header = f"{'BTC-t':>6}  {'SOL-t':>6}  {'Sharpe':>7}  {'Sortino':>8}  {'Trades':>7}  {'BTC':>5}  {'SOL':>5}  {'Return%':>8}"
        print(header)
        print("-" * len(header))
        results = []
        for bt in btc_thresholds:
            for st in sol_thresholds:
                ret, port, trades, gates = run_portfolio_backtest(
                    btc_feat, sol_feat, btc_probas, sol_probas,
                    bt, st, initial_capital=args.capital, exit_threshold=0.10,
                )
                stats = compute_stats(ret, port, trades, args.capital)
                results.append((bt, st, stats))

        results.sort(key=lambda r: r[2]["sharpe"], reverse=True)
        for bt, st, s in results:
            print(f"  {bt:>4.2f}   {st:>4.2f}   {s['sharpe']:>7.3f}  {s['sortino']:>8.3f}  "
                  f"{s['n_trades']:>7}  {s['btc_trades']:>5}  {s['sol_trades']:>5}  "
                  f"{s['total_return_pct']:>8.2f}")
    else:
        print(f"\nRunning portfolio backtest (BTC t={args.btc_threshold}, SOL t={args.sol_threshold})...")
        ret, port, trades, gates = run_portfolio_backtest(
            btc_feat, sol_feat, btc_probas, sol_probas,
            args.btc_threshold, args.sol_threshold,
            initial_capital=args.capital, exit_threshold=0.10,
        )
        stats = compute_stats(ret, port, trades, args.capital)
        print_report(stats, args.btc_threshold, args.sol_threshold, args.capital)

        # Also print solo BTC for comparison
        print("\nFor comparison — BTC-only at same threshold:")
        from scripts.backtest_15m import run_backtest as btc_backtest, compute_stats_report
        ret_b, port_b, trades_b, gates_b = btc_backtest(
            btc_feat, btc_probas, args.btc_threshold,
            initial_capital=args.capital,
            atr_stop_multiplier=10.0,
            exit_threshold=0.10,
        )
        s_btc = compute_stats_report(ret_b, port_b, trades_b, gates_b, args.capital)
        print(f"  Sharpe: {s_btc['sharpe']:.3f}  |  Trades: {s_btc['n_trades']}  |  Return: {s_btc['total_return_pct']:+.2f}%")


if __name__ == "__main__":
    main()

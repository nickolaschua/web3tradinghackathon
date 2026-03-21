#!/usr/bin/env python3
"""
Tune size-overlay momentum rule on shared engine.

Base rule:
    if base_signal:
        size = base_size
        if funding_z < funding_z_threshold: size *= funding_mult
        if macro_condition: size *= macro_mult
        if vol_spike > vol_threshold: size *= vol_mult
        BUY(size)
"""

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs
import yaml
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bot.data.features import compute_cross_asset_features, compute_features
from bot.execution.portfolio import PortfolioAllocator
from bot.execution.risk import RiskDecision, RiskManager

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

DATA_DIR = Path("research_data")
RESULTS_DIR = Path("research_results")
CONFIG_PATH = Path("bot/config/config.yaml")
PERIODS_4H = 2190


def safe_metric(v):
    if v is None:
        return 0.0
    if isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v)):
        return 0.0
    return float(v)


def load_framework_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data():
    btc = pd.read_parquet(DATA_DIR / "BTCUSDT_4h.parquet")
    btc.columns = btc.columns.str.lower()
    eth = pd.read_parquet(DATA_DIR / "ETHUSDT_4h.parquet") if (DATA_DIR / "ETHUSDT_4h.parquet").exists() else None
    sol = pd.read_parquet(DATA_DIR / "SOLUSDT_4h.parquet") if (DATA_DIR / "SOLUSDT_4h.parquet").exists() else None
    oil = pd.read_parquet(DATA_DIR / "oil_daily.parquet") if (DATA_DIR / "oil_daily.parquet").exists() else None
    dxy = pd.read_parquet(DATA_DIR / "dxy_daily.parquet") if (DATA_DIR / "dxy_daily.parquet").exists() else None
    btc_funding = pd.read_parquet(DATA_DIR / "BTCUSDT_funding.parquet") if (DATA_DIR / "BTCUSDT_funding.parquet").exists() else None
    return btc, eth, sol, oil, dxy, btc_funding


def build_df(start: str):
    btc, eth, sol, oil, dxy, btc_funding = load_data()

    feat = compute_features(btc.copy())
    others = {}
    if eth is not None:
        others["ETH/USD"] = eth
    if sol is not None:
        others["SOL/USD"] = sol
    feat = compute_cross_asset_features(feat, others)

    for name, mdf in [("oil", oil), ("dxy", dxy)]:
        if mdf is None:
            continue
        close = mdf["close"].copy()
        aligned = close.resample("4h").ffill().reindex(feat.index, method="ffill")
        feat[f"{name}_return_1d"] = aligned.pct_change(6).shift(1)

    if btc_funding is not None:
        fr = btc_funding["funding_rate"].copy()
        aligned = fr.resample("4h").ffill().reindex(feat.index, method="ffill")
        rm = aligned.rolling(90).mean()
        rs = aligned.rolling(90).std()
        feat["btc_funding_zscore"] = ((aligned - rm) / rs.replace(0, np.nan)).shift(1)

    feat["btc_close"] = feat["close"]
    if eth is not None:
        feat["eth_close"] = eth["close"].reindex(feat.index)
    if sol is not None:
        feat["sol_close"] = sol["close"].reindex(feat.index)

    feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    needed = [
        "close",
        "atr_proxy",
        "RSI_14",
        "MACDh_12_26_9",
        "EMA_20",
        "EMA_50",
        "volume_ratio",
        "oil_return_1d",
        "dxy_return_1d",
        "btc_funding_zscore",
        "btc_close",
        "eth_close",
        "sol_close",
    ]
    return feat[needed].dropna().copy()


def build_weekly_windows(df, window_days=7, step_days=7, min_train_bars=500):
    windows = []
    idx = df.index
    if len(idx) == 0:
        return windows
    start_ts = idx.min().normalize() + pd.Timedelta(days=window_days)
    end_ts = idx.max().normalize()
    cursor = start_ts
    while cursor <= end_ts:
        test_start = cursor - pd.Timedelta(days=window_days)
        test_end = cursor
        train_mask = idx < test_start
        test_mask = (idx >= test_start) & (idx < test_end)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) >= min_train_bars and len(test_idx) > 0:
            windows.append((train_idx, test_idx))
        cursor += pd.Timedelta(days=step_days)
    return windows


def compute_stats(returns, trades):
    returns = returns[returns.index.notna()]
    sharpe = safe_metric(qs.stats.sharpe(returns, periods=PERIODS_4H))
    sortino = safe_metric(qs.stats.sortino(returns, periods=PERIODS_4H))
    calmar = safe_metric(qs.stats.calmar(returns))
    return {
        "total_return_pct": safe_metric(((1 + returns).prod() - 1) * 100),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "composite_score": 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar,
        "max_drawdown_pct": safe_metric(qs.stats.max_drawdown(returns) * 100),
        "n_trades": len(trades),
    }


def run_backtest(test_df, portfolio_weight, config, params, fee_bps=10, initial_capital=10_000.0):
    fee_rate = fee_bps / 10_000
    rm = RiskManager(config=config)
    rm.initialize_hwm(initial_capital)
    pair = "BTC/USD"

    free_balance = initial_capital
    position_units = 0.0
    entry_price = None
    prev_portfolio = initial_capital
    returns = []
    trades = []
    activity_days = set()
    utilization = []

    for idx, row in test_df.iterrows():
        close = float(row["close"])
        atr = float(row["atr_proxy"]) if not pd.isna(row["atr_proxy"]) else np.nan

        # Base signal (always considered for entries)
        in_uptrend = row["EMA_20"] > row["EMA_50"]
        base_signal = in_uptrend and (row["RSI_14"] < 50) and (row["MACDh_12_26_9"] > 0)
        exit_signal = (row["RSI_14"] > 65) or (row["MACDh_12_26_9"] < 0)

        just_exited = False
        if position_units > 0:
            stop_result = rm.check_stops(pair, close, atr)
            if stop_result.should_exit or exit_signal:
                exit_price = close * (1 - fee_rate)
                proceeds = position_units * close * (1 - fee_rate)
                pnl_pct = (exit_price - entry_price) / entry_price
                trades.append({"pnl_pct": pnl_pct})
                activity_days.add(pd.Timestamp(idx).normalize())
                free_balance += proceeds
                position_units = 0.0
                entry_price = None
                rm.record_exit(pair)
                just_exited = True

        total_portfolio = free_balance + position_units * close
        rm.check_circuit_breaker(total_portfolio)

        if position_units == 0.0 and (not just_exited) and base_signal:
            size_mult = params["base_size"]
            if row["btc_funding_zscore"] < params["funding_z_threshold"]:
                size_mult *= params["funding_mult"]
            macro_condition = (row["oil_return_1d"] > params["macro_threshold"]) and (
                row["dxy_return_1d"] < -params["macro_threshold"]
            )
            if macro_condition:
                size_mult *= params["macro_mult"]
            if row["volume_ratio"] > params["vol_threshold"]:
                size_mult *= params["vol_mult"]

            sizing = rm.size_new_position(
                pair=pair,
                current_price=close,
                current_atr=atr,
                free_balance_usd=free_balance,
                open_positions={},
                regime_multiplier=1.0,
                confidence=0.70,
                portfolio_weight=portfolio_weight * size_mult,
            )
            if sizing.decision == RiskDecision.APPROVED and sizing.approved_usd_value >= 10.0:
                target_usd = sizing.approved_usd_value
                position_units = target_usd / close
                free_balance -= (target_usd + target_usd * fee_rate)
                entry_price = close * (1 + fee_rate)
                activity_days.add(pd.Timestamp(idx).normalize())
                rm.record_entry(pair, entry_price, sizing.trailing_stop_price)

        end_portfolio = free_balance + position_units * close
        returns.append((end_portfolio / prev_portfolio - 1.0) if prev_portfolio > 0 else 0.0)
        prev_portfolio = end_portfolio
        util = (position_units * close / end_portfolio) if end_portfolio > 0 else 0.0
        utilization.append(util)

    stats = compute_stats(pd.Series(returns, index=test_df.index), trades)
    cal_days = int(test_df.index.normalize().nunique())
    stats["activity_days"] = len(activity_days)
    stats["calendar_days"] = cal_days
    stats["daily_coverage"] = (len(activity_days) / cal_days) if cal_days > 0 else 0.0
    stats["trades_per_day"] = (stats["n_trades"] / cal_days) if cal_days > 0 else 0.0
    stats["avg_utilization"] = float(np.mean(utilization)) if utilization else 0.0
    return stats


def evaluate_params(df, config, splits, params, fee_bps, min_daily_coverage, min_trades_per_day):
    fold_stats = []
    total_trades = 0
    for train_idx, test_idx in splits:
        train_slice = df.iloc[train_idx]
        test_slice = df.iloc[test_idx]
        pa = PortfolioAllocator(config=config)
        pa.compute_weights(
            {
                "BTC/USD": pd.DataFrame({"close": train_slice["btc_close"].dropna()}),
                "ETH/USD": pd.DataFrame({"close": train_slice["eth_close"].dropna()}),
                "SOL/USD": pd.DataFrame({"close": train_slice["sol_close"].dropna()}),
            }
        )
        btc_w = pa.get_pair_weight("BTC/USD", n_active_pairs=3)
        stats = run_backtest(test_slice, btc_w, config, params, fee_bps=fee_bps)
        fold_stats.append(stats)
        total_trades += stats["n_trades"]

    if not fold_stats:
        return None

    avg_cov = float(np.mean([s["daily_coverage"] for s in fold_stats]))
    avg_tpd = float(np.mean([s["trades_per_day"] for s in fold_stats]))
    avg_util = float(np.mean([s["avg_utilization"] for s in fold_stats]))
    out = {
        "params": params,
        "composite_score": float(np.mean([s["composite_score"] for s in fold_stats])),
        "sharpe": float(np.mean([s["sharpe"] for s in fold_stats])),
        "sortino": float(np.mean([s["sortino"] for s in fold_stats])),
        "calmar": float(np.mean([s["calmar"] for s in fold_stats])),
        "total_return_pct": float(np.mean([s["total_return_pct"] for s in fold_stats])),
        "max_drawdown_pct": float(np.mean([s["max_drawdown_pct"] for s in fold_stats])),
        "n_trades_total": int(total_trades),
        "avg_daily_coverage": avg_cov,
        "avg_trades_per_day": avg_tpd,
        "avg_utilization": avg_util,
        "n_windows": len(fold_stats),
        "feasible": (avg_cov >= min_daily_coverage) and (avg_tpd >= min_trades_per_day),
    }
    return out


def pick_winner(results):
    feasible = [r for r in results if r["feasible"]]
    target = feasible if feasible else results
    return sorted(target, key=lambda r: (r["composite_score"], r["avg_utilization"]), reverse=True)[0]


def eval_on_split_subset(df, config, splits, params, fee_bps, min_daily_coverage, min_trades_per_day):
    return evaluate_params(
        df=df,
        config=config,
        splits=splits,
        params=params,
        fee_bps=fee_bps,
        min_daily_coverage=min_daily_coverage,
        min_trades_per_day=min_trades_per_day,
    )


def main():
    parser = argparse.ArgumentParser(description="Tune size overlay rule")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--fee-bps", type=int, default=10)
    parser.add_argument("--evaluation-mode", choices=["cv", "weekly"], default="weekly")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--min-train-bars", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--min-daily-coverage", type=float, default=1.0)
    parser.add_argument("--min-trades-per-day", type=float, default=1.0)
    args = parser.parse_args()

    config = load_framework_config()
    df = build_df(args.start)

    if args.evaluation_mode == "weekly":
        splits = build_weekly_windows(df, args.window_days, args.step_days, args.min_train_bars)
    else:
        splits = list(TimeSeriesSplit(n_splits=args.n_splits, gap=24).split(df))

    baseline = {
        "base_size": 1.0,
        "funding_z_threshold": -1.0,
        "funding_mult": 1.5,
        "macro_threshold": 0.0,
        "macro_mult": 1.3,
        "vol_threshold": 1.2,
        "vol_mult": 1.2,
    }

    sweeps = [
        ("funding_z_threshold", [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25]),
        ("funding_mult", [1.0 + 0.1 * i for i in range(10)]),
        ("macro_threshold", [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]),
        ("macro_mult", [1.0 + 0.1 * i for i in range(10)]),
        ("vol_threshold", [1.0 + 0.1 * i for i in range(10)]),
        ("vol_mult", [1.0 + 0.1 * i for i in range(10)]),
    ]

    sweep_results = {}
    winners = {}
    for pname, vals in sweeps:
        res = []
        for v in vals:
            p = dict(baseline)
            p[pname] = float(v)
            out = evaluate_params(
                df=df,
                config=config,
                splits=splits,
                params=p,
                fee_bps=args.fee_bps,
                min_daily_coverage=args.min_daily_coverage,
                min_trades_per_day=args.min_trades_per_day,
            )
            if out is not None:
                res.append(out)
        res.sort(key=lambda r: (r["composite_score"], r["avg_utilization"]), reverse=True)
        sweep_results[pname] = res
        winners[pname] = pick_winner(res)

    combined_best_params = dict(baseline)
    for pname, _ in sweeps:
        combined_best_params[pname] = winners[pname]["params"][pname]

    combined_best = evaluate_params(
        df=df,
        config=config,
        splits=splits,
        params=combined_best_params,
        fee_bps=args.fee_bps,
        min_daily_coverage=args.min_daily_coverage,
        min_trades_per_day=args.min_trades_per_day,
    )

    # Overfit checks
    n = len(splits)
    n_tune = max(1, int(n * 0.6))
    tune_splits = splits[:n_tune]
    holdout_splits = splits[n_tune:] if n_tune < n else []
    tune_perf = eval_on_split_subset(
        df, config, tune_splits, combined_best_params, args.fee_bps, args.min_daily_coverage, args.min_trades_per_day
    )
    holdout_perf = (
        eval_on_split_subset(
            df, config, holdout_splits, combined_best_params, args.fee_bps, args.min_daily_coverage, args.min_trades_per_day
        )
        if holdout_splits
        else None
    )

    neighbors = []
    for dz in [-0.25, 0.0, 0.25]:
        for dv in [-0.1, 0.0, 0.1]:
            p = dict(combined_best_params)
            p["funding_z_threshold"] = p["funding_z_threshold"] + dz
            p["vol_threshold"] = max(0.8, p["vol_threshold"] + dv)
            r = evaluate_params(
                df=df,
                config=config,
                splits=splits,
                params=p,
                fee_bps=args.fee_bps,
                min_daily_coverage=args.min_daily_coverage,
                min_trades_per_day=args.min_trades_per_day,
            )
            if r:
                neighbors.append(r)
    neighbor_composites = [r["composite_score"] for r in neighbors]
    stability = {
        "n_neighbors": len(neighbors),
        "mean_composite": float(np.mean(neighbor_composites)) if neighbor_composites else 0.0,
        "std_composite": float(np.std(neighbor_composites)) if neighbor_composites else 0.0,
        "min_composite": float(np.min(neighbor_composites)) if neighbor_composites else 0.0,
        "max_composite": float(np.max(neighbor_composites)) if neighbor_composites else 0.0,
    }

    payload = {
        "baseline": baseline,
        "evaluation": {
            "mode": args.evaluation_mode,
            "window_days": args.window_days,
            "step_days": args.step_days,
            "min_train_bars": args.min_train_bars,
            "min_daily_coverage": args.min_daily_coverage,
            "min_trades_per_day": args.min_trades_per_day,
            "n_windows": len(splits),
        },
        "winners": winners,
        "sweeps": sweep_results,
        "combined_best": combined_best,
        "combined_best_params": combined_best_params,
        "overfit_checks": {
            "tune_perf": tune_perf,
            "holdout_perf": holdout_perf,
            "local_sensitivity": stability,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "size_overlay_tuning.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 110)
    print("  SIZE OVERLAY TUNING (SHARED ENGINE)")
    print("=" * 110)
    print(f"Saved full sweep: {out_path}")
    print(f"Windows: {len(splits)} | Mode: {args.evaluation_mode}")
    print(
        f"{'Sweep':<22} {'Best':>10} {'Comp':>8} {'Trades':>8} {'Cov%':>7} {'T/D':>6} {'Util%':>7} {'OK':>4}"
    )
    print("-" * 110)
    for pname, _ in sweeps:
        w = winners[pname]
        print(
            f"{pname:<22} {w['params'][pname]:>10.3f} {w['composite_score']:>8.3f} {w['n_trades_total']:>8} "
            f"{100*w['avg_daily_coverage']:>6.1f}% {w['avg_trades_per_day']:>6.2f} {100*w['avg_utilization']:>6.1f}% "
            f"{'yes' if w['feasible'] else 'no':>4}"
        )
    if combined_best:
        print("\nCombined best params:")
        print(combined_best_params)
        print(
            f"Combined: comp={combined_best['composite_score']:.3f} trades={combined_best['n_trades_total']} "
            f"cov={100*combined_best['avg_daily_coverage']:.1f}% tpd={combined_best['avg_trades_per_day']:.2f} "
            f"util={100*combined_best['avg_utilization']:.1f}% feasible={combined_best['feasible']}"
        )


if __name__ == "__main__":
    main()

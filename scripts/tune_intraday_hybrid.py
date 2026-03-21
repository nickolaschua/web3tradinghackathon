#!/usr/bin/env python3
"""
Frequency-first tuner for hybrid intraday strategy.

Goal order:
1) satisfy weekly feasibility constraints:
   - daily_coverage >= min_daily_coverage
   - trades_per_day >= min_trades_per_day
2) maximize utilization + composite score among feasible sets
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.intraday_features import compute_intraday_features, merge_slow_bias_to_intraday
from bot.strategy.intraday_momentum import IntradayMomentumStrategy
from scripts.backtest_intraday_hybrid import (
    build_slow_bias_4h,
    load_config,
    load_df,
    run_backtest,
)


RESULTS_DIR = Path("research_results")


def build_merged_feature_df(
    interval: str,
    btc_intraday: str,
    btc_4h: str,
    oil_daily: str,
    dxy_daily: str,
    btc_funding: str,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    btc_i = load_df(btc_intraday)
    btc4 = load_df(btc_4h)
    oil = load_df(oil_daily) if Path(oil_daily).exists() else None
    dxy = load_df(dxy_daily) if Path(dxy_daily).exists() else None
    fr = load_df(btc_funding) if Path(btc_funding).exists() else None

    fast = compute_intraday_features(btc_i, interval=interval)
    slow = build_slow_bias_4h(btc4, oil, dxy, fr)
    feat = merge_slow_bias_to_intraday(fast, slow).dropna()

    start_ts = pd.Timestamp(start, tz="UTC")
    feat = feat[feat.index >= start_ts]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]
    return feat


def build_weekly_windows(index: pd.DatetimeIndex, max_windows: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    days = pd.Index(index.normalize().unique()).sort_values()
    if len(days) < 7:
        return []

    windows = []
    i = 0
    while i + 6 < len(days):
        s = pd.Timestamp(days[i])
        e = pd.Timestamp(days[i + 6]) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        if s.tz is None:
            s = s.tz_localize("UTC")
        else:
            s = s.tz_convert("UTC")
        if e.tz is None:
            e = e.tz_localize("UTC")
        else:
            e = e.tz_convert("UTC")
        windows.append((s, e))
        i += 7

    if max_windows > 0 and len(windows) > max_windows:
        windows = windows[-max_windows:]
    return windows


def objective_tuple(
    feasible_all: bool,
    feasible_ratio: float,
    avg_tpd: float,
    avg_cov: float,
    avg_util: float,
    avg_comp: float,
) -> tuple:
    # Frequency-first ordering, then deployment quality.
    return (
        int(feasible_all),
        feasible_ratio,
        min(avg_tpd, 3.0),  # cap to avoid over-prioritizing pathological churn
        avg_cov,
        avg_util,
        avg_comp,
    )


def evaluate_param_set(
    feat: pd.DataFrame,
    cfg: dict,
    interval: str,
    params: dict,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    min_daily_coverage: float,
    min_trades_per_day: float,
    fee_bps: int,
    capital: float,
) -> dict:
    rows = []
    for start, end in windows:
        wdf = feat[(feat.index >= start) & (feat.index <= end)]
        if len(wdf) < 20:
            continue
        strategy = IntradayMomentumStrategy(params)
        s = run_backtest(
            feat=wdf,
            strategy=strategy,
            config=cfg,
            interval=interval,
            fee_bps=fee_bps,
            initial_capital=capital,
            max_hold_bars=int(params["max_hold_bars"]),
        )
        feasible = (s["daily_coverage"] >= min_daily_coverage) and (s["trades_per_day"] >= min_trades_per_day)
        rows.append(
            {
                "feasible": feasible,
                "daily_coverage": s["daily_coverage"],
                "trades_per_day": s["trades_per_day"],
                "avg_utilization": s["avg_utilization"],
                "composite_score": s["composite_score"],
                "sharpe": s["sharpe"],
                "max_drawdown_pct": s["max_drawdown_pct"],
            }
        )

    if not rows:
        return {
            "params": params,
            "n_windows": 0,
            "feasible_all": False,
            "feasible_ratio": 0.0,
            "avg_daily_coverage": 0.0,
            "avg_trades_per_day": 0.0,
            "avg_utilization": 0.0,
            "avg_composite": -math.inf,
            "avg_sharpe": -math.inf,
            "avg_max_dd_pct": math.inf,
            "objective": objective_tuple(False, 0.0, 0.0, 0.0, 0.0, -math.inf),
        }

    df = pd.DataFrame(rows)
    feasible_all = bool(df["feasible"].all())
    feasible_ratio = float(df["feasible"].mean())
    avg_cov = float(df["daily_coverage"].mean())
    avg_tpd = float(df["trades_per_day"].mean())
    avg_util = float(df["avg_utilization"].mean())
    avg_comp = float(df["composite_score"].mean())
    avg_sharpe = float(df["sharpe"].mean())
    avg_dd = float(df["max_drawdown_pct"].mean())

    return {
        "params": params,
        "n_windows": int(len(df)),
        "feasible_all": feasible_all,
        "feasible_ratio": feasible_ratio,
        "avg_daily_coverage": avg_cov,
        "avg_trades_per_day": avg_tpd,
        "avg_utilization": avg_util,
        "avg_composite": avg_comp,
        "avg_sharpe": avg_sharpe,
        "avg_max_dd_pct": avg_dd,
        "objective": objective_tuple(feasible_all, feasible_ratio, avg_tpd, avg_cov, avg_util, avg_comp),
    }


def build_param_grid(interval: str) -> list[dict]:
    common_size = [0.08, 0.12]
    common_bias = [0.05, 0.1]
    common_hold = [1, 2]

    momentum_grid = [
        {
            "interval": interval,
            "trigger_mode": "momentum_breakout",
            "return_threshold": r,
            "volume_ratio_threshold": v,
            "zscore_threshold": 1.5,
            "base_size": b,
            "bias_weight": bw,
            "max_hold_bars": h,
        }
        for r, v, b, bw, h in product(
            [0.00003, 0.0001, 0.0002],
            [0.75, 0.9],
            common_size,
            common_bias,
            common_hold,
        )
    ]
    mean_rev_grid = [
        {
            "interval": interval,
            "trigger_mode": "mean_reversion",
            "return_threshold": 0.0015,
            "volume_ratio_threshold": 1.2,
            "zscore_threshold": z,
            "base_size": b,
            "bias_weight": bw,
            "max_hold_bars": h,
        }
        for z, b, bw, h in product(
            [0.3, 0.5, 0.7],
            common_size,
            common_bias,
            common_hold,
        )
    ]

    # Runtime-bounded grid for rapid iteration.
    return momentum_grid + mean_rev_grid


def tune_interval(
    interval: str,
    feat: pd.DataFrame,
    cfg: dict,
    min_daily_coverage: float,
    min_trades_per_day: float,
    fee_bps: int,
    capital: float,
    coarse_windows: int,
    validate_windows: int,
    top_k: int,
) -> dict:
    all_windows = build_weekly_windows(feat.index, max_windows=0)
    if not all_windows:
        return {"interval": interval, "error": "No weekly windows available"}

    coarse = all_windows[-coarse_windows:] if coarse_windows > 0 else all_windows
    validate = all_windows[-validate_windows:] if validate_windows > 0 else all_windows

    grid = build_param_grid(interval)
    coarse_results = []
    for p in grid:
        r = evaluate_param_set(
            feat=feat,
            cfg=cfg,
            interval=interval,
            params=p,
            windows=coarse,
            min_daily_coverage=min_daily_coverage,
            min_trades_per_day=min_trades_per_day,
            fee_bps=fee_bps,
            capital=capital,
        )
        coarse_results.append(r)

    coarse_results.sort(key=lambda x: x["objective"], reverse=True)
    finalists = coarse_results[:top_k]

    validated = []
    for c in finalists:
        p = c["params"]
        vr = evaluate_param_set(
            feat=feat,
            cfg=cfg,
            interval=interval,
            params=p,
            windows=validate,
            min_daily_coverage=min_daily_coverage,
            min_trades_per_day=min_trades_per_day,
            fee_bps=fee_bps,
            capital=capital,
        )
        validated.append(vr)

    validated.sort(key=lambda x: x["objective"], reverse=True)
    best = validated[0] if validated else None
    feasible = [r for r in validated if r["feasible_all"]]
    best_feasible = feasible[0] if feasible else None

    return {
        "interval": interval,
        "n_candidates": len(grid),
        "coarse_windows": len(coarse),
        "validate_windows": len(validate),
        "best_overall": best,
        "best_feasible": best_feasible,
        "top_validated": validated[:5],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Tune hybrid intraday strategy for frequency constraints")
    p.add_argument("--btc-4h", default="research_data/BTCUSDT_4h.parquet")
    p.add_argument("--oil-daily", default="research_data/oil_daily.parquet")
    p.add_argument("--dxy-daily", default="research_data/dxy_daily.parquet")
    p.add_argument("--btc-funding", default="research_data/BTCUSDT_funding.parquet")
    p.add_argument("--btc-5m", default="research_data/BTCUSDT_5m.parquet")
    p.add_argument("--btc-15m", default="research_data/BTCUSDT_15m.parquet")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--fee-bps", type=int, default=10)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--min-daily-coverage", type=float, default=1.0)
    p.add_argument("--min-trades-per-day", type=float, default=1.0)
    p.add_argument("--coarse-windows", type=int, default=6)
    p.add_argument("--validate-windows", type=int, default=12)
    p.add_argument("--top-k", type=int, default=6)
    return p.parse_args()


def main():
    # Keep tuner output readable.
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("bot.execution.risk").setLevel(logging.ERROR)

    args = parse_args()
    cfg = load_config()

    feat_5m = build_merged_feature_df(
        interval="5m",
        btc_intraday=args.btc_5m,
        btc_4h=args.btc_4h,
        oil_daily=args.oil_daily,
        dxy_daily=args.dxy_daily,
        btc_funding=args.btc_funding,
        start=args.start,
        end=args.end,
    )
    feat_15m = build_merged_feature_df(
        interval="15m",
        btc_intraday=args.btc_15m,
        btc_4h=args.btc_4h,
        oil_daily=args.oil_daily,
        dxy_daily=args.dxy_daily,
        btc_funding=args.btc_funding,
        start=args.start,
        end=args.end,
    )

    res_5m = tune_interval(
        interval="5m",
        feat=feat_5m,
        cfg=cfg,
        min_daily_coverage=args.min_daily_coverage,
        min_trades_per_day=args.min_trades_per_day,
        fee_bps=args.fee_bps,
        capital=args.capital,
        coarse_windows=args.coarse_windows,
        validate_windows=args.validate_windows,
        top_k=args.top_k,
    )
    res_15m = tune_interval(
        interval="15m",
        feat=feat_15m,
        cfg=cfg,
        min_daily_coverage=args.min_daily_coverage,
        min_trades_per_day=args.min_trades_per_day,
        fee_bps=args.fee_bps,
        capital=args.capital,
        coarse_windows=args.coarse_windows,
        validate_windows=args.validate_windows,
        top_k=args.top_k,
    )

    payload = {
        "constraints": {
            "min_daily_coverage": args.min_daily_coverage,
            "min_trades_per_day": args.min_trades_per_day,
        },
        "period": {"start": args.start, "end": args.end},
        "results": {"5m": res_5m, "15m": res_15m},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "intraday_hybrid_tuning.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print("INTRADAY HYBRID TUNING")
    print("=" * 80)
    for k in ["5m", "15m"]:
        r = payload["results"][k]
        print(f"\nInterval: {k}")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        bo = r["best_overall"]
        bf = r["best_feasible"]
        print(f"  Candidates: {r['n_candidates']} | Coarse windows: {r['coarse_windows']} | Validate windows: {r['validate_windows']}")
        print(
            "  Best overall -> "
            f"feasible_all={bo['feasible_all']}, "
            f"feasible_ratio={bo['feasible_ratio']:.2f}, "
            f"tpd={bo['avg_trades_per_day']:.2f}, "
            f"coverage={100*bo['avg_daily_coverage']:.1f}%, "
            f"util={100*bo['avg_utilization']:.1f}%, "
            f"comp={bo['avg_composite']:.3f}, "
            f"sharpe={bo['avg_sharpe']:.3f}"
        )
        print(f"  Best overall params: {bo['params']}")
        if bf is None:
            print("  Best feasible: none")
        else:
            print(
                "  Best feasible -> "
                f"tpd={bf['avg_trades_per_day']:.2f}, "
                f"coverage={100*bf['avg_daily_coverage']:.1f}%, "
                f"util={100*bf['avg_utilization']:.1f}%, "
                f"comp={bf['avg_composite']:.3f}, "
                f"sharpe={bf['avg_sharpe']:.3f}"
            )
            print(f"  Best feasible params: {bf['params']}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

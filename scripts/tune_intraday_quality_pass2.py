#!/usr/bin/env python3
"""
Second-pass tuning: quality under frequency constraints.

Implements requested fixes:
1) stronger z-score threshold
2) trend confirmation
3) volume confirmation
4) cooldown bars
5) minimum expected edge
6) max trades/day cap
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from itertools import product
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.strategy.intraday_momentum import IntradayMomentumStrategy
from scripts.backtest_intraday_hybrid import load_config, run_backtest
from scripts.tune_intraday_hybrid import build_merged_feature_df, build_weekly_windows


RESULTS_DIR = Path("research_results")


def summarize_runs(rows: list[dict], params: dict) -> dict:
    if not rows:
        return {
            "params": params,
            "n_windows": 0,
            "feasible_all": False,
            "feasible_ratio": 0.0,
            "avg_trades_per_day": 0.0,
            "avg_daily_coverage": 0.0,
            "avg_utilization": 0.0,
            "avg_sharpe": -math.inf,
            "avg_sortino": -math.inf,
            "avg_calmar": -math.inf,
            "avg_composite": -math.inf,
            "avg_max_dd_pct": math.inf,
            "objective": (-1e9,),
        }
    df = pd.DataFrame(rows)
    feasible_all = bool(df["feasible"].all())
    feasible_ratio = float(df["feasible"].mean())
    def _clean(v: float, fallback: float) -> float:
        return float(v) if math.isfinite(float(v)) else fallback

    avg_tpd = _clean(df["trades_per_day"].mean(), 0.0)
    avg_cov = _clean(df["daily_coverage"].mean(), 0.0)
    avg_util = _clean(df["avg_utilization"].mean(), 0.0)
    avg_sharpe = _clean(df["sharpe"].mean(), -999.0)
    avg_sortino = _clean(df["sortino"].mean(), -999.0)
    avg_calmar = _clean(df["calmar"].mean(), -999.0)
    avg_comp = _clean(df["composite_score"].mean(), -999.0)
    avg_dd = _clean(df["max_drawdown_pct"].mean(), 999.0)
    objective = (
        int(feasible_all),
        feasible_ratio,
        avg_sortino,
        avg_sharpe,
        avg_calmar,
        avg_comp,
        avg_util,
        -avg_dd,
    )
    return {
        "params": params,
        "n_windows": int(len(df)),
        "feasible_all": feasible_all,
        "feasible_ratio": feasible_ratio,
        "avg_trades_per_day": avg_tpd,
        "avg_daily_coverage": avg_cov,
        "avg_utilization": avg_util,
        "avg_sharpe": avg_sharpe,
        "avg_sortino": avg_sortino,
        "avg_calmar": avg_calmar,
        "avg_composite": avg_comp,
        "avg_max_dd_pct": avg_dd,
        "objective": objective,
    }


def eval_params(
    feat: pd.DataFrame,
    cfg: dict,
    interval: str,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    params: dict,
    min_daily_coverage: float,
    min_trades_per_day: float,
    fee_bps: int,
    capital: float,
) -> dict:
    rows = []
    strat_cfg = {
        "interval": interval,
        "trigger_mode": "mean_reversion",
        "zscore_threshold": float(params["zscore_threshold"]),
        "volume_ratio_threshold": float(params["volume_ratio_threshold"]),
        "base_size": float(params["base_size"]),
        "bias_weight": float(params["bias_weight"]),
        "return_threshold": 0.0015,
        "require_trend_confirmation": bool(params["require_trend_confirmation"]),
        "require_volume_confirmation": bool(params["require_volume_confirmation"]),
    }
    for s, e in windows:
        wdf = feat[(feat.index >= s) & (feat.index <= e)]
        if len(wdf) < 30:
            continue
        strategy = IntradayMomentumStrategy(strat_cfg)
        stats = run_backtest(
            feat=wdf,
            strategy=strategy,
            config=cfg,
            interval=interval,
            fee_bps=fee_bps,
            initial_capital=capital,
            max_hold_bars=int(params["max_hold_bars"]),
            cooldown_bars=int(params["cooldown_bars"]),
            max_trades_per_day=int(params["max_trades_per_day"]),
            min_expected_edge=float(params["min_expected_edge"]),
        )
        feasible = (stats["daily_coverage"] >= min_daily_coverage) and (stats["trades_per_day"] >= min_trades_per_day)
        rows.append({"feasible": feasible, **stats})
    return summarize_runs(rows, params)


def best_result(results: list[dict]) -> dict:
    if not results:
        return {}
    results = sorted(results, key=lambda x: x["objective"], reverse=True)
    feasible = [r for r in results if r["feasible_all"]]
    return feasible[0] if feasible else results[0]


def run_interval_pass(
    interval: str,
    feat: pd.DataFrame,
    cfg: dict,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    min_daily_coverage: float,
    min_trades_per_day: float,
    fee_bps: int,
    capital: float,
) -> dict:
    base = {
        "zscore_threshold": 1.2,
        "require_trend_confirmation": False,
        "require_volume_confirmation": False,
        "volume_ratio_threshold": 1.2,
        "max_hold_bars": 2,
        "cooldown_bars": 0,
        "min_expected_edge": 0.0,
        "max_trades_per_day": 1000,
        "base_size": 0.12,
        "bias_weight": 0.1,
    }

    # Individual fix sweeps
    z_sweep = []
    for z in [1.0, 1.2, 1.5]:
        p = {**base, "zscore_threshold": z}
        z_sweep.append(eval_params(feat, cfg, interval, windows, p, min_daily_coverage, min_trades_per_day, fee_bps, capital))

    trend_sweep = []
    for flag in [False, True]:
        p = {**base, "zscore_threshold": 1.2, "require_trend_confirmation": flag}
        trend_sweep.append(eval_params(feat, cfg, interval, windows, p, min_daily_coverage, min_trades_per_day, fee_bps, capital))

    vol_sweep = []
    for vt in [1.1, 1.2, 1.3]:
        p = {**base, "zscore_threshold": 1.2, "require_volume_confirmation": True, "volume_ratio_threshold": vt}
        vol_sweep.append(eval_params(feat, cfg, interval, windows, p, min_daily_coverage, min_trades_per_day, fee_bps, capital))

    cooldown_sweep = []
    for c in [3, 4, 6]:
        p = {**base, "zscore_threshold": 1.2, "cooldown_bars": c}
        cooldown_sweep.append(eval_params(feat, cfg, interval, windows, p, min_daily_coverage, min_trades_per_day, fee_bps, capital))

    edge_sweep = []
    for edge in [0.002, 0.003, 0.004]:
        p = {**base, "zscore_threshold": 1.2, "min_expected_edge": edge}
        edge_sweep.append(eval_params(feat, cfg, interval, windows, p, min_daily_coverage, min_trades_per_day, fee_bps, capital))

    cap_sweep = []
    for cap in [10, 15, 20]:
        p = {**base, "zscore_threshold": 1.2, "max_trades_per_day": cap}
        cap_sweep.append(eval_params(feat, cfg, interval, windows, p, min_daily_coverage, min_trades_per_day, fee_bps, capital))

    best_z = best_result(z_sweep)
    best_trend = best_result(trend_sweep)
    best_vol = best_result(vol_sweep)
    best_cd = best_result(cooldown_sweep)
    best_edge = best_result(edge_sweep)
    best_cap = best_result(cap_sweep)

    # Composite trials: combine top settings from individual fixes
    z_vals = sorted({best_z["params"]["zscore_threshold"], 1.2})
    vol_vals = sorted({best_vol["params"]["volume_ratio_threshold"], 1.2})
    cd_vals = sorted({best_cd["params"]["cooldown_bars"], 3})
    edge_vals = sorted({best_edge["params"]["min_expected_edge"], 0.003})
    cap_vals = sorted({best_cap["params"]["max_trades_per_day"], 15})

    composite = []
    for z, vt, cd, ed, cap in product(z_vals, vol_vals, cd_vals, edge_vals, cap_vals):
        p = {
            **base,
            "zscore_threshold": z,
            "require_trend_confirmation": True,
            "require_volume_confirmation": True,
            "volume_ratio_threshold": vt,
            "cooldown_bars": cd,
            "min_expected_edge": ed,
            "max_trades_per_day": cap,
        }
        composite.append(eval_params(feat, cfg, interval, windows, p, min_daily_coverage, min_trades_per_day, fee_bps, capital))
    composite = sorted(composite, key=lambda x: x["objective"], reverse=True)
    best_composite = best_result(composite)

    # Overfit diagnostics: tune vs holdout windows + local sensitivity
    split = max(1, int(0.6 * len(windows)))
    tune_windows = windows[:split]
    hold_windows = windows[split:] if split < len(windows) else windows[-1:]
    tune_perf = eval_params(feat, cfg, interval, tune_windows, best_composite["params"], min_daily_coverage, min_trades_per_day, fee_bps, capital)
    hold_perf = eval_params(feat, cfg, interval, hold_windows, best_composite["params"], min_daily_coverage, min_trades_per_day, fee_bps, capital)

    local = []
    bp = best_composite["params"]
    for dz, dcd, ded in [(-0.2, -1, -0.001), (-0.2, 0, 0), (0, -1, 0), (0, 0, 0), (0, 1, 0), (0.2, 0, 0), (0.2, 1, 0.001)]:
        p = {
            **bp,
            "zscore_threshold": max(0.8, bp["zscore_threshold"] + dz),
            "cooldown_bars": max(0, bp["cooldown_bars"] + dcd),
            "min_expected_edge": max(0.0, bp["min_expected_edge"] + ded),
        }
        local.append(eval_params(feat, cfg, interval, windows, p, min_daily_coverage, min_trades_per_day, fee_bps, capital))
    local_df = pd.DataFrame(local)

    return {
        "interval": interval,
        "individual": {
            "zscore_threshold": sorted(z_sweep, key=lambda x: x["objective"], reverse=True),
            "trend_confirmation": sorted(trend_sweep, key=lambda x: x["objective"], reverse=True),
            "volume_confirmation": sorted(vol_sweep, key=lambda x: x["objective"], reverse=True),
            "cooldown_bars": sorted(cooldown_sweep, key=lambda x: x["objective"], reverse=True),
            "min_expected_edge": sorted(edge_sweep, key=lambda x: x["objective"], reverse=True),
            "max_trades_per_day": sorted(cap_sweep, key=lambda x: x["objective"], reverse=True),
        },
        "composite_top": composite[:5],
        "best_composite": best_composite,
        "overfit": {
            "tune_vs_holdout": {
                "tune": tune_perf,
                "holdout": hold_perf,
            },
            "local_sensitivity": {
                "mean_composite": float(local_df["avg_composite"].mean()),
                "std_composite": float(local_df["avg_composite"].std(ddof=0)),
                "mean_sharpe": float(local_df["avg_sharpe"].mean()),
                "std_sharpe": float(local_df["avg_sharpe"].std(ddof=0)),
            },
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="Second-pass intraday quality tuning")
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
    p.add_argument("--max-windows", type=int, default=8)
    return p.parse_args()


def main():
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("bot.execution.risk").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="quantstats")

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

    windows_5m = build_weekly_windows(feat_5m.index, max_windows=args.max_windows)
    windows_15m = build_weekly_windows(feat_15m.index, max_windows=args.max_windows)

    res_5m = run_interval_pass(
        "5m",
        feat_5m,
        cfg,
        windows_5m,
        args.min_daily_coverage,
        args.min_trades_per_day,
        args.fee_bps,
        args.capital,
    )
    res_15m = run_interval_pass(
        "15m",
        feat_15m,
        cfg,
        windows_15m,
        args.min_daily_coverage,
        args.min_trades_per_day,
        args.fee_bps,
        args.capital,
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
    out = RESULTS_DIR / "intraday_quality_pass2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print("INTRADAY QUALITY PASS-2")
    print("=" * 80)
    for interval in ["5m", "15m"]:
        best = payload["results"][interval]["best_composite"]
        print(f"{interval}: feasible_all={best['feasible_all']} ratio={best['feasible_ratio']:.2f} "
              f"tpd={best['avg_trades_per_day']:.2f} cov={100*best['avg_daily_coverage']:.1f}% "
              f"sharpe={best['avg_sharpe']:.3f} sortino={best['avg_sortino']:.3f} "
              f"calmar={best['avg_calmar']:.3f} comp={best['avg_composite']:.3f}")
        print(f"  params={best['params']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

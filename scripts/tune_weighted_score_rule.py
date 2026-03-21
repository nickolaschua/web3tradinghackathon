#!/usr/bin/env python3
"""
Tune weighted score rule on shared engine.

Rule:
    if momentum:
        score = 1.0
        if funding_zscore < funding_z_threshold:
            score += funding_bonus
        if macro_condition:
            score += macro_bonus
        if vol_spike > vol_threshold:
            score += vol_bonus
        if score >= score_threshold:
            BUY
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
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("bot.execution.risk").setLevel(logging.ERROR)

DATA_DIR = Path("research_data")
RESULTS_DIR = Path("research_results")
CONFIG_PATH = Path("bot/config/config.yaml")
PERIODS_4H = 2190


def load_framework_config() -> dict:
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


def build_feature_df(start: str) -> pd.DataFrame:
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
        close_4h = close.resample("4h").ffill()
        aligned = close_4h.reindex(feat.index, method="ffill")
        feat[f"{name}_return_1d"] = aligned.pct_change(6).shift(1)

    if btc_funding is not None:
        fr = btc_funding["funding_rate"].copy()
        fr_4h = fr.resample("4h").ffill()
        aligned = fr_4h.reindex(feat.index, method="ffill")
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


def safe_metric(value: float) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (float, np.floating)) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return float(value)


def compute_stats(returns: pd.Series, trades: list[dict]) -> dict:
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


def run_backtest(
    test_df: pd.DataFrame,
    buy_mask: pd.Series,
    portfolio_weight: float,
    config: dict,
    fee_bps: int,
    initial_capital: float = 10_000.0,
) -> dict:
    fee_rate = fee_bps / 10_000
    rm = RiskManager(config=config)
    rm.initialize_hwm(initial_capital)

    free_balance = initial_capital
    position_units = 0.0
    entry_price = None
    pair = "BTC/USD"
    prev_portfolio = initial_capital
    returns = []
    trades = []
    activity_days = set()

    for idx, row in test_df.iterrows():
        close = float(row["close"])
        atr = float(row["atr_proxy"]) if not pd.isna(row["atr_proxy"]) else np.nan
        just_exited = False

        if position_units > 0:
            stop_result = rm.check_stops(pair, close, atr)
            if stop_result.should_exit:
                exit_price = close * (1 - fee_rate)
                proceeds = position_units * close * (1 - fee_rate)
                pnl_pct = (exit_price - entry_price) / entry_price
                trades.append({"pnl_pct": pnl_pct, "exit_bar": idx})
                activity_days.add(pd.Timestamp(idx).normalize())
                free_balance += proceeds
                position_units = 0.0
                entry_price = None
                rm.record_exit(pair)
                just_exited = True

        total_portfolio = free_balance + position_units * close
        rm.check_circuit_breaker(total_portfolio)

        if position_units == 0.0 and (not just_exited) and bool(buy_mask.loc[idx]):
            sizing = rm.size_new_position(
                pair=pair,
                current_price=close,
                current_atr=atr,
                free_balance_usd=free_balance,
                open_positions={},
                regime_multiplier=1.0,
                confidence=0.70,
                portfolio_weight=portfolio_weight,
            )
            if sizing.decision == RiskDecision.APPROVED and sizing.approved_usd_value >= 10.0:
                target_usd = sizing.approved_usd_value
                position_units = target_usd / close
                free_balance -= (target_usd + target_usd * fee_rate)
                entry_price = close * (1 + fee_rate)
                rm.record_entry(pair, entry_price, sizing.trailing_stop_price)
                activity_days.add(pd.Timestamp(idx).normalize())

        end_portfolio = free_balance + position_units * close
        returns.append((end_portfolio / prev_portfolio - 1.0) if prev_portfolio > 0 else 0.0)
        prev_portfolio = end_portfolio

    stats = compute_stats(pd.Series(returns, index=test_df.index), trades)
    calendar_days = test_df.index.normalize().nunique()
    stats["activity_days"] = int(len(activity_days))
    stats["calendar_days"] = int(calendar_days)
    stats["daily_coverage"] = float(len(activity_days) / calendar_days) if calendar_days > 0 else 0.0
    stats["trades_per_day"] = float(stats["n_trades"] / calendar_days) if calendar_days > 0 else 0.0
    return stats


def build_weekly_windows(
    df: pd.DataFrame,
    window_days: int,
    step_days: int,
    min_train_bars: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build rolling train/test splits with fixed test window length in calendar days.
    Train = all bars before test start.
    """
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    idx = df.index
    if len(idx) == 0:
        return windows

    start_ts = idx.min().normalize()
    end_ts = idx.max().normalize()
    cursor = start_ts + pd.Timedelta(days=window_days)

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


def build_buy_mask(
    df: pd.DataFrame,
    funding_z_threshold: float,
    macro_threshold: float,
    vol_threshold: float,
    funding_bonus: float,
    macro_bonus: float,
    vol_bonus: float,
    score_threshold: float,
) -> pd.Series:
    momentum = (df["EMA_20"] > df["EMA_50"]) & (df["RSI_14"] < 50) & (df["MACDh_12_26_9"] > 0)
    macro_condition = (df["oil_return_1d"] > macro_threshold) & (df["dxy_return_1d"] < -macro_threshold)
    funding_condition = df["btc_funding_zscore"] < funding_z_threshold
    vol_condition = df["volume_ratio"] > vol_threshold

    score = np.ones(len(df), dtype=float)
    score += funding_bonus * funding_condition.astype(float)
    score += macro_bonus * macro_condition.astype(float)
    score += vol_bonus * vol_condition.astype(float)
    return pd.Series(momentum & (score >= score_threshold), index=df.index)


def evaluate_param_set(
    df: pd.DataFrame,
    config: dict,
    fee_bps: int,
    n_splits: int,
    params: dict,
    evaluation_mode: str = "cv",
    window_days: int = 7,
    step_days: int = 7,
    min_train_bars: int = 500,
    min_daily_coverage: float = 1.0,
    min_trades_per_day: float = 1.0,
) -> dict:
    if evaluation_mode == "weekly":
        splits = build_weekly_windows(
            df=df,
            window_days=window_days,
            step_days=step_days,
            min_train_bars=min_train_bars,
        )
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=24)
        splits = list(tscv.split(df))

    fold_stats = []
    total_trades = 0

    buy_mask_full = build_buy_mask(df=df, **params)

    for train_idx, test_idx in splits:
        train_slice = df.iloc[train_idx]
        test_slice = df.iloc[test_idx]
        test_mask = buy_mask_full.iloc[test_idx]

        pa = PortfolioAllocator(config=config)
        price_history = {
            "BTC/USD": pd.DataFrame({"close": train_slice["btc_close"].dropna()}),
            "ETH/USD": pd.DataFrame({"close": train_slice["eth_close"].dropna()}),
            "SOL/USD": pd.DataFrame({"close": train_slice["sol_close"].dropna()}),
        }
        pa.compute_weights(price_history)
        btc_w = pa.get_pair_weight("BTC/USD", n_active_pairs=3)

        stats = run_backtest(
            test_df=test_slice,
            buy_mask=test_mask,
            portfolio_weight=btc_w,
            config=config,
            fee_bps=fee_bps,
        )
        fold_stats.append(stats)
        total_trades += stats["n_trades"]

    if not fold_stats:
        return {
            "params": params,
            "composite_score": -1e9,
            "sharpe": -1e9,
            "sortino": -1e9,
            "calmar": -1e9,
            "total_return_pct": -1e9,
            "max_drawdown_pct": 0.0,
            "n_trades_total": 0,
            "avg_daily_coverage": 0.0,
            "avg_trades_per_day": 0.0,
            "n_windows": 0,
            "feasible": False,
        }

    avg_daily_coverage = float(np.mean([s["daily_coverage"] for s in fold_stats]))
    avg_trades_per_day = float(np.mean([s["trades_per_day"] for s in fold_stats]))
    feasible = (avg_daily_coverage >= min_daily_coverage) and (avg_trades_per_day >= min_trades_per_day)

    result = {
        "params": params,
        "composite_score": float(np.mean([s["composite_score"] for s in fold_stats])),
        "sharpe": float(np.mean([s["sharpe"] for s in fold_stats])),
        "sortino": float(np.mean([s["sortino"] for s in fold_stats])),
        "calmar": float(np.mean([s["calmar"] for s in fold_stats])),
        "total_return_pct": float(np.mean([s["total_return_pct"] for s in fold_stats])),
        "max_drawdown_pct": float(np.mean([s["max_drawdown_pct"] for s in fold_stats])),
        "n_trades_total": int(total_trades),
        "avg_daily_coverage": avg_daily_coverage,
        "avg_trades_per_day": avg_trades_per_day,
        "n_windows": len(fold_stats),
        "feasible": feasible,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Tune weighted score rule")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--fee-bps", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--evaluation-mode", choices=["cv", "weekly"], default="cv")
    parser.add_argument("--window-days", type=int, default=7, help="Used in weekly mode")
    parser.add_argument("--step-days", type=int, default=7, help="Used in weekly mode")
    parser.add_argument("--min-train-bars", type=int, default=500, help="Used in weekly mode")
    parser.add_argument("--min-daily-coverage", type=float, default=1.0, help="Min fraction of days with trade activity")
    parser.add_argument("--min-trades-per-day", type=float, default=1.0, help="Min avg trades/day")
    args = parser.parse_args()

    config = load_framework_config()
    df = build_feature_df(start=args.start)

    baseline = {
        "funding_z_threshold": -2.0,
        "macro_threshold": 0.0,
        "vol_threshold": 1.2,
        "funding_bonus": 0.5,
        "macro_bonus": 0.5,
        "vol_bonus": 0.3,
        "score_threshold": 1.8,
    }

    score_threshold_tests = [1.2 + 0.1 * i for i in range(10)]
    funding_threshold_tests = [-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5, -2.75, -3.0, -3.5]
    vol_threshold_tests = [1.0 + 0.1 * i for i in range(10)]
    macro_threshold_tests = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]

    sweeps = [
        ("score_threshold", score_threshold_tests),
        ("funding_z_threshold", funding_threshold_tests),
        ("vol_threshold", vol_threshold_tests),
        ("macro_threshold", macro_threshold_tests),
    ]

    all_results = {}
    winners = {}

    for name, values in sweeps:
        res = []
        for v in values:
            params = dict(baseline)
            params[name] = float(v)
            res.append(
                evaluate_param_set(
                    df=df,
                    config=config,
                    fee_bps=args.fee_bps,
                    n_splits=args.n_splits,
                    params=params,
                    evaluation_mode=args.evaluation_mode,
                    window_days=args.window_days,
                    step_days=args.step_days,
                    min_train_bars=args.min_train_bars,
                    min_daily_coverage=args.min_daily_coverage,
                    min_trades_per_day=args.min_trades_per_day,
                )
            )
        res.sort(key=lambda x: x["composite_score"], reverse=True)
        all_results[name] = res
        feasible_res = [r for r in res if r["feasible"]]
        winners[name] = feasible_res[0] if feasible_res else res[0]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "weighted_score_tuning.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": baseline,
                "evaluation": {
                    "mode": args.evaluation_mode,
                    "window_days": args.window_days,
                    "step_days": args.step_days,
                    "min_train_bars": args.min_train_bars,
                    "min_daily_coverage": args.min_daily_coverage,
                    "min_trades_per_day": args.min_trades_per_day,
                },
                "winners": winners,
                "sweeps": all_results,
            },
            f,
            indent=2,
        )

    print("=" * 100)
    print("  WEIGHTED SCORE TUNING (SHARED ENGINE)")
    print("=" * 100)
    print("Base rule: momentum + weighted boosters (funding/macro/vol)")
    print(f"Evaluation mode: {args.evaluation_mode}")
    print(f"Saved full sweep: {out_path}\n")
    print(
        f"{'Sweep':<22} {'Best Value':>10} {'Composite':>10} {'Sharpe':>8} {'Sortino':>8} "
        f"{'Calmar':>8} {'Trades':>8} {'Ret%':>7} {'Cov%':>7} {'T/D':>6} {'OK':>4}"
    )
    print("-" * 100)
    for name, _ in sweeps:
        w = winners[name]
        print(
            f"{name:<22} {w['params'][name]:>10.3f} {w['composite_score']:>10.3f} "
            f"{w['sharpe']:>8.3f} {w['sortino']:>8.3f} {w['calmar']:>8.3f} "
            f"{w['n_trades_total']:>8} {w['total_return_pct']:>6.2f}% "
            f"{100*w['avg_daily_coverage']:>6.1f}% {w['avg_trades_per_day']:>6.2f} "
            f"{'yes' if w['feasible'] else 'no':>4}"
        )


if __name__ == "__main__":
    main()

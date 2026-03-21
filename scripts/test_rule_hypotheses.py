#!/usr/bin/env python3
"""
Run rule-based hypothesis tests on the shared backtest engine.

Tests:
1) macro + funding z-score extreme (buy on < -2, avoid on > 2)
2) macro + funding condition + EMA trend filter
3) volume spike allow-trade
4) full conjunction: macro + funding extreme + EMA trend + vol spike
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs
import yaml
from sklearn.model_selection import TimeSeriesSplit

# Add project root to import bot package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bot.data.features import compute_cross_asset_features, compute_features
from bot.execution.portfolio import PortfolioAllocator
from bot.execution.risk import RiskDecision, RiskManager


DATA_DIR = Path("research_data")
RESULTS_DIR = Path("research_results")
CONFIG_PATH = Path("bot/config/config.yaml")
PERIODS_4H = 2190


def load_framework_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all_data():
    btc = pd.read_parquet(DATA_DIR / "BTCUSDT_4h.parquet")
    btc.columns = btc.columns.str.lower()
    eth = pd.read_parquet(DATA_DIR / "ETHUSDT_4h.parquet") if (DATA_DIR / "ETHUSDT_4h.parquet").exists() else None
    sol = pd.read_parquet(DATA_DIR / "SOLUSDT_4h.parquet") if (DATA_DIR / "SOLUSDT_4h.parquet").exists() else None
    oil = pd.read_parquet(DATA_DIR / "oil_daily.parquet") if (DATA_DIR / "oil_daily.parquet").exists() else None
    dxy = pd.read_parquet(DATA_DIR / "dxy_daily.parquet") if (DATA_DIR / "dxy_daily.parquet").exists() else None
    btc_funding = pd.read_parquet(DATA_DIR / "BTCUSDT_funding.parquet") if (DATA_DIR / "BTCUSDT_funding.parquet").exists() else None
    return btc, eth, sol, oil, dxy, btc_funding


def add_macro_features(df: pd.DataFrame, oil_df: pd.DataFrame | None, dxy_df: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    for name, mdf in [("oil", oil_df), ("dxy", dxy_df)]:
        if mdf is None:
            continue
        close = mdf["close"].copy()
        close_4h = close.resample("4h").ffill()
        aligned = close_4h.reindex(out.index, method="ffill")
        out[f"{name}_return_1d"] = aligned.pct_change(6).shift(1)
    return out


def add_funding_features(df: pd.DataFrame, btc_funding: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if btc_funding is None:
        return out
    fr = btc_funding["funding_rate"].copy()
    fr_4h = fr.resample("4h").ffill()
    aligned = fr_4h.reindex(out.index, method="ffill")
    rm = aligned.rolling(90).mean()
    rs = aligned.rolling(90).std()
    out["btc_funding_zscore"] = ((aligned - rm) / rs.replace(0, np.nan)).shift(1)
    return out


def compute_stats(returns: pd.Series, trades: list[dict]) -> dict:
    returns = returns[returns.index.notna()]
    n_trades = len(trades)
    return {
        "total_return_pct": float((1 + returns).prod() - 1) * 100,
        "sharpe": float(qs.stats.sharpe(returns, periods=PERIODS_4H)),
        "sortino": float(qs.stats.sortino(returns, periods=PERIODS_4H)),
        "calmar": float(qs.stats.calmar(returns)),
        "composite_score": 0.4 * float(qs.stats.sortino(returns, periods=PERIODS_4H))
        + 0.3 * float(qs.stats.sharpe(returns, periods=PERIODS_4H))
        + 0.3 * float(qs.stats.calmar(returns)),
        "max_drawdown_pct": float(qs.stats.max_drawdown(returns)) * 100,
        "n_trades": n_trades,
        "n_bars": len(returns),
    }


def run_rule_backtest(
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

    pair = "BTC/USD"
    free_balance = initial_capital
    position_units = 0.0
    entry_price = None
    entry_bar = None

    returns = []
    trades = []
    prev_portfolio = initial_capital

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
                trades.append(
                    {
                        "entry_bar": entry_bar,
                        "exit_bar": idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "stop",
                    }
                )
                free_balance += proceeds
                position_units = 0.0
                entry_price = None
                entry_bar = None
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
                fee_cost = target_usd * fee_rate
                free_balance -= (target_usd + fee_cost)
                entry_price = close * (1 + fee_rate)
                entry_bar = idx
                rm.record_entry(pair, entry_price, sizing.trailing_stop_price)

        end_portfolio = free_balance + position_units * close
        bar_return = (end_portfolio / prev_portfolio - 1.0) if prev_portfolio > 0 else 0.0
        returns.append(bar_return)
        prev_portfolio = end_portfolio

    returns_series = pd.Series(returns, index=test_df.index)
    return {"returns": returns_series, "trades": trades, "stats": compute_stats(returns_series, trades)}


def main():
    parser = argparse.ArgumentParser(description="Run rule-based hypothesis tests")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--fee-bps", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()

    config = load_framework_config()
    btc, eth, sol, oil, dxy, btc_funding = load_all_data()

    feat = compute_features(btc.copy())
    others = {}
    if eth is not None:
        others["ETH/USD"] = eth
    if sol is not None:
        others["SOL/USD"] = sol
    feat = compute_cross_asset_features(feat, others)
    feat = add_macro_features(feat, oil, dxy)
    feat = add_funding_features(feat, btc_funding)

    feat["btc_close"] = feat["close"]
    if eth is not None:
        feat["eth_close"] = eth["close"].reindex(feat.index)
    if sol is not None:
        feat["sol_close"] = sol["close"].reindex(feat.index)

    start_ts = pd.Timestamp(args.start, tz="UTC")
    feat = feat[feat.index >= start_ts]

    needed = [
        "close",
        "atr_proxy",
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
    df = feat[needed].dropna().copy()

    macro_condition = (df["oil_return_1d"] > 0) & (df["dxy_return_1d"] < 0)
    funding_z_lt2 = df["btc_funding_zscore"] < -2
    funding_z_gt2 = df["btc_funding_zscore"] > 2
    funding_condition = funding_z_lt2
    ema_trend = df["EMA_20"] > df["EMA_50"]
    vol_spike = df["volume_ratio"] > 1.5

    rule_masks = {
        "Test 1 - Macro + Funding Extreme": macro_condition & funding_z_lt2 & (~funding_z_gt2),
        "Test 2 - Macro + Funding + EMA": macro_condition & funding_condition & ema_trend,
        "Test 3 - Vol Spike > 1.5": vol_spike,
        "Test 4 - Full Conjunction": macro_condition & funding_z_lt2 & ema_trend & vol_spike,
    }

    tscv = TimeSeriesSplit(n_splits=args.n_splits, gap=24)
    results = []
    for test_name, mask in rule_masks.items():
        fold_stats = []
        total_trades = 0
        for train_idx, test_idx in tscv.split(df):
            train_slice = df.iloc[train_idx]
            test_slice = df.iloc[test_idx]
            test_mask = mask.iloc[test_idx]

            pa = PortfolioAllocator(config=config)
            price_history = {
                "BTC/USD": pd.DataFrame({"close": train_slice["btc_close"].dropna()}),
                "ETH/USD": pd.DataFrame({"close": train_slice["eth_close"].dropna()}),
                "SOL/USD": pd.DataFrame({"close": train_slice["sol_close"].dropna()}),
            }
            pa.compute_weights(price_history)
            btc_w = pa.get_pair_weight("BTC/USD", n_active_pairs=3)

            out = run_rule_backtest(
                test_df=test_slice,
                buy_mask=test_mask,
                portfolio_weight=btc_w,
                config=config,
                fee_bps=args.fee_bps,
            )
            fold_stats.append(out["stats"])
            total_trades += out["stats"]["n_trades"]

        agg = {
            "strategy": test_name,
            "composite_score": float(np.mean([s["composite_score"] for s in fold_stats])),
            "sharpe": float(np.mean([s["sharpe"] for s in fold_stats])),
            "sortino": float(np.mean([s["sortino"] for s in fold_stats])),
            "calmar": float(np.mean([s["calmar"] for s in fold_stats])),
            "total_return_pct": float(np.mean([s["total_return_pct"] for s in fold_stats])),
            "max_drawdown_pct": float(np.mean([s["max_drawdown_pct"] for s in fold_stats])),
            "n_trades_total": int(total_trades),
        }
        results.append(agg)

    results.sort(key=lambda x: x["composite_score"], reverse=True)

    print("=" * 90)
    print("  RULE HYPOTHESIS TESTS (SHARED ENGINE)")
    print("=" * 90)
    print(f"{'Strategy':<36} {'Composite':>10} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'Trades':>8} {'Return%':>9}")
    print("-" * 90)
    for i, r in enumerate(results):
        mark = " 🏆" if i == 0 else ""
        print(
            f"{r['strategy']:<36} {r['composite_score']:>10.3f} {r['sharpe']:>8.3f} "
            f"{r['sortino']:>8.3f} {r['calmar']:>8.3f} {r['n_trades_total']:>8} "
            f"{r['total_return_pct']:>8.2f}%{mark}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "rule_hypothesis_tests.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

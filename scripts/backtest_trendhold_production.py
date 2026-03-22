#!/usr/bin/env python3
"""
TrendHold backtest using PRODUCTION RiskManager + XGBoost models.

Combines:
  - Conviction holds in ETH/SOL (XGBoost-gated entries)
  - BTC daily filler for 8/10 active day requirement
  - Production risk management: ATR trailing stops, tiered circuit breaker, Kelly sizing
  - Portfolio-level drawdown cap

Usage:
  python scripts/backtest_trendhold_production.py
  python scripts/backtest_trendhold_production.py --windows 2026-03-11,2026-03-01 --window-days 10
  python scripts/backtest_trendhold_production.py --hard-stop 0.12 --atr-mult 3.0
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import (
    compute_btc_context_features,
    compute_cross_asset_features,
    compute_features,
)
from bot.execution.risk import RiskManager, RiskDecision

DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models"

_BINANCE_TO_ROOSTOO = {
    "BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD", "SOLUSDT": "SOL/USD",
    "LINKUSDT": "LINK/USD", "DOGEUSDT": "DOGE/USD", "PAXGUSDT": "PAXG/USD",
}

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for path in sorted(DATA_DIR.glob("*_15m.parquet")):
        sym = path.stem.replace("_15m", "")
        pair = _BINANCE_TO_ROOSTOO.get(sym)
        if pair is None:
            continue
        df = pd.read_parquet(path)
        df.columns = df.columns.str.lower()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        data[pair] = df
    print(f"Loaded {len(data)} coins")
    return data


def compute_all_features(raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    btc_raw = raw_data.get("BTC/USD")
    eth_raw = raw_data.get("ETH/USD")
    sol_raw = raw_data.get("SOL/USD")

    features = {}
    for pair, df in raw_data.items():
        feat = compute_features(df)
        if pair == "BTC/USD" and eth_raw is not None and sol_raw is not None:
            feat = compute_cross_asset_features(feat, {"ETH/USD": eth_raw, "SOL/USD": sol_raw})
            feat = compute_btc_context_features(feat, eth_raw, sol_raw, window=2880)
            for sym, alt_df in [("eth", eth_raw), ("sol", sol_raw)]:
                alt_close = alt_df["close"].reindex(feat.index, method="ffill")
                alt_ret = np.log(alt_close / alt_close.shift(1))
                feat[f"{sym}_return_4h"] = alt_ret.rolling(16).sum().shift(1)
                feat[f"{sym}_return_1d"] = alt_ret.rolling(96).sum().shift(1)
        elif pair in ("SOL/USD", "ETH/USD") and btc_raw is not None:
            cross_pairs = {"BTC/USD": btc_raw}
            if pair == "SOL/USD" and eth_raw is not None:
                cross_pairs["ETH/USD"] = eth_raw
            if pair == "ETH/USD" and sol_raw is not None:
                cross_pairs["SOL/USD"] = sol_raw
            feat = compute_cross_asset_features(feat, cross_pairs)

            btc_close = btc_raw["close"].reindex(feat.index, method="ffill")
            btc_ret = np.log(btc_close / btc_close.shift(1))
            feat["btc_return_4h"] = btc_ret.rolling(16).sum().shift(1)
            feat["btc_return_1d"] = btc_ret.rolling(96).sum().shift(1)

            target_ret = np.log(df["close"] / df["close"].shift(1)).reindex(feat.index)
            btc_ret_aligned = btc_ret.reindex(feat.index)

            if pair == "ETH/USD":
                corr_prefix = "eth_btc"
                if sol_raw is not None:
                    sol_close = sol_raw["close"].reindex(feat.index, method="ffill")
                    sol_ret = np.log(sol_close / sol_close.shift(1))
                    feat["sol_return_4h"] = sol_ret.rolling(16).sum().shift(1)
                    feat["sol_return_1d"] = sol_ret.rolling(96).sum().shift(1)
            else:
                corr_prefix = "sol_btc"
                if eth_raw is not None:
                    eth_close = eth_raw["close"].reindex(feat.index, method="ffill")
                    eth_ret = np.log(eth_close / eth_close.shift(1))
                    feat["eth_return_4h"] = eth_ret.rolling(16).sum().shift(1)
                    feat["eth_return_1d"] = eth_ret.rolling(96).sum().shift(1)

            corr = target_ret.rolling(2880).corr(btc_ret_aligned)
            cov = target_ret.rolling(2880).cov(btc_ret_aligned)
            var_btc = btc_ret_aligned.rolling(2880).var()
            feat[f"{corr_prefix}_corr"] = corr.shift(1)
            feat[f"{corr_prefix}_beta"] = (cov / (var_btc + 1e-10)).shift(1)
        features[pair] = feat
    return features


# ── XGBoost Probability Maps ─────────────────────────────────────────────────

def build_proba_map(features_df: pd.DataFrame, model_path: str, oos_start: str) -> dict:
    path = Path(model_path)
    if not path.exists():
        print(f"  Model not found: {model_path}")
        return {}
    with open(path, "rb") as f:
        model = pickle.load(f)
    feat_cols = list(model.feature_names_in_)
    oos = features_df[features_df.index >= pd.Timestamp(oos_start, tz="UTC")].copy()
    missing = [c for c in feat_cols if c not in oos.columns]
    for c in missing:
        oos[c] = np.nan
    X = oos[feat_cols]
    probas = model.predict_proba(X)[:, 1]
    pmap = dict(zip(oos.index, probas))
    print(f"  Built proba map: {path.name} → {len(pmap)} bars, mean P={np.mean(probas):.3f}")
    return pmap


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_trendhold_backtest(
    features: dict[str, pd.DataFrame],
    proba_maps: dict[str, dict],
    common_index: pd.DatetimeIndex,
    args,
    label: str = "",
) -> dict:
    """Run TrendHold backtest with production RiskManager."""

    risk_config = {
        "hard_stop_pct": args.hard_stop,
        "atr_stop_multiplier": args.atr_mult,
        "max_positions": 6,
        "max_single_position_pct": 0.45,
        "risk_per_trade_pct": args.risk_per_trade,
        "expected_win_loss_ratio": 1.5,
        "circuit_breaker": {"halt_threshold": args.cb_halt},
    }
    risk_mgr = RiskManager(risk_config)

    initial_capital = 1_000_000.0
    cash = initial_capital
    risk_mgr.initialize_hwm(cash)
    fee_rate = 10 / 10_000

    # Position tracking: {pair: {"qty", "entry_price", "entry_bar", "source"}}
    positions: dict[str, dict] = {}
    closed_trades: list[dict] = []
    trade_dates: set = set()

    hold_pairs = {"ETH/USD": args.eth_pct, "SOL/USD": args.sol_pct}
    if args.btc_pct > 0:
        hold_pairs["BTC/USD"] = args.btc_pct
    filler_pair = args.filler_pair
    filler_pct = args.filler_pct

    n = len(common_index)
    equity = np.zeros(n)
    equity[0] = initial_capital

    # Pre-build close prices and ATR for fast access
    close_cache: dict[str, np.ndarray] = {}
    atr_cache: dict[str, np.ndarray] = {}
    for pair in features:
        aligned = features[pair].reindex(common_index)
        close_cache[pair] = aligned["close"].values
        atr_cache[pair] = aligned.get("atr_proxy", aligned["close"] * 0.02).values

    cb_halted_bars = 0

    for bar_idx in range(n):
        ts = common_index[bar_idx]

        # ── Mark to market ────────────────────────────────────────────
        pos_value = 0.0
        for pair, pos in positions.items():
            c = close_cache.get(pair, np.array([]))[bar_idx] if pair in close_cache else np.nan
            if not np.isnan(c):
                pos_value += pos["qty"] * c
        portfolio_value = cash + pos_value
        equity[bar_idx] = portfolio_value

        # ── Circuit breaker check ─────────────────────────────────────
        cb_active = risk_mgr.check_circuit_breaker(portfolio_value)
        cb_mult = risk_mgr.get_cb_size_multiplier(portfolio_value)
        if cb_active:
            cb_halted_bars += 1

        # ── Check stops for all positions ─────────────────────────────
        for pair in list(positions.keys()):
            c = close_cache.get(pair, np.array([]))[bar_idx] if pair in close_cache else np.nan
            if np.isnan(c):
                continue
            atr_val = atr_cache.get(pair, np.array([]))[bar_idx] if pair in atr_cache else c * 0.02
            if np.isnan(atr_val):
                atr_val = c * 0.02

            stop_result = risk_mgr.check_stops(pair, c, atr_val)
            if stop_result.should_exit:
                pos = positions[pair]
                proceeds = pos["qty"] * c * (1 - fee_rate)
                pnl_pct = (c * (1 - fee_rate) - pos["entry_price"]) / pos["entry_price"]
                closed_trades.append({
                    "pair": pair, "source": pos["source"],
                    "entry_bar": pos["entry_ts"], "exit_bar": ts,
                    "pnl_pct": pnl_pct, "exit_reason": stop_result.exit_type,
                })
                cash += proceeds
                risk_mgr.record_exit(pair)
                del positions[pair]
                trade_dates.add(ts.date())

        # ── XGB signal-based exits for hold positions ─────────────────
        for pair in list(positions.keys()):
            pos = positions[pair]
            if pos["source"] != "hold":
                continue
            pmap = proba_maps.get(pair, {})
            p = pmap.get(ts, 0.5)
            if p <= args.xgb_exit:
                c = close_cache[pair][bar_idx]
                if np.isnan(c):
                    continue
                proceeds = pos["qty"] * c * (1 - fee_rate)
                pnl_pct = (c * (1 - fee_rate) - pos["entry_price"]) / pos["entry_price"]
                closed_trades.append({
                    "pair": pair, "source": "hold_xgb_exit",
                    "entry_bar": pos["entry_ts"], "exit_bar": ts,
                    "pnl_pct": pnl_pct, "exit_reason": "xgb_bearish",
                })
                cash += proceeds
                risk_mgr.record_exit(pair)
                del positions[pair]
                trade_dates.add(ts.date())

        # ── Filler cycling ────────────────────────────────────────────
        if filler_pair in positions:
            pos = positions[filler_pair]
            if pos["source"] == "filler" and bar_idx - pos["entry_bar_idx"] >= args.filler_bars:
                c = close_cache.get(filler_pair, np.array([]))[bar_idx]
                if not np.isnan(c):
                    proceeds = pos["qty"] * c * (1 - fee_rate)
                    pnl_pct = (c * (1 - fee_rate) - pos["entry_price"]) / pos["entry_price"]
                    closed_trades.append({
                        "pair": filler_pair, "source": "filler",
                        "entry_bar": pos["entry_ts"], "exit_bar": ts,
                        "pnl_pct": pnl_pct, "exit_reason": "filler_cycle",
                    })
                    cash += proceeds
                    risk_mgr.record_exit(filler_pair)
                    del positions[filler_pair]
                    trade_dates.add(ts.date())

        # ── Hold entries (XGB-gated) ──────────────────────────────────
        if not cb_active:
            for pair, alloc_pct in hold_pairs.items():
                if pair in positions:
                    continue
                c = close_cache.get(pair, np.array([]))[bar_idx]
                if np.isnan(c) or c <= 0:
                    continue

                pmap = proba_maps.get(pair, {})
                p = pmap.get(ts, 0.5)
                if p < args.xgb_threshold:
                    continue

                atr_val = atr_cache.get(pair, np.array([]))[bar_idx]
                if np.isnan(atr_val):
                    atr_val = c * 0.02

                open_pos_usd = {
                    pp: pos["qty"] * (close_cache.get(pp, np.array([np.nan]))[bar_idx])
                    for pp, pos in positions.items()
                    if not np.isnan(close_cache.get(pp, np.array([np.nan]))[bar_idx])
                }

                sizing = risk_mgr.size_new_position(
                    pair=pair, current_price=c, current_atr=atr_val,
                    free_balance_usd=cash, open_positions=open_pos_usd,
                    regime_multiplier=1.0, confidence=p,
                    portfolio_weight=alloc_pct / 0.33,
                )

                if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity > 0:
                    entry_cost = sizing.approved_quantity * c * (1 + fee_rate)
                    if entry_cost <= cash:
                        cash -= entry_cost
                        positions[pair] = {
                            "qty": sizing.approved_quantity,
                            "entry_price": c * (1 + fee_rate),
                            "entry_ts": ts,
                            "entry_bar_idx": bar_idx,
                            "source": "hold",
                        }
                        risk_mgr.record_entry(pair, c * (1 + fee_rate), sizing.trailing_stop_price)
                        trade_dates.add(ts.date())

        # ── Filler entry ──────────────────────────────────────────────
        if filler_pair not in positions and not cb_active:
            c = close_cache.get(filler_pair, np.array([]))[bar_idx]
            if not np.isnan(c) and c > 0:
                atr_val = atr_cache.get(filler_pair, np.array([]))[bar_idx]
                if np.isnan(atr_val):
                    atr_val = c * 0.02

                open_pos_usd = {
                    pp: pos["qty"] * (close_cache.get(pp, np.array([np.nan]))[bar_idx])
                    for pp, pos in positions.items()
                    if not np.isnan(close_cache.get(pp, np.array([np.nan]))[bar_idx])
                }

                sizing = risk_mgr.size_new_position(
                    pair=filler_pair, current_price=c, current_atr=atr_val,
                    free_balance_usd=cash, open_positions=open_pos_usd,
                    regime_multiplier=1.0, confidence=0.55,
                    portfolio_weight=filler_pct / 0.33,
                )

                if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity > 0:
                    entry_cost = sizing.approved_quantity * c * (1 + fee_rate)
                    if entry_cost <= cash:
                        cash -= entry_cost
                        positions[filler_pair] = {
                            "qty": sizing.approved_quantity,
                            "entry_price": c * (1 + fee_rate),
                            "entry_ts": ts,
                            "entry_bar_idx": bar_idx,
                            "source": "filler",
                        }
                        risk_mgr.record_entry(filler_pair, c * (1 + fee_rate), sizing.trailing_stop_price)
                        trade_dates.add(ts.date())

    # ── Compute metrics ───────────────────────────────────────────────
    rets = np.diff(equity) / equity[:-1]
    total_return = (equity[-1] / equity[0]) - 1.0
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-10
    PERIODS = 35_040
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(PERIODS)

    down = rets[rets < 0]
    down_std = np.std(down, ddof=1) if len(down) > 1 else 1e-10
    sortino = (mean_ret / (down_std + 1e-10)) * np.sqrt(PERIODS)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    calmar = (mean_ret * PERIODS) / (abs(max_dd) + 1e-10) if max_dd < 0 else 0.0

    total_days = (common_index[-1] - common_index[0]).days + 1
    active_days = len(trade_dates)

    # Source breakdown
    src_counts = defaultdict(int)
    src_pnl = defaultdict(float)
    src_wins = defaultdict(int)
    src_exits = defaultdict(lambda: defaultdict(int))
    for t in closed_trades:
        s = t["source"]
        src_counts[s] += 1
        src_pnl[s] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            src_wins[s] += 1
        src_exits[s][t["exit_reason"]] += 1

    return {
        "label": label,
        "n_trades": len(closed_trades),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "active_days": active_days,
        "total_days": total_days,
        "cb_halted_bars": cb_halted_bars,
        "open_positions": len(positions),
        "source_breakdown": {
            s: {
                "count": src_counts[s],
                "win_pct": round(src_wins[s] / src_counts[s] * 100, 1) if src_counts[s] else 0,
                "avg_pnl": round(src_pnl[s] / src_counts[s] * 100, 3) if src_counts[s] else 0,
                "exits": dict(src_exits[s]),
            }
            for s in src_counts
        },
    }


def print_report(r: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  RESULTS  {r['label']}")
    print("=" * 60)
    print(f"  Trades: {r['n_trades']}  |  Open: {r['open_positions']}")
    print(f"  Active days: {r['active_days']}/{r['total_days']}  "
          f"{'PASS' if r['active_days'] >= 8 else 'FAIL'}")
    if r['cb_halted_bars'] > 0:
        print(f"  Circuit breaker halted: {r['cb_halted_bars']} bars")
    print()
    for src, info in r["source_breakdown"].items():
        exits_str = " ".join(f"{k}:{v}" for k, v in info["exits"].items())
        print(f"    {src:20s}: {info['count']:3d} trades, "
              f"win={info['win_pct']:5.1f}%, avg={info['avg_pnl']:+.3f}%  [{exits_str}]")
    print()
    print(f"  Sharpe:  {r['sharpe']:>8.3f}    Sortino: {r['sortino']:>8.3f}")
    print(f"  Calmar:  {r['calmar']:>8.3f}")
    print(f"  Return:  {r['total_return_pct']:>+7.2f}%   MaxDD: {r['max_drawdown_pct']:>7.2f}%")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrendHold + Production RiskManager backtest")

    # Allocation
    parser.add_argument("--eth-pct", type=float, default=0.35)
    parser.add_argument("--sol-pct", type=float, default=0.30)
    parser.add_argument("--btc-pct", type=float, default=0.0,
                        help="BTC main hold allocation (0 = BTC used as filler only)")
    parser.add_argument("--filler-pair", type=str, default="BTC/USD")
    parser.add_argument("--filler-pct", type=float, default=0.10)
    parser.add_argument("--filler-bars", type=int, default=96, help="Filler cycle bars (96 = 1 day)")

    # XGBoost
    parser.add_argument("--xgb-threshold", type=float, default=0.55,
                        help="XGB P(BUY) threshold for hold entries (lower = more entries)")
    parser.add_argument("--xgb-exit", type=float, default=0.10,
                        help="XGB P(BUY) exit threshold (model very bearish → exit)")

    # Risk Management (production RiskManager params)
    parser.add_argument("--hard-stop", type=float, default=0.12,
                        help="Hard stop loss %% (default 12%%)")
    parser.add_argument("--atr-mult", type=float, default=25.0,
                        help="ATR trailing stop multiplier (default 25.0)")
    parser.add_argument("--risk-per-trade", type=float, default=0.025,
                        help="Risk per trade as %% of portfolio (default 2.5%%)")
    parser.add_argument("--cb-halt", type=float, default=0.15,
                        help="Circuit breaker halt threshold (default 15%% drawdown)")

    # Backtest
    parser.add_argument("--windows", type=str, default=None)
    parser.add_argument("--window-days", type=int, default=10)
    parser.add_argument("--oos-start", type=str, default="2024-01-01")

    args = parser.parse_args()

    print("=" * 60)
    print("  TRENDHOLD + PRODUCTION RISK BACKTEST")
    print("=" * 60)

    raw_data = load_data()
    print("\nComputing features...")
    all_features = compute_all_features(raw_data)
    print(f"Computed features for {len(all_features)} coins")

    # Build XGB proba maps for ETH and SOL
    print("\nBuilding XGBoost probability maps...")
    proba_maps = {}
    model_configs = {
        "ETH/USD": "models/xgb_eth_15m.pkl",
        "SOL/USD": "models/xgb_sol_15m.pkl",
        "BTC/USD": "models/xgb_btc_15m.pkl",
    }
    for pair, mpath in model_configs.items():
        if pair in all_features:
            pm = build_proba_map(all_features[pair], mpath, args.oos_start)
            if pm:
                proba_maps[pair] = pm

    # OOS features
    oos_ts = pd.Timestamp(args.oos_start, tz="UTC")
    oos_features = {p: f[f.index >= oos_ts] for p, f in all_features.items() if len(f[f.index >= oos_ts]) > 100}

    print(f"\nConfig:")
    print(f"  Hold: ETH={args.eth_pct*100:.0f}% SOL={args.sol_pct*100:.0f}% BTC={args.btc_pct*100:.0f}%")
    print(f"  Filler: {args.filler_pair} {args.filler_pct*100:.0f}% (cycle every {args.filler_bars} bars)")
    print(f"  XGB: entry≥{args.xgb_threshold}, exit≤{args.xgb_exit}")
    print(f"  Risk: hard_stop={args.hard_stop*100:.0f}%, ATR={args.atr_mult}x, "
          f"risk/trade={args.risk_per_trade*100:.1f}%, CB_halt={args.cb_halt*100:.0f}%")

    if args.windows:
        starts = args.windows.split(",")
        print(f"\nRunning {len(starts)} window(s) of {args.window_days} days...")

        all_reports = []
        for start_str in starts:
            start = pd.Timestamp(start_str, tz="UTC")
            end = start + pd.Timedelta(days=args.window_days)

            window_features = {}
            for pair, feat_df in oos_features.items():
                w = feat_df[(feat_df.index >= start) & (feat_df.index < end)]
                if not w.empty:
                    window_features[pair] = w

            common_idx = pd.DatetimeIndex(sorted(
                set().union(*(set(df.index) for df in window_features.values()))
            ))
            print(f"\n  Window {start_str} ({args.window_days}d): {len(common_idx)} bars")

            report = run_trendhold_backtest(
                window_features, proba_maps, common_idx, args,
                label=f"{start_str} ({args.window_days}d)",
            )
            all_reports.append(report)
            print_report(report)

        if len(all_reports) > 1:
            print(f"\n{'=' * 60}")
            print("  WINDOW SUMMARY")
            print("=" * 60)
            print(f"  {'Window':30s} {'Trades':>7s} {'Return':>8s} {'Sharpe':>8s} "
                  f"{'Sortino':>8s} {'MaxDD':>8s} {'Active':>8s} {'CB':>4s}")
            for r in all_reports:
                cb_str = f"{r['cb_halted_bars']}" if r['cb_halted_bars'] > 0 else "-"
                print(f"  {r['label']:30s} {r['n_trades']:>7d} "
                      f"{r['total_return_pct']:>+7.2f}% {r['sharpe']:>8.3f} "
                      f"{r['sortino']:>8.3f} {r['max_drawdown_pct']:>7.2f}% "
                      f"{r['active_days']:>4d}/{r['total_days']}   {cb_str:>4s}")
    else:
        common_index = pd.DatetimeIndex(sorted(
            set().union(*(set(df.index) for df in oos_features.values()))
        ))
        report = run_trendhold_backtest(
            oos_features, proba_maps, common_index, args, label="Full OOS",
        )
        print_report(report)


if __name__ == "__main__":
    main()

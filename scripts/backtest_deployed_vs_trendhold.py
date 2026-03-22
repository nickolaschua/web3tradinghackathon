#!/usr/bin/env python3
"""
Head-to-head: Deployed strategy (XGB BTC + XGB SOL + MR fallback)
              vs TrendHold (ML-gated conviction holds + filler)

Runs BOTH strategies on the same rolling windows for direct comparison.
Both use the production RiskManager from bot/execution/risk.py.

Usage:
  python scripts/backtest_deployed_vs_trendhold.py
"""
from __future__ import annotations

import pickle
import sys
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

# ── Shared Data Loading (same as trendhold_production.py) ─────────────────────

_BINANCE_TO_ROOSTOO = {
    "BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD", "SOLUSDT": "SOL/USD",
}


def load_data():
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
    return data


def compute_all_features(raw_data):
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


def build_proba_map(features_df, model_path, oos_start="2024-01-01"):
    path = Path(model_path)
    if not path.exists():
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
    return dict(zip(oos.index, probas))


# ── MR Signal Logic (replicates bot/strategy/mean_reversion.py) ───────────────

def mr_signal(row) -> tuple[str, float]:
    """Returns (direction, confidence). direction in ('BUY','SELL','HOLD')."""
    ema20 = row.get("EMA_20", np.nan)
    ema50 = row.get("EMA_50", np.nan)
    if not (ema20 > ema50):
        return "HOLD", 0.0
    rsi = row.get("RSI_14", 50.0)
    bb = row.get("bb_pos", 0.5)
    macdh = row.get("MACDh_12_26_9", 0.0)
    if rsi < 30 and bb < 0.15 and macdh > 0:
        return "BUY", 0.60
    if rsi < 25:
        return "BUY", 0.55
    if rsi > 55 or bb > 0.6:
        return "SELL", 0.70
    return "HOLD", 0.0


# ── Backtest Core ─────────────────────────────────────────────────────────────

def compute_metrics(equity, trade_dates, closed_trades, common_index, positions):
    rets = np.diff(equity) / equity[:-1]
    total_return = (equity[-1] / equity[0]) - 1.0
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-10
    P = 35_040
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(P)
    down = rets[rets < 0]
    down_std = np.std(down, ddof=1) if len(down) > 1 else 1e-10
    sortino = (mean_ret / (down_std + 1e-10)) * np.sqrt(P)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    calmar = (mean_ret * P) / (abs(max_dd) + 1e-10) if max_dd < 0 else 0.0
    total_days = (common_index[-1] - common_index[0]).days + 1
    active_days = len(trade_dates)
    return {
        "n_trades": len(closed_trades),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "active_days": active_days,
        "total_days": total_days,
        "open_positions": len(positions),
    }


def run_deployed(features, proba_maps, common_index):
    """
    Emulates main.py deployed strategy:
      - XGB BTC (threshold=0.65, exit=0.10)
      - XGB SOL (threshold=0.75, exit=0.10)
      - MR fallback on all 3 pairs when XGB says HOLD
      - Production RiskManager: hard_stop=8%, ATR=2x, max_positions=3
    """
    risk_config = {
        "hard_stop_pct": 0.08,
        "atr_stop_multiplier": 2.0,
        "max_positions": 3,
        "max_single_position_pct": 0.40,
        "risk_per_trade_pct": 0.02,
        "expected_win_loss_ratio": 1.5,
        "circuit_breaker": {"halt_threshold": 0.30},
    }
    risk_mgr = RiskManager(risk_config)
    cash = 1_000_000.0
    risk_mgr.initialize_hwm(cash)
    fee = 10 / 10_000

    xgb_thresholds = {"BTC/USD": 0.65, "SOL/USD": 0.75, "ETH/USD": 0.65}
    xgb_exit = 0.10

    positions = {}
    closed_trades = []
    trade_dates = set()
    n = len(common_index)
    equity = np.zeros(n)
    equity[0] = cash

    close_cache = {}
    atr_cache = {}
    feat_cache = {}
    for pair in features:
        aligned = features[pair].reindex(common_index)
        close_cache[pair] = aligned["close"].values
        atr_cache[pair] = aligned.get("atr_proxy", aligned["close"] * 0.02).values
        feat_cache[pair] = aligned

    for bar_idx in range(n):
        ts = common_index[bar_idx]

        pos_value = sum(
            pos["qty"] * close_cache[p][bar_idx]
            for p, pos in positions.items()
            if not np.isnan(close_cache[p][bar_idx])
        )
        portfolio_value = cash + pos_value
        equity[bar_idx] = portfolio_value
        cb_active = risk_mgr.check_circuit_breaker(portfolio_value)

        # Check stops
        for pair in list(positions.keys()):
            c = close_cache[pair][bar_idx]
            if np.isnan(c):
                continue
            atr_val = atr_cache[pair][bar_idx]
            if np.isnan(atr_val):
                atr_val = c * 0.02
            stop = risk_mgr.check_stops(pair, c, atr_val)
            if stop.should_exit:
                pos = positions[pair]
                proceeds = pos["qty"] * c * (1 - fee)
                pnl = (c * (1 - fee) - pos["entry_price"]) / pos["entry_price"]
                closed_trades.append({"pair": pair, "pnl_pct": pnl, "exit": stop.exit_type})
                cash += proceeds
                risk_mgr.record_exit(pair)
                del positions[pair]
                trade_dates.add(ts.date())

        # Generate signals per pair
        for pair in ["BTC/USD", "SOL/USD", "ETH/USD"]:
            c = close_cache.get(pair, np.array([]))[bar_idx] if pair in close_cache else np.nan
            if np.isnan(c) or c <= 0:
                continue

            pmap = proba_maps.get(pair, {})
            p = pmap.get(ts, 0.5)
            threshold = xgb_thresholds.get(pair, 0.65)

            direction = "HOLD"
            confidence = 0.5

            # Primary: XGB signal
            if p >= threshold:
                direction = "BUY"
                confidence = p
            elif p <= xgb_exit:
                direction = "SELL"
                confidence = p

            # Fallback: MR when XGB says HOLD
            if direction == "HOLD":
                row = feat_cache[pair].iloc[bar_idx] if pair in feat_cache else {}
                if hasattr(row, "get"):
                    mr_dir, mr_conf = mr_signal(row)
                    if mr_dir != "HOLD":
                        direction = mr_dir
                        confidence = mr_conf

            # Execute
            if direction == "BUY" and pair not in positions and not cb_active:
                atr_val = atr_cache.get(pair, np.array([]))[bar_idx]
                if np.isnan(atr_val):
                    atr_val = c * 0.02
                open_usd = {
                    pp: pos["qty"] * close_cache[pp][bar_idx]
                    for pp, pos in positions.items()
                    if not np.isnan(close_cache[pp][bar_idx])
                }
                sizing = risk_mgr.size_new_position(
                    pair=pair, current_price=c, current_atr=atr_val,
                    free_balance_usd=cash, open_positions=open_usd,
                    regime_multiplier=1.0, confidence=confidence,
                    portfolio_weight=0.33,
                )
                if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity > 0:
                    entry_cost = sizing.approved_quantity * c * (1 + fee)
                    if entry_cost <= cash:
                        cash -= entry_cost
                        positions[pair] = {
                            "qty": sizing.approved_quantity,
                            "entry_price": c * (1 + fee),
                        }
                        risk_mgr.record_entry(pair, c * (1 + fee), sizing.trailing_stop_price)
                        trade_dates.add(ts.date())

            elif direction == "SELL" and pair in positions:
                pos = positions[pair]
                proceeds = pos["qty"] * c * (1 - fee)
                pnl = (c * (1 - fee) - pos["entry_price"]) / pos["entry_price"]
                closed_trades.append({"pair": pair, "pnl_pct": pnl, "exit": "signal"})
                cash += proceeds
                risk_mgr.record_exit(pair)
                del positions[pair]
                trade_dates.add(ts.date())

    return compute_metrics(equity, trade_dates, closed_trades, common_index, positions)


def run_trendhold(features, proba_maps, common_index):
    """
    TrendHold with production risk: XGB-gated ETH/SOL holds + BTC filler.
    """
    risk_config = {
        "hard_stop_pct": 0.12,
        "atr_stop_multiplier": 25.0,
        "max_positions": 6,
        "max_single_position_pct": 0.45,
        "risk_per_trade_pct": 0.03,
        "expected_win_loss_ratio": 1.5,
        "circuit_breaker": {"halt_threshold": 0.15},
    }
    risk_mgr = RiskManager(risk_config)
    cash = 1_000_000.0
    risk_mgr.initialize_hwm(cash)
    fee = 10 / 10_000

    hold_pairs = {"ETH/USD": 0.40, "SOL/USD": 0.30}
    filler_pair = "BTC/USD"
    filler_bars = 96
    xgb_threshold = 0.55
    xgb_exit_thresh = 0.10

    positions = {}
    closed_trades = []
    trade_dates = set()
    n = len(common_index)
    equity = np.zeros(n)
    equity[0] = cash

    close_cache = {}
    atr_cache = {}
    for pair in features:
        aligned = features[pair].reindex(common_index)
        close_cache[pair] = aligned["close"].values
        atr_cache[pair] = aligned.get("atr_proxy", aligned["close"] * 0.02).values

    for bar_idx in range(n):
        ts = common_index[bar_idx]
        pos_value = sum(
            pos["qty"] * close_cache[p][bar_idx]
            for p, pos in positions.items()
            if not np.isnan(close_cache[p][bar_idx])
        )
        portfolio_value = cash + pos_value
        equity[bar_idx] = portfolio_value
        cb_active = risk_mgr.check_circuit_breaker(portfolio_value)

        # Stops
        for pair in list(positions.keys()):
            c = close_cache[pair][bar_idx]
            if np.isnan(c):
                continue
            atr_val = atr_cache[pair][bar_idx]
            if np.isnan(atr_val):
                atr_val = c * 0.02
            stop = risk_mgr.check_stops(pair, c, atr_val)
            if stop.should_exit:
                pos = positions[pair]
                proceeds = pos["qty"] * c * (1 - fee)
                pnl = (c * (1 - fee) - pos["entry_price"]) / pos["entry_price"]
                closed_trades.append({"pair": pair, "pnl_pct": pnl, "exit": stop.exit_type})
                cash += proceeds
                risk_mgr.record_exit(pair)
                del positions[pair]
                trade_dates.add(ts.date())

        # XGB exits for hold positions
        for pair in list(positions.keys()):
            if positions[pair].get("source") != "hold":
                continue
            p = proba_maps.get(pair, {}).get(ts, 0.5)
            if p <= xgb_exit_thresh:
                c = close_cache[pair][bar_idx]
                if np.isnan(c):
                    continue
                pos = positions[pair]
                proceeds = pos["qty"] * c * (1 - fee)
                pnl = (c * (1 - fee) - pos["entry_price"]) / pos["entry_price"]
                closed_trades.append({"pair": pair, "pnl_pct": pnl, "exit": "xgb_exit"})
                cash += proceeds
                risk_mgr.record_exit(pair)
                del positions[pair]
                trade_dates.add(ts.date())

        # Filler cycling
        if filler_pair in positions and positions[filler_pair].get("source") == "filler":
            if bar_idx - positions[filler_pair]["entry_bar"] >= filler_bars:
                c = close_cache[filler_pair][bar_idx]
                if not np.isnan(c):
                    pos = positions[filler_pair]
                    proceeds = pos["qty"] * c * (1 - fee)
                    pnl = (c * (1 - fee) - pos["entry_price"]) / pos["entry_price"]
                    closed_trades.append({"pair": filler_pair, "pnl_pct": pnl, "exit": "filler_cycle"})
                    cash += proceeds
                    risk_mgr.record_exit(filler_pair)
                    del positions[filler_pair]
                    trade_dates.add(ts.date())

        # Hold entries
        if not cb_active:
            for pair, alloc in hold_pairs.items():
                if pair in positions:
                    continue
                c = close_cache.get(pair, np.array([]))[bar_idx] if pair in close_cache else np.nan
                if np.isnan(c) or c <= 0:
                    continue
                p = proba_maps.get(pair, {}).get(ts, 0.5)
                if p < xgb_threshold:
                    continue
                atr_val = atr_cache.get(pair, np.array([]))[bar_idx]
                if np.isnan(atr_val):
                    atr_val = c * 0.02
                open_usd = {
                    pp: pos["qty"] * close_cache[pp][bar_idx]
                    for pp, pos in positions.items()
                    if not np.isnan(close_cache[pp][bar_idx])
                }
                sizing = risk_mgr.size_new_position(
                    pair=pair, current_price=c, current_atr=atr_val,
                    free_balance_usd=cash, open_positions=open_usd,
                    regime_multiplier=1.0, confidence=p,
                    portfolio_weight=alloc / 0.33,
                )
                if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity > 0:
                    entry_cost = sizing.approved_quantity * c * (1 + fee)
                    if entry_cost <= cash:
                        cash -= entry_cost
                        positions[pair] = {
                            "qty": sizing.approved_quantity,
                            "entry_price": c * (1 + fee),
                            "entry_bar": bar_idx,
                            "source": "hold",
                        }
                        risk_mgr.record_entry(pair, c * (1 + fee), sizing.trailing_stop_price)
                        trade_dates.add(ts.date())

        # Filler entry
        if filler_pair not in positions and not cb_active:
            c = close_cache.get(filler_pair, np.array([]))[bar_idx] if filler_pair in close_cache else np.nan
            if not np.isnan(c) and c > 0:
                atr_val = atr_cache.get(filler_pair, np.array([]))[bar_idx]
                if np.isnan(atr_val):
                    atr_val = c * 0.02
                open_usd = {
                    pp: pos["qty"] * close_cache[pp][bar_idx]
                    for pp, pos in positions.items()
                    if not np.isnan(close_cache[pp][bar_idx])
                }
                sizing = risk_mgr.size_new_position(
                    pair=filler_pair, current_price=c, current_atr=atr_val,
                    free_balance_usd=cash, open_positions=open_usd,
                    regime_multiplier=1.0, confidence=0.55,
                    portfolio_weight=0.10 / 0.33,
                )
                if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity > 0:
                    entry_cost = sizing.approved_quantity * c * (1 + fee)
                    if entry_cost <= cash:
                        cash -= entry_cost
                        positions[filler_pair] = {
                            "qty": sizing.approved_quantity,
                            "entry_price": c * (1 + fee),
                            "entry_bar": bar_idx,
                            "source": "filler",
                        }
                        risk_mgr.record_entry(filler_pair, c * (1 + fee), sizing.trailing_stop_price)
                        trade_dates.add(ts.date())

    return compute_metrics(equity, trade_dates, closed_trades, common_index, positions)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  HEAD-TO-HEAD: Deployed (XGB+MR) vs TrendHold (ML-gated + Prod Risk)")
    print("=" * 80)

    raw_data = load_data()
    all_features = compute_all_features(raw_data)

    proba_maps = {}
    for pair, mpath in [
        ("ETH/USD", "models/xgb_eth_15m.pkl"),
        ("SOL/USD", "models/xgb_sol_15m.pkl"),
        ("BTC/USD", "models/xgb_btc_15m.pkl"),
    ]:
        if pair in all_features:
            pm = build_proba_map(all_features[pair], mpath)
            if pm:
                proba_maps[pair] = pm

    oos_ts = pd.Timestamp("2024-01-01", tz="UTC")
    oos_features = {
        p: f[f.index >= oos_ts]
        for p, f in all_features.items()
        if len(f[f.index >= oos_ts]) > 100
    }

    windows = [
        "2024-03-01", "2024-05-01", "2024-07-01", "2024-09-01", "2024-11-01",
        "2025-01-01", "2025-03-01", "2025-05-01", "2025-07-01", "2025-09-01",
        "2025-11-01", "2026-01-01", "2026-02-01", "2026-02-19",
        "2026-03-01", "2026-03-11",
    ]

    deployed_results = []
    trendhold_results = []

    for start_str in windows:
        start = pd.Timestamp(start_str, tz="UTC")
        end = start + pd.Timedelta(days=10)

        wf = {}
        for pair, feat_df in oos_features.items():
            w = feat_df[(feat_df.index >= start) & (feat_df.index < end)]
            if not w.empty:
                wf[pair] = w

        common_idx = pd.DatetimeIndex(sorted(
            set().union(*(set(df.index) for df in wf.values()))
        ))

        d = run_deployed(wf, proba_maps, common_idx)
        t = run_trendhold(wf, proba_maps, common_idx)
        d["window"] = start_str
        t["window"] = start_str
        deployed_results.append(d)
        trendhold_results.append(t)

    # Print comparison
    print(f"\n{'Window':>12s}  |  {'--- Deployed (XGB+MR) ---':^36s}  |  {'--- TrendHold (ML+Risk) ---':^36s}  |  {'Winner':>8s}")
    print(f"{'':>12s}  |  {'Ret':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Days':>5s}  |  {'Ret':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Days':>5s}  |")
    print("-" * 110)

    d_wins = 0
    t_wins = 0
    for d, t in zip(deployed_results, trendhold_results):
        d_sharpe = d["sharpe"]
        t_sharpe = t["sharpe"]
        winner = "DEP" if d_sharpe > t_sharpe else "TH"
        if d_sharpe > t_sharpe:
            d_wins += 1
        else:
            t_wins += 1

        print(
            f"  {d['window']:>10s}  |  "
            f"{d['total_return_pct']:>+6.2f}% {d['sharpe']:>7.2f} {d['max_drawdown_pct']:>6.2f}% {d['active_days']:>3d}/{d['total_days']}  |  "
            f"{t['total_return_pct']:>+6.2f}% {t['sharpe']:>7.2f} {t['max_drawdown_pct']:>6.2f}% {t['active_days']:>3d}/{t['total_days']}  |  "
            f"{'<< DEP' if winner == 'DEP' else 'TH >>'}"
        )

    # Averages
    avg_d_sharpe = np.mean([r["sharpe"] for r in deployed_results])
    avg_t_sharpe = np.mean([r["sharpe"] for r in trendhold_results])
    avg_d_ret = np.mean([r["total_return_pct"] for r in deployed_results])
    avg_t_ret = np.mean([r["total_return_pct"] for r in trendhold_results])
    avg_d_dd = np.mean([r["max_drawdown_pct"] for r in deployed_results])
    avg_t_dd = np.mean([r["max_drawdown_pct"] for r in trendhold_results])
    avg_d_days = np.mean([r["active_days"] for r in deployed_results])
    avg_t_days = np.mean([r["active_days"] for r in trendhold_results])

    print("-" * 110)
    print(
        f"  {'AVERAGE':>10s}  |  "
        f"{avg_d_ret:>+6.2f}% {avg_d_sharpe:>7.2f} {avg_d_dd:>6.2f}% {avg_d_days:>5.1f}  |  "
        f"{avg_t_ret:>+6.2f}% {avg_t_sharpe:>7.2f} {avg_t_dd:>6.2f}% {avg_t_days:>5.1f}  |"
    )
    print(f"\n  Deployed wins: {d_wins}/{len(windows)}    TrendHold wins: {t_wins}/{len(windows)}")

    # Competition-relevant windows
    print(f"\n  COMPETITION-RELEVANT (2026 windows only):")
    comp_windows = ["2026-02-19", "2026-03-01", "2026-03-11"]
    for d, t in zip(deployed_results, trendhold_results):
        if d["window"] in comp_windows:
            print(
                f"    {d['window']:>10s}  DEP: {d['total_return_pct']:>+6.2f}% Sharpe={d['sharpe']:>6.2f}  "
                f"TH: {t['total_return_pct']:>+6.2f}% Sharpe={t['sharpe']:>6.2f}"
            )


if __name__ == "__main__":
    main()

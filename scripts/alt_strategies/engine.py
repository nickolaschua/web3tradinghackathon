"""
Multi-coin backtest engine with pluggable strategies.

Supports:
- Multiple concurrent positions (configurable max)
- ATR trailing stops with hard stop floor
- Tiered circuit breaker (can disable for 10-day windows)
- Portfolio-weight or risk-based position sizing
- Per-trade source attribution and reporting
- 10-day window backtests for competition testing
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    max_positions: int = 5
    risk_per_trade_pct: float = 0.02
    hard_stop_pct: float = 0.05
    atr_stop_mult: float = 10.0
    max_single_position_pct: float = 0.40
    fee_bps: int = 10
    cb_halt: float = 0.30
    cb_heavy: float = 0.20
    cb_light: float = 0.10
    enable_cb: bool = True   # set False for 10-day window tests
    expected_win_loss_ratio: float = 1.5


# ── Data Types ────────────────────────────────────────────────────────────────

@dataclass
class PositionState:
    pair: str
    entry_price: float
    units: float
    trail_stop: float
    source: str
    entry_bar: pd.Timestamp


@dataclass
class SignalCandidate:
    pair: str
    direction: str        # "BUY"
    source: str           # strategy name
    confidence: float     # 0-1
    size: float           # portfolio fraction for portfolio_pct, or raw P for risk_based
    sizing_mode: str = "portfolio_pct"  # "portfolio_pct" | "risk_based"


@dataclass
class BacktestResult:
    returns: pd.Series
    portfolio: pd.Series
    trades: list[dict]
    gate_stats: dict
    window_label: str = ""


# ── Strategy Protocol ─────────────────────────────────────────────────────────

class Strategy(Protocol):
    name: str

    def generate_entries(
        self,
        ts: pd.Timestamp,
        bar_idx: int,
        coin_data: dict[str, dict],
        open_pairs: set[str],
        portfolio_value: float,
    ) -> list[SignalCandidate]:
        ...

    def check_exit(
        self,
        ts: pd.Timestamp,
        bar_idx: int,
        pair: str,
        source: str,
        coin_row: dict,
        coin_data: dict[str, dict],
    ) -> bool:
        ...


# ── Circuit Breaker ───────────────────────────────────────────────────────────

def cb_multiplier(drawdown: float, cfg: BacktestConfig) -> float:
    if not cfg.enable_cb:
        return 1.0
    if drawdown >= cfg.cb_halt:
        return 0.0
    if drawdown >= cfg.cb_heavy:
        return 0.25
    if drawdown >= cfg.cb_light:
        return 0.50
    return 1.0


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Bar-by-bar multi-coin backtest with pluggable strategies.

    Usage:
        engine = BacktestEngine(config, strategies)
        result = engine.run(coin_features, common_index)
    """

    def __init__(self, config: BacktestConfig, strategies: list):
        self.cfg = config
        self.strategies = strategies

    def run(
        self,
        coin_features: dict[str, pd.DataFrame],
        common_index: pd.DatetimeIndex,
        label: str = "",
    ) -> BacktestResult:
        cfg = self.cfg
        fee_rate = cfg.fee_bps / 10_000.0
        n = len(common_index)

        # Pre-build aligned numpy matrices for fast bar access
        pairs = sorted(coin_features.keys())
        n_coins = len(pairs)
        pair_to_idx = {p: i for i, p in enumerate(pairs)}

        close_mat = np.full((n, n_coins), np.nan)
        atr_mat = np.full((n, n_coins), np.nan)

        # Pre-build feature dicts using reindex + to_dict for speed
        feature_rows: list[dict[str, dict]] = [dict() for _ in range(n)]

        print("  Pre-building feature matrices...")
        bar_lookup = {ts: i for i, ts in enumerate(common_index)}

        for ci, pair in enumerate(pairs):
            feat_df = coin_features[pair]
            aligned = feat_df.reindex(common_index)

            close_mat[:, ci] = aligned["close"].values
            if "atr_proxy" in aligned.columns:
                atr_mat[:, ci] = aligned["atr_proxy"].values

            # Convert to row-oriented dict in bulk (fast), then assign to bar slots
            valid_mask = aligned["close"].notna()
            valid_df = aligned[valid_mask]
            row_dicts = valid_df.to_dict(orient="index")
            for ts, row_dict in row_dicts.items():
                bar_idx = bar_lookup[ts]
                feature_rows[bar_idx][pair] = row_dict

        print(f"  Matrices built: {n_coins} coins × {n:,} bars")

        # Portfolio state
        free_balance = cfg.initial_capital
        portfolio_hwm = cfg.initial_capital
        positions: dict[str, PositionState] = {}

        portfolio_values = np.zeros(n)
        portfolio_values[0] = cfg.initial_capital
        returns = np.zeros(n)
        closed_trades: list[dict] = []
        gate_stats = defaultdict(int)

        t_start = time.time()

        for bar_idx in range(n):
            ts = common_index[bar_idx]
            coin_data = feature_rows[bar_idx]

            # ── 1. Update trailing stops ─────────────────────────────────
            for pair, pos in list(positions.items()):
                ci = pair_to_idx.get(pair)
                if ci is None:
                    continue
                c = close_mat[bar_idx, ci]
                atr = atr_mat[bar_idx, ci]
                if np.isnan(c):
                    continue
                if not np.isnan(atr) and atr > 0:
                    pos.trail_stop = max(pos.trail_stop, c - cfg.atr_stop_mult * atr)

            # ── 2. Check exits ───────────────────────────────────────────
            just_exited = set()
            for pair, pos in list(positions.items()):
                ci = pair_to_idx.get(pair)
                if ci is None or np.isnan(close_mat[bar_idx, ci]):
                    continue
                c = close_mat[bar_idx, ci]
                coin_row = coin_data.get(pair, {})

                stop_hit = c <= pos.trail_stop

                # Ask the source strategy for signal-based exit
                signal_exit = False
                for strat in self.strategies:
                    if strat.name == pos.source:
                        signal_exit = strat.check_exit(
                            ts, bar_idx, pair, pos.source, coin_row, coin_data,
                        )
                        break

                if stop_hit or signal_exit:
                    proceeds = pos.units * c * (1.0 - fee_rate)
                    net_exit = c * (1.0 - fee_rate)
                    pnl_pct = (net_exit - pos.entry_price) / pos.entry_price
                    closed_trades.append({
                        "pair": pair, "entry_bar": pos.entry_bar, "exit_bar": ts,
                        "entry_price": pos.entry_price, "exit_price": net_exit,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "stop" if stop_hit else "signal",
                        "source": pos.source,
                    })
                    free_balance += proceeds
                    del positions[pair]
                    just_exited.add(pair)

            # ── 3. Mark to market ────────────────────────────────────────
            pos_value = 0.0
            for pair, pos in positions.items():
                ci = pair_to_idx.get(pair)
                if ci is not None and not np.isnan(close_mat[bar_idx, ci]):
                    pos_value += pos.units * close_mat[bar_idx, ci]
            total_portfolio = free_balance + pos_value
            portfolio_hwm = max(portfolio_hwm, total_portfolio)
            dd = (portfolio_hwm - total_portfolio) / portfolio_hwm if portfolio_hwm > 0 else 0.0
            cb_mult = cb_multiplier(dd, cfg)

            # ── 4. Collect BUY signals from all strategies ───────────────
            open_pairs = set(positions.keys()) | just_exited
            all_candidates: list[SignalCandidate] = []

            for strat in self.strategies:
                cands = strat.generate_entries(
                    ts, bar_idx, coin_data, open_pairs, total_portfolio,
                )
                all_candidates.extend(cands)

            # Deduplicate: keep highest-confidence signal per pair
            best_per_pair: dict[str, SignalCandidate] = {}
            for c in all_candidates:
                if c.pair in open_pairs:
                    continue
                existing = best_per_pair.get(c.pair)
                if existing is None or c.confidence > existing.confidence:
                    best_per_pair[c.pair] = c

            # Rank by confidence descending
            candidates = sorted(best_per_pair.values(), key=lambda s: s.confidence, reverse=True)

            # ── 5. Size and fill ─────────────────────────────────────────
            for cand in candidates:
                if len(positions) >= cfg.max_positions:
                    gate_stats["max_pos_blocked"] += 1
                    continue

                if cb_mult == 0.0:
                    gate_stats["cb_halted"] += 1
                    continue

                ci = pair_to_idx.get(cand.pair)
                if ci is None:
                    continue
                c = close_mat[bar_idx, ci]
                atr = atr_mat[bar_idx, ci]
                if np.isnan(c):
                    continue

                if cb_mult < 1.0:
                    gate_stats["cb_reduced"] += 1

                # Compute stop
                hard_stop_price = c * (1.0 - cfg.hard_stop_pct)
                if not np.isnan(atr) and atr > 0:
                    atr_stop_price = c - cfg.atr_stop_mult * atr
                    initial_stop = max(hard_stop_price, atr_stop_price)
                else:
                    initial_stop = hard_stop_price

                # Position sizing
                if cand.sizing_mode == "risk_based":
                    stop_distance = c - initial_stop
                    stop_distance = min(stop_distance, c * cfg.hard_stop_pct)
                    if stop_distance <= 0:
                        stop_distance = c * cfg.hard_stop_pct
                    risk_usd = total_portfolio * cfg.risk_per_trade_pct * cand.size * cb_mult
                    quantity = risk_usd / stop_distance
                    target_usd = quantity * c
                else:
                    target_usd = total_portfolio * cand.size * cb_mult

                usable = free_balance * 0.95
                target_usd = min(target_usd, total_portfolio * cfg.max_single_position_pct, usable)

                if target_usd >= 10.0:
                    entry_units = target_usd / c
                    entry_fee = target_usd * fee_rate
                    free_balance -= (target_usd + entry_fee)
                    positions[cand.pair] = PositionState(
                        pair=cand.pair,
                        entry_price=c * (1.0 + fee_rate),
                        units=entry_units,
                        trail_stop=initial_stop,
                        source=cand.source,
                        entry_bar=ts,
                    )

            # ── 6. Record portfolio value ────────────────────────────────
            pos_value = 0.0
            for pair, pos in positions.items():
                ci = pair_to_idx.get(pair)
                if ci is not None and not np.isnan(close_mat[bar_idx, ci]):
                    pos_value += pos.units * close_mat[bar_idx, ci]
            portfolio_values[bar_idx] = free_balance + pos_value
            if bar_idx > 0:
                returns[bar_idx] = portfolio_values[bar_idx] / portfolio_values[bar_idx - 1] - 1.0

            if bar_idx > 0 and bar_idx % 10_000 == 0:
                elapsed = time.time() - t_start
                pct = bar_idx / n * 100
                print(f"    [{pct:5.1f}%] bar {bar_idx:,}/{n:,}  trades={len(closed_trades)}  "
                      f"open={len(positions)}  equity=${portfolio_values[bar_idx]:,.0f}  "
                      f"elapsed={elapsed:.0f}s")

        elapsed = time.time() - t_start
        print(f"  Done: {len(closed_trades)} closed, {len(positions)} open, {elapsed:.1f}s")

        return BacktestResult(
            returns=pd.Series(returns, index=common_index),
            portfolio=pd.Series(portfolio_values, index=common_index),
            trades=closed_trades,
            gate_stats=dict(gate_stats),
            window_label=label,
        )

    def run_windows(
        self,
        coin_features: dict[str, pd.DataFrame],
        window_starts: list[str],
        window_days: int = 10,
    ) -> list[BacktestResult]:
        """Run backtest on multiple N-day windows for competition testing."""
        results = []
        for start_str in window_starts:
            start = pd.Timestamp(start_str, tz="UTC")
            end = start + pd.Timedelta(days=window_days)
            label = f"{start_str} ({window_days}d)"

            # Filter features to window
            window_features = {}
            for pair, feat_df in coin_features.items():
                window = feat_df[(feat_df.index >= start) & (feat_df.index < end)]
                if not window.empty:
                    window_features[pair] = window

            if not window_features:
                print(f"  Window {label}: no data, skipping")
                continue

            common_idx = pd.DatetimeIndex(sorted(
                set().union(*(set(df.index) for df in window_features.values()))
            ))
            print(f"\n  Window {label}: {len(common_idx):,} bars, {len(window_features)} coins")
            for strat in self.strategies:
                if hasattr(strat, "reset"):
                    strat.reset()
            result = self.run(window_features, common_idx, label=label)
            results.append(result)

        return results


# ── Reporting ─────────────────────────────────────────────────────────────────

PERIODS_15M = 35_040

def compute_report(result: BacktestResult, initial_capital: float = 1_000_000.0) -> dict:
    trades = result.trades
    n_trades = len(trades)
    eq = result.portfolio.values

    source_counts = defaultdict(int)
    source_pnl = defaultdict(float)
    coin_counts = defaultdict(int)
    for t in trades:
        source_counts[t["source"]] += 1
        source_pnl[t["source"]] += t["pnl_pct"]
        coin_counts[t["pair"]] += 1

    winners = sum(1 for t in trades if t["pnl_pct"] > 0) if n_trades else 0
    win_rate = winners / n_trades if n_trades else 0.0
    avg_pnl = sum(t["pnl_pct"] for t in trades) / n_trades if n_trades else 0.0
    stop_exits = sum(1 for t in trades if t["exit_reason"] == "stop")

    trade_dates = set()
    for t in trades:
        trade_dates.add(pd.Timestamp(t["entry_bar"]).date())
        trade_dates.add(pd.Timestamp(t["exit_bar"]).date())

    total_days = (result.portfolio.index[-1] - result.portfolio.index[0]).days + 1
    active_days = len(trade_dates)

    # 10-day sliding window check
    min_active_10 = 0
    if trade_dates:
        all_dates = pd.date_range(
            result.portfolio.index[0].date(),
            result.portfolio.index[-1].date(), freq="D",
        )
        min_active_10 = total_days
        for i in range(len(all_dates) - 9):
            window = set(all_dates[i:i+10].date)
            min_active_10 = min(min_active_10, len(window & trade_dates))

    # Risk metrics (numpy, no quantstats dependency)
    rets = np.diff(eq) / eq[:-1]
    total_return = (eq[-1] / eq[0]) - 1.0
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-10
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(PERIODS_15M)

    down = rets[rets < 0]
    down_std = np.std(down, ddof=1) if len(down) > 1 else 1e-10
    sortino = (mean_ret / (down_std + 1e-10)) * np.sqrt(PERIODS_15M)

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    calmar = (mean_ret * PERIODS_15M) / (abs(max_dd) + 1e-10) if max_dd < 0 else 0.0

    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    return {
        "label": result.window_label,
        "n_trades": n_trades,
        "trades_per_day": n_trades / max(total_days, 1),
        "active_days": active_days,
        "total_days": total_days,
        "active_pct": active_days / max(total_days, 1) * 100,
        "source_counts": dict(source_counts),
        "source_avg_pnl": {k: v / max(source_counts[k], 1) * 100 for k, v in source_pnl.items()},
        "coin_counts": dict(coin_counts),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "composite": round(composite, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_return * 100, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_pnl_pct": round(avg_pnl * 100, 3),
        "stop_exits": stop_exits,
        "signal_exits": n_trades - stop_exits,
        "min_active_in_10d": min_active_10,
        "meets_8_of_10": min_active_10 >= 8,
        "gate_stats": result.gate_stats,
        "final_portfolio": float(eq[-1]),
    }


def print_report(report: dict) -> None:
    sep = "=" * 60
    label = f"  {report['label']}" if report.get("label") else ""
    print(f"\n{sep}")
    print(f"  BACKTEST RESULTS{label}")
    print(sep)
    print(f"  Trades: {report['n_trades']}  ({report['trades_per_day']:.2f}/day)")
    print(f"  Active days: {report['active_days']}/{report['total_days']} ({report['active_pct']:.1f}%)")
    print(f"  Min in 10d window: {report['min_active_in_10d']}/10  "
          f"{'PASS' if report['meets_8_of_10'] else 'FAIL'}")
    print()
    for src in sorted(report["source_counts"].keys()):
        cnt = report["source_counts"][src]
        pct = cnt / max(report["n_trades"], 1) * 100
        avg = report["source_avg_pnl"].get(src, 0)
        print(f"    {src:20s}: {cnt:>4d} ({pct:5.1f}%)  avg PnL: {avg:+.3f}%")
    print()
    top_coins = sorted(report["coin_counts"].items(), key=lambda x: x[1], reverse=True)[:10]
    for pair, cnt in top_coins:
        print(f"    {pair:12s}: {cnt:>4d}")
    print()
    print(f"  Sharpe:   {report['sharpe']:>8.3f}    Sortino:  {report['sortino']:>8.3f}")
    print(f"  Calmar:   {report['calmar']:>8.3f}    Composite:{report['composite']:>8.3f}")
    print(f"  Return:   {report['total_return_pct']:>+7.2f}%   MaxDD: {report['max_drawdown_pct']:>7.2f}%")
    print(f"  Win rate: {report['win_rate_pct']:>6.1f}%    Avg PnL: {report['avg_pnl_pct']:>+.3f}%")
    print(f"  Stops: {report['stop_exits']}  Signals: {report['signal_exits']}")
    for k, v in report["gate_stats"].items():
        if v > 0:
            print(f"    {k}: {v}")
    print(sep)

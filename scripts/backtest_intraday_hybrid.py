#!/usr/bin/env python3
"""
Hybrid intraday backtest:
- Fast layer (5m/15m) generates triggers
- Slow 4H layer provides directional bias for size scaling
- Execution uses shared RiskManager + PortfolioAllocator semantics
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_features
from bot.data.intraday_features import compute_intraday_features, merge_slow_bias_to_intraday
from bot.execution.portfolio import PortfolioAllocator
from bot.execution.risk import RiskDecision, RiskManager
from bot.strategy.base import SignalDirection
from bot.strategy.intraday_momentum import IntradayMomentumStrategy


RESULTS_DIR = Path("research_results")
CONFIG_PATH = Path("bot/config/config.yaml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = df.columns.str.lower()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def build_slow_bias_4h(btc_4h: pd.DataFrame, oil_daily: pd.DataFrame | None, dxy_daily: pd.DataFrame | None, btc_funding: pd.DataFrame | None) -> pd.DataFrame:
    slow = compute_features(btc_4h.copy())

    if oil_daily is not None:
        oil = oil_daily["close"].resample("4h").ffill().reindex(slow.index, method="ffill")
        slow["oil_return_1d_4h"] = oil.pct_change(6).shift(1)
    else:
        slow["oil_return_1d_4h"] = np.nan

    if dxy_daily is not None:
        dxy = dxy_daily["close"].resample("4h").ffill().reindex(slow.index, method="ffill")
        slow["dxy_return_1d_4h"] = dxy.pct_change(6).shift(1)
    else:
        slow["dxy_return_1d_4h"] = np.nan

    if btc_funding is not None:
        fr = btc_funding["funding_rate"].resample("4h").ffill().reindex(slow.index, method="ffill")
        rm = fr.rolling(90).mean()
        rs = fr.rolling(90).std()
        slow["btc_funding_zscore_4h"] = ((fr - rm) / rs.replace(0, np.nan)).shift(1)
    else:
        slow["btc_funding_zscore_4h"] = np.nan

    slow["EMA_20_4h"] = slow["EMA_20"]
    slow["EMA_50_4h"] = slow["EMA_50"]
    keep = ["EMA_20_4h", "EMA_50_4h", "btc_funding_zscore_4h", "oil_return_1d_4h", "dxy_return_1d_4h"]
    return slow[keep]


def compute_metrics(returns: pd.Series, trades: list[dict], utilization: pd.Series, periods: int) -> dict:
    returns = returns[returns.index.notna()]
    n_trades = len(trades)
    trade_days = {pd.Timestamp(t["ts"]).normalize() for t in trades} if trades else set()
    cal_days = int(returns.index.normalize().nunique()) if len(returns) > 0 else 0
    daily_coverage = (len(trade_days) / cal_days) if cal_days > 0 else 0.0
    trades_per_day = (n_trades / cal_days) if cal_days > 0 else 0.0

    return {
        "total_return_pct": float((1 + returns).prod() - 1) * 100,
        "sharpe": float(qs.stats.sharpe(returns, periods=periods)),
        "sortino": float(qs.stats.sortino(returns, periods=periods)),
        "calmar": float(qs.stats.calmar(returns)),
        "composite_score": 0.4 * float(qs.stats.sortino(returns, periods=periods))
        + 0.3 * float(qs.stats.sharpe(returns, periods=periods))
        + 0.3 * float(qs.stats.calmar(returns)),
        "max_drawdown_pct": float(qs.stats.max_drawdown(returns)) * 100,
        "n_trades": n_trades,
        "daily_coverage": daily_coverage,
        "trades_per_day": trades_per_day,
        "avg_utilization": float(utilization.mean()) if len(utilization) > 0 else 0.0,
        "n_bars": len(returns),
    }


def run_backtest(
    feat: pd.DataFrame,
    strategy: IntradayMomentumStrategy,
    config: dict,
    interval: str,
    fee_bps: int,
    initial_capital: float,
    max_hold_bars: int,
    cooldown_bars: int = 0,
    max_trades_per_day: int = 10_000,
    min_expected_edge: float = 0.0,
):
    fee_rate = fee_bps / 10_000
    pair = "BTC/USD"
    rm = RiskManager(config=config)
    rm.initialize_hwm(initial_capital)

    pa = PortfolioAllocator(config=config)
    # single-asset backtest still calls allocator for interface consistency
    price_hist = {"BTC/USD": pd.DataFrame({"close": feat["close"]})}
    pa.compute_weights(price_hist)
    base_weight = pa.get_pair_weight("BTC/USD", n_active_pairs=1)

    free_balance = initial_capital
    position_units = 0.0
    entry_price = None
    bars_in_position = 0
    cooldown_remaining = 0
    day_trade_count: dict[pd.Timestamp, int] = {}
    prev_portfolio = initial_capital
    returns = []
    util = []
    trades = []
    for idx, row in feat.iterrows():
        close = float(row["close"])
        atr = float(row["atr_proxy"]) if not pd.isna(row["atr_proxy"]) else np.nan
        sig = strategy.generate_signal(pair, pd.DataFrame([row]))
        just_exited = False

        if position_units > 0:
            bars_in_position += 1
            stop_result = rm.check_stops(pair, close, atr)
            time_exit = max_hold_bars > 0 and bars_in_position >= max_hold_bars
            if stop_result.should_exit or sig.direction == SignalDirection.SELL or time_exit:
                exit_price = close * (1 - fee_rate)
                proceeds = position_units * close * (1 - fee_rate)
                pnl_pct = (exit_price - entry_price) / entry_price
                trades.append({"ts": idx, "pnl_pct": pnl_pct})
                free_balance += proceeds
                position_units = 0.0
                entry_price = None
                bars_in_position = 0
                cooldown_remaining = max(0, int(cooldown_bars))
                rm.record_exit(pair)
                just_exited = True

        total_portfolio = free_balance + position_units * close
        rm.check_circuit_breaker(total_portfolio)

        current_day = pd.Timestamp(idx).normalize()
        day_trade_count.setdefault(current_day, 0)
        trade_cap_hit = day_trade_count[current_day] >= int(max_trades_per_day)

        vol_col = "vol_5m" if interval == "5m" else "vol_15m"
        z_col = "zscore_5m" if interval == "5m" else "zscore_15m"
        if strategy.trigger_mode == "mean_reversion":
            expected_edge = abs(float(row.get(z_col, 0.0))) * float(row.get(vol_col, 0.0))
        else:
            expected_edge = abs(float(row.get("return_5m" if interval == "5m" else "return_15m", 0.0)))

        if position_units == 0.0 and (not just_exited) and sig.direction == SignalDirection.BUY and cooldown_remaining == 0 and (not trade_cap_hit) and expected_edge >= min_expected_edge:
            sizing = rm.size_new_position(
                pair=pair,
                current_price=close,
                current_atr=atr,
                free_balance_usd=free_balance,
                open_positions={},
                regime_multiplier=1.0,
                confidence=max(0.1, min(1.0, sig.confidence)),
                portfolio_weight=max(0.01, min(1.0, base_weight * sig.size)),
            )
            if sizing.decision == RiskDecision.APPROVED and sizing.approved_usd_value >= 10.0:
                target_usd = sizing.approved_usd_value
                position_units = target_usd / close
                free_balance -= (target_usd + target_usd * fee_rate)
                entry_price = close * (1 + fee_rate)
                bars_in_position = 0
                day_trade_count[current_day] += 1
                rm.record_entry(pair, entry_price, sizing.trailing_stop_price)

        end_portfolio = free_balance + position_units * close
        returns.append((end_portfolio / prev_portfolio - 1.0) if prev_portfolio > 0 else 0.0)
        prev_portfolio = end_portfolio
        util.append((position_units * close / end_portfolio) if end_portfolio > 0 else 0.0)
        if cooldown_remaining > 0 and position_units == 0.0:
            cooldown_remaining -= 1

    periods = 105_120 if interval == "5m" else 35_040
    returns_s = pd.Series(returns, index=feat.index)
    util_s = pd.Series(util, index=feat.index)
    return compute_metrics(returns_s, trades, util_s, periods)


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid intraday trigger + 4H bias backtest")
    p.add_argument("--interval", choices=["5m", "15m"], default="15m")
    p.add_argument("--btc-intraday", default=None, help="Path to BTCUSDT intraday parquet; defaults by interval")
    p.add_argument("--btc-4h", default="research_data/BTCUSDT_4h.parquet")
    p.add_argument("--oil-daily", default="research_data/oil_daily.parquet")
    p.add_argument("--dxy-daily", default="research_data/dxy_daily.parquet")
    p.add_argument("--btc-funding", default="research_data/BTCUSDT_funding.parquet")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--fee-bps", type=int, default=10)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--trigger-mode", choices=["momentum_breakout", "mean_reversion"], default="momentum_breakout")
    p.add_argument("--return-threshold", type=float, default=0.0015)
    p.add_argument("--volume-ratio-threshold", type=float, default=1.2)
    p.add_argument("--zscore-threshold", type=float, default=1.5)
    p.add_argument("--base-size", type=float, default=0.7)
    p.add_argument("--bias-weight", type=float, default=0.3)
    p.add_argument("--max-hold-bars", type=int, default=8, help="Force exit after N bars to keep intraday turnover")
    p.add_argument("--cooldown-bars", type=int, default=0, help="Bars to wait after an exit before re-entry")
    p.add_argument("--max-trades-per-day", type=int, default=10000, help="Cap entries per day")
    p.add_argument("--min-expected-edge", type=float, default=0.0, help="Skip entries with estimated edge below threshold")
    p.add_argument("--require-trend-confirmation", action="store_true")
    p.add_argument("--require-volume-confirmation", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config()

    intraday_path = args.btc_intraday or f"data/BTCUSDT_{args.interval}.parquet"
    btc_i = load_df(intraday_path)
    btc_4h = load_df(args.btc_4h)
    oil = load_df(args.oil_daily) if Path(args.oil_daily).exists() else None
    dxy = load_df(args.dxy_daily) if Path(args.dxy_daily).exists() else None
    fr = load_df(args.btc_funding) if Path(args.btc_funding).exists() else None

    fast = compute_intraday_features(btc_i, interval=args.interval)
    slow = build_slow_bias_4h(btc_4h, oil, dxy, fr)
    feat = merge_slow_bias_to_intraday(fast, slow)
    feat = feat.dropna()

    start_ts = pd.Timestamp(args.start, tz="UTC")
    feat = feat[feat.index >= start_ts]
    if args.end:
        feat = feat[feat.index <= pd.Timestamp(args.end, tz="UTC")]

    strat_cfg = {
        "interval": args.interval,
        "trigger_mode": args.trigger_mode,
        "return_threshold": args.return_threshold,
        "volume_ratio_threshold": args.volume_ratio_threshold,
        "zscore_threshold": args.zscore_threshold,
        "base_size": args.base_size,
        "bias_weight": args.bias_weight,
        "require_trend_confirmation": args.require_trend_confirmation,
        "require_volume_confirmation": args.require_volume_confirmation,
    }
    strategy = IntradayMomentumStrategy(strat_cfg)
    stats = run_backtest(
        feat,
        strategy,
        config,
        args.interval,
        args.fee_bps,
        args.capital,
        args.max_hold_bars,
        args.cooldown_bars,
        args.max_trades_per_day,
        args.min_expected_edge,
    )

    print("=" * 80)
    print("HYBRID INTRADAY BACKTEST")
    print("=" * 80)
    print(f"Interval: {args.interval} | Trigger: {args.trigger_mode}")
    print(f"Bars: {len(feat):,} | Date range: {feat.index[0]} -> {feat.index[-1]}")
    print(f"Return: {stats['total_return_pct']:+.2f}%")
    print(f"Sharpe: {stats['sharpe']:.3f} | Sortino: {stats['sortino']:.3f} | Calmar: {stats['calmar']:.3f}")
    print(f"Composite: {stats['composite_score']:.3f} | MaxDD: {stats['max_drawdown_pct']:.2f}%")
    print(f"Trades: {stats['n_trades']} | Trades/day: {stats['trades_per_day']:.2f} | Coverage: {100*stats['daily_coverage']:.1f}%")
    print(f"Avg utilization: {100*stats['avg_utilization']:.1f}%")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"intraday_hybrid_{args.interval}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-coin portfolio backtest for the unified 20-coin XGBoost model.

Simulates trading all 20 coins with up to max_positions concurrent positions.
Includes all Sortino improvements from the design spec:
  - Correlation-aware position selection (corr > 0.80 → halve size)
  - Per-coin regime detection (EMA_96/EMA_240 at 15M)
  - Signal ranking by P(BUY) descending
  - Time-based exit (32 bars = 8h stale → close)

Reports aggregate portfolio metrics + per-coin Sharpe breakdown.

Usage:
  python scripts/backtest_unified_20coin.py --model models/xgb_unified_20coin_15m.pkl
  python scripts/backtest_unified_20coin.py --model models/xgb_unified_20coin_15m.pkl --sweep
  python scripts/backtest_unified_20coin.py --model models/xgb_unified_20coin_15m.pkl --threshold 0.55

Spec: docs/superpowers/specs/2026-03-21-unified-20coin-xgboost-design.md
"""

import argparse
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import quantstats as qs

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import (
    compute_features,
    compute_market_context_features,
    compute_coin_identity_features,
)

HORIZON_15M = 16
PERIODS_15M = 35_040
TRAIN_CUTOFF = "2024-01-01"

COIN_UNIVERSE = [
    "BTC", "ETH", "BNB",
    "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC",
    "UNI", "NEAR", "SUI", "APT", "PEPE", "ARB", "SHIB", "FIL", "HBAR",
]

LIQUIDITY_TIERS = {
    "BTC": 1, "ETH": 1, "BNB": 1,
    "SOL": 2, "XRP": 2, "DOGE": 2, "ADA": 2, "AVAX": 2,
    "LINK": 2, "DOT": 2, "LTC": 2,
    "UNI": 3, "NEAR": 3, "SUI": 3, "APT": 3, "PEPE": 3,
    "ARB": 3, "SHIB": 3, "FIL": 3, "HBAR": 3,
}

FEATURE_COLS = [
    "atr_proxy", "RSI_14", "RSI_7",
    "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    "bb_width", "bb_pos", "volume_ratio", "candle_body",
    "btc_return_4h", "btc_return_1d", "eth_return_4h", "eth_return_1d",
    "btc_corr_30d", "relative_vol", "vol_rank", "liquidity_tier",
]


@dataclass
class Position:
    coin: str
    units: float
    entry_price: float        # effective (incl. fee)
    trail_stop: float
    entry_bar_idx: int
    entry_atr: float


def _cb_multiplier(drawdown: float) -> float:
    if drawdown >= 0.30: return 0.0
    if drawdown >= 0.20: return 0.25
    if drawdown >= 0.10: return 0.50
    return 1.0


def prepare_coin_data(data_dir: str, start: str = None) -> dict:
    """Load and prepare feature matrices for all 20 coins."""
    data_path = Path(data_dir)

    btc_raw = pd.read_parquet(data_path / "BTCUSDT_15m.parquet")
    eth_raw = pd.read_parquet(data_path / "ETHUSDT_15m.parquet")
    for df in (btc_raw, eth_raw):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    # Pre-compute all ATR proxies for vol_rank
    all_atr_proxies = {}
    for coin in COIN_UNIVERSE:
        coin_df = pd.read_parquet(data_path / f"{coin}USDT_15m.parquet")
        coin_df.index = pd.to_datetime(coin_df.index)
        coin_df.columns = coin_df.columns.str.lower()
        lr = np.log(coin_df["close"] / coin_df["close"].shift(1))
        all_atr_proxies[coin] = lr.rolling(14).std() * coin_df["close"] * 1.25

    coin_data = {}
    for coin in COIN_UNIVERSE:
        coin_df = pd.read_parquet(data_path / f"{coin}USDT_15m.parquet")
        coin_df.index = pd.to_datetime(coin_df.index)
        coin_df.columns = coin_df.columns.str.lower()

        feat = compute_features(coin_df)
        feat = compute_market_context_features(feat, btc_raw, eth_raw)
        feat = compute_coin_identity_features(
            feat, btc_raw,
            liquidity_tier=LIQUIDITY_TIERS[coin],
            all_atr_proxies={**all_atr_proxies, "self": all_atr_proxies[coin]},
        )
        feat = feat.dropna(subset=FEATURE_COLS)

        # Date filter
        if start:
            feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]

        # Pre-compute EMA_96 and EMA_240 for regime detection
        ema_96 = feat["close"].ewm(span=96, adjust=False).mean()
        ema_240 = feat["close"].ewm(span=240, adjust=False).mean()

        coin_data[coin] = {
            "feat": feat,
            "closes": feat["close"].values,
            "atrs": feat["atr_proxy"].values,
            "timestamps": feat.index,
            "ema_96": ema_96.values,
            "ema_240": ema_240.values,
        }
        print(f"  {coin:>5}: {len(feat):,} bars")

    return coin_data


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def batch_predict_all(model, coin_data: dict) -> dict:
    """Run model predictions for all coins."""
    feature_cols = list(model.feature_names_in_)
    probas = {}
    for coin, data in coin_data.items():
        feat = data["feat"]
        missing = [c for c in feature_cols if c not in feat.columns]
        if missing:
            print(f"  WARNING: {coin} missing features: {missing[:3]}")
            continue
        probas[coin] = model.predict_proba(feat[feature_cols])[:, 1]
    return probas


def compute_return_correlations(coin_data: dict, window: int = 2880) -> dict:
    """Pre-compute pairwise rolling correlations for correlation-aware sizing."""
    # Build return series for each coin
    returns = {}
    for coin, data in coin_data.items():
        c = data["closes"]
        ret = np.zeros(len(c))
        ret[1:] = np.diff(np.log(c))
        returns[coin] = pd.Series(ret, index=data["timestamps"])

    # Compute pairwise rolling correlations
    corr_cache = {}
    coins = list(returns.keys())
    for i, c1 in enumerate(coins):
        for c2 in coins[i+1:]:
            # Align on common timestamps
            aligned = pd.concat([returns[c1], returns[c2]], axis=1, join="inner")
            if len(aligned) < window:
                continue
            aligned.columns = ["a", "b"]
            rolling_corr = aligned["a"].rolling(window).corr(aligned["b"])
            corr_cache[(c1, c2)] = rolling_corr
            corr_cache[(c2, c1)] = rolling_corr

    return corr_cache


def run_backtest(
    coin_data: dict,
    probas: dict,
    threshold: float,
    corr_cache: dict,
    initial_capital: float = 1_000_000.0,
    risk_per_trade_pct: float = 0.02,
    hard_stop_pct: float = 0.05,
    atr_stop_multiplier: float = 10.0,
    max_positions: int = 5,
    max_single_position_pct: float = 0.40,
    expected_win_loss_ratio: float = 1.5,
    fee_bps: int = 10,
    exit_threshold: float = 0.10,
    corr_threshold: float = 0.80,
    stale_bars: int = 32,
) -> tuple:
    """
    Multi-coin portfolio backtest with all Sortino improvements.

    Returns:
        (returns_series, portfolio_series, closed_trades, gate_stats, per_coin_trades)
    """
    fee_rate = fee_bps / 10_000.0

    # Build unified timeline (union of all coin timestamps)
    all_ts = set()
    for data in coin_data.values():
        all_ts.update(data["timestamps"].tolist())
    timeline = sorted(all_ts)
    n = len(timeline)
    ts_to_idx = {ts: i for i, ts in enumerate(timeline)}

    # Build per-coin index mapping: for each timeline bar, which coin bar is it?
    coin_bar_map = {}
    for coin, data in coin_data.items():
        mapping = {}
        coin_ts = data["timestamps"]
        for ci, cts in enumerate(coin_ts):
            if cts in ts_to_idx:
                mapping[ts_to_idx[cts]] = ci
        coin_bar_map[coin] = mapping

    # Portfolio state
    free_balance = initial_capital
    portfolio_hwm = initial_capital
    positions: dict[str, Position] = {}  # coin -> Position

    portfolio_values = np.zeros(n)
    portfolio_values[0] = initial_capital
    returns = np.zeros(n)
    closed_trades = []
    per_coin_trades: dict[str, list] = {coin: [] for coin in COIN_UNIVERSE}
    gate_stats = {"kelly_blocked": 0, "cb_halted": 0, "cb_reduced": 0,
                  "regime_blocked": 0, "corr_blocked": 0, "stale_exits": 0}

    for i in range(n):
        ts = timeline[i]

        # --- Update trailing stops + check exits for all open positions ---
        coins_to_close = []
        for coin, pos in positions.items():
            ci = coin_bar_map[coin].get(i)
            if ci is None:
                continue

            data = coin_data[coin]
            c = data["closes"][ci]
            atr = data["atrs"][ci]

            # Update trailing stop
            if not np.isnan(atr) and atr > 0:
                new_stop = c - atr_stop_multiplier * atr
                pos.trail_stop = max(pos.trail_stop, new_stop)

            # Check stop
            stop_hit = c <= pos.trail_stop

            # Check signal exit
            p = probas[coin][ci] if coin in probas and ci < len(probas[coin]) else 0.5
            signal_exit = p <= exit_threshold

            # Check stale exit (time-based)
            bars_held = ci - pos.entry_bar_idx if pos.entry_bar_idx >= 0 else 0
            stale_exit = False
            if bars_held >= stale_bars and pos.entry_price > 0:
                price_move = abs(c - pos.entry_price) / pos.entry_price
                if pos.entry_atr > 0 and price_move < 0.5 * pos.entry_atr / pos.entry_price:
                    stale_exit = True

            if stop_hit or signal_exit or stale_exit:
                proceeds = pos.units * c * (1.0 - fee_rate)
                net_exit = c * (1.0 - fee_rate)
                pnl_pct = (net_exit - pos.entry_price) / pos.entry_price

                reason = "stop" if stop_hit else ("stale" if stale_exit else "signal")
                if stale_exit:
                    gate_stats["stale_exits"] += 1

                trade = {
                    "coin": coin, "entry_bar": pos.entry_bar_idx,
                    "exit_bar": ci, "entry_ts": None, "exit_ts": ts,
                    "entry_price": pos.entry_price, "exit_price": net_exit,
                    "pnl_pct": pnl_pct, "exit_reason": reason,
                }
                closed_trades.append(trade)
                per_coin_trades[coin].append(trade)
                free_balance += proceeds
                coins_to_close.append(coin)

        for coin in coins_to_close:
            del positions[coin]

        # --- Mark to market ---
        position_value = sum(
            pos.units * coin_data[coin]["closes"][coin_bar_map[coin].get(i, 0)]
            for coin, pos in positions.items()
            if coin_bar_map[coin].get(i) is not None
        )
        total_portfolio = free_balance + position_value
        portfolio_hwm = max(portfolio_hwm, total_portfolio)
        drawdown = (portfolio_hwm - total_portfolio) / portfolio_hwm if portfolio_hwm > 0 else 0.0
        cb_mult = _cb_multiplier(drawdown)

        # --- Collect BUY signals from all coins ---
        buy_candidates = []
        for coin in COIN_UNIVERSE:
            if coin in positions:
                continue  # already holding
            ci = coin_bar_map[coin].get(i)
            if ci is None:
                continue

            data = coin_data[coin]
            if coin not in probas or ci >= len(probas[coin]):
                continue
            p = probas[coin][ci]

            if p >= threshold:
                buy_candidates.append((coin, p, ci))

        # Sort by confidence descending (signal ranking)
        buy_candidates.sort(key=lambda x: x[1], reverse=True)

        # --- Execute BUY signals (up to max_positions) ---
        for coin, p, ci in buy_candidates:
            if len(positions) >= max_positions:
                break

            data = coin_data[coin]
            c = data["closes"][ci]
            atr = data["atrs"][ci]

            # Per-coin regime check
            if ci < 240:
                regime_mult = 0.5
            else:
                ema_fast = data["ema_96"][ci]
                ema_slow = data["ema_240"][ci]
                spread = abs(ema_fast - ema_slow) / (ema_slow + 1e-10)
                if ema_fast > ema_slow:
                    regime_mult = 1.0
                elif spread < 0.001:
                    regime_mult = 0.5
                else:
                    regime_mult = 0.0

            if regime_mult == 0.0:
                gate_stats["regime_blocked"] += 1
                continue

            # Circuit breaker
            if cb_mult == 0.0:
                gate_stats["cb_halted"] += 1
                continue
            if cb_mult < 1.0:
                gate_stats["cb_reduced"] += 1

            # Kelly criterion
            kelly = (p * expected_win_loss_ratio - (1.0 - p)) / expected_win_loss_ratio
            if kelly <= 0:
                gate_stats["kelly_blocked"] += 1
                continue

            # Correlation check with existing positions
            corr_penalty = 1.0
            for held_coin in positions:
                key = (coin, held_coin)
                if key in corr_cache:
                    corr_val = corr_cache[key].get(ts, np.nan)
                    if not np.isnan(corr_val) and abs(corr_val) > corr_threshold:
                        corr_penalty = 0.5
                        gate_stats["corr_blocked"] += 1
                        break

            # Position sizing
            hard_stop_price = c * (1.0 - hard_stop_pct)
            if not np.isnan(atr) and atr > 0:
                atr_stop_price = c - atr_stop_multiplier * atr
                initial_stop = max(hard_stop_price, atr_stop_price)
            else:
                initial_stop = hard_stop_price

            stop_distance = c - initial_stop
            stop_distance = min(stop_distance, c * hard_stop_pct)
            if stop_distance <= 0:
                stop_distance = c * hard_stop_pct

            eff_mult = cb_mult * regime_mult * corr_penalty
            risk_usd = total_portfolio * risk_per_trade_pct * p * eff_mult
            quantity = risk_usd / stop_distance
            target_usd = quantity * c

            usable = free_balance * 0.95
            target_usd = min(target_usd, total_portfolio * max_single_position_pct, usable)

            if target_usd >= 10.0:
                units = target_usd / c
                entry_fee = target_usd * fee_rate
                free_balance -= (target_usd + entry_fee)
                entry_eff = c * (1.0 + fee_rate)

                positions[coin] = Position(
                    coin=coin, units=units, entry_price=entry_eff,
                    trail_stop=initial_stop, entry_bar_idx=ci,
                    entry_atr=atr if not np.isnan(atr) else c * 0.02,
                )

                # Recompute portfolio
                position_value = sum(
                    pos.units * coin_data[cn]["closes"][coin_bar_map[cn].get(i, 0)]
                    for cn, pos in positions.items()
                    if coin_bar_map[cn].get(i) is not None
                )
                total_portfolio = free_balance + position_value

        # Record portfolio value
        position_value = sum(
            pos.units * coin_data[coin]["closes"][coin_bar_map[coin].get(i, 0)]
            for coin, pos in positions.items()
            if coin_bar_map[coin].get(i) is not None
        )
        portfolio_values[i] = free_balance + position_value
        if i > 0:
            returns[i] = portfolio_values[i] / portfolio_values[i - 1] - 1.0

    returns_series = pd.Series(returns, index=timeline)
    portfolio_series = pd.Series(portfolio_values, index=timeline)
    return returns_series, portfolio_series, closed_trades, gate_stats, per_coin_trades


def print_report(
    returns: pd.Series,
    portfolio: pd.Series,
    closed_trades: list,
    gate_stats: dict,
    per_coin_trades: dict,
    initial_capital: float,
    threshold: float,
):
    n_trades = len(closed_trades)
    final = portfolio.iloc[-1]
    total_ret = (final - initial_capital) / initial_capital

    print("=" * 70)
    print(f"  UNIFIED 20-COIN BACKTEST RESULTS (threshold={threshold})")
    print("=" * 70)
    print(f"  Capital: ${initial_capital:,.0f} -> ${final:,.0f} ({total_ret:+.2%})")
    print(f"  Sharpe:  {float(qs.stats.sharpe(returns, periods=PERIODS_15M)):.3f}")
    print(f"  Sortino: {float(qs.stats.sortino(returns, periods=PERIODS_15M)):.3f}")
    print(f"  Calmar:  {float(qs.stats.calmar(returns)):.3f}")
    print(f"  Max DD:  {float(qs.stats.max_drawdown(returns))*100:.2f}%")
    print(f"  Trades:  {n_trades}")

    if n_trades > 0:
        winners = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
        print(f"  Win rate: {winners/n_trades:.1%}")
        stop_exits = sum(1 for t in closed_trades if t["exit_reason"] == "stop")
        stale_exits = sum(1 for t in closed_trades if t["exit_reason"] == "stale")
        signal_exits = n_trades - stop_exits - stale_exits
        print(f"  Exits: {signal_exits} signal / {stop_exits} stop / {stale_exits} stale")

    print(f"\n  Gates: kelly={gate_stats['kelly_blocked']} "
          f"cb_halt={gate_stats['cb_halted']} "
          f"cb_reduce={gate_stats['cb_reduced']} "
          f"regime={gate_stats['regime_blocked']} "
          f"corr={gate_stats['corr_blocked']}")

    # Per-coin Sharpe breakdown
    print(f"\n  {'Coin':>6}  {'Trades':>7}  {'Win%':>6}  {'PnL%':>8}")
    print("  " + "-" * 35)
    positive_sharpe_coins = 0
    for coin in COIN_UNIVERSE:
        trades = per_coin_trades.get(coin, [])
        nt = len(trades)
        if nt == 0:
            print(f"  {coin:>6}  {0:>7}  {'N/A':>6}  {'N/A':>8}")
            continue
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        avg_pnl = sum(t["pnl_pct"] for t in trades) / nt
        print(f"  {coin:>6}  {nt:>7}  {wins/nt:>5.1%}  {avg_pnl:>+7.2%}")
        if avg_pnl > 0:
            positive_sharpe_coins += 1

    print(f"\n  Coins with positive avg PnL: {positive_sharpe_coins}/20")

    # Active trading days
    trade_dates = set()
    for t in closed_trades:
        if t.get("exit_ts"):
            trade_dates.add(pd.Timestamp(t["exit_ts"]).date())
    total_days = (portfolio.index[-1] - portfolio.index[0]).days
    print(f"  Active trading days: {len(trade_dates)} / {total_days}")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-coin portfolio backtest")
    p.add_argument("--model", required=True, help="Path to unified .pkl model")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--exit-threshold", type=float, default=0.10)
    p.add_argument("--atr-mult", type=float, default=10.0)
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--fee-bps", type=int, default=10)
    p.add_argument("--sweep", action="store_true", help="Sweep thresholds 0.45-0.75")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading coin data...")
    coin_data = prepare_coin_data(args.data_dir, args.start)

    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model)
    print(f"  Model expects {len(model.feature_names_in_)} features")

    print("Running predictions...")
    probas = batch_predict_all(model, coin_data)

    print("Computing return correlations...")
    corr_cache = compute_return_correlations(coin_data)

    if args.sweep:
        thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        print(f"\nSweeping thresholds: {thresholds}")
        print(f"{'Thresh':>8}  {'Sharpe':>8}  {'Sortino':>8}  {'Trades':>7}  "
              f"{'Return%':>9}  {'MaxDD%':>8}")
        print("-" * 60)

        best = None
        for t in thresholds:
            ret, port, trades, gates, per_coin = run_backtest(
                coin_data, probas, t, corr_cache,
                initial_capital=args.capital,
                atr_stop_multiplier=args.atr_mult,
                max_positions=args.max_positions,
                fee_bps=args.fee_bps,
                exit_threshold=args.exit_threshold,
            )
            sharpe = float(qs.stats.sharpe(ret, periods=PERIODS_15M))
            sortino = float(qs.stats.sortino(ret, periods=PERIODS_15M))
            total_ret = (port.iloc[-1] - args.capital) / args.capital * 100
            max_dd = float(qs.stats.max_drawdown(ret)) * 100
            n_trades = len(trades)
            marker = ""
            if best is None or sortino > best[1]:
                best = (t, sortino)
                marker = " <-- best"
            print(f"  {t:>6.2f}  {sharpe:>8.3f}  {sortino:>8.3f}  {n_trades:>7}  "
                  f"{total_ret:>+8.2f}%  {max_dd:>7.2f}%{marker}")

        print(f"\nBest threshold (Sortino): {best[0]} (Sortino={best[1]:.3f})")
        return

    print(f"\nRunning backtest (threshold={args.threshold})...")
    ret, port, trades, gates, per_coin = run_backtest(
        coin_data, probas, args.threshold, corr_cache,
        initial_capital=args.capital,
        atr_stop_multiplier=args.atr_mult,
        max_positions=args.max_positions,
        fee_bps=args.fee_bps,
        exit_threshold=args.exit_threshold,
    )

    print_report(ret, port, trades, gates, per_coin, args.capital, args.threshold)


if __name__ == "__main__":
    main()

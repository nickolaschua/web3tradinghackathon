#!/usr/bin/env python3
"""
Combined multi-asset backtest: BTC XGBoost + SOL XGBoost + MR + Relaxed MR + Regime.

Simulates the full live-bot strategy cascade with:
  - Two independent position slots (BTC, SOL)
  - Signal cascade per slot: XGBoost → MR → Relaxed MR
  - Regime detection (EMA20/EMA50 on daily BTC) with configurable multipliers
  - Full risk management (ATR stops, circuit breakers)
  - Proper features per asset (BTC indicators from BTC data, SOL from SOL)

Usage:
  python scripts/backtest_combined.py
  python scripts/backtest_combined.py --bear-mult 0.35 --sol-threshold 0.70 --exit-threshold 0.08
  python scripts/backtest_combined.py --bear-mult 0.0   # simulate current broken config
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_features, compute_btc_context_features
from bot.strategy.mean_reversion import MeanReversionStrategy
from bot.strategy.relaxed_mean_reversion import RelaxedMeanReversionStrategy

PERIODS_15M = 35_040
TRAIN_CUTOFF = "2024-01-01"


# -- Feature pipelines --------------------------------------------------------

def prepare_btc_features(btc_path, eth_path, sol_path, start=None, end=None):
    """BTC feature matrix for BTC XGBoost model."""
    btc = pd.read_parquet(btc_path)
    eth = pd.read_parquet(eth_path)
    sol = pd.read_parquet(sol_path)
    for df in (btc, eth, sol):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    feat = compute_features(btc)

    # Cross-asset return lags (4H=16 bars, 1D=96 bars)
    for asset, df in [("btc", btc), ("eth", eth), ("sol", sol)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    feat = compute_btc_context_features(feat, eth, sol, window=2880)
    feat = feat.dropna()

    if start:
        feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]

    return feat, btc, eth, sol


def prepare_sol_features(btc_df, eth_df, sol_df, btc_feat_index, start=None, end=None):
    """SOL feature matrix for SOL XGBoost model — SOL's own indicators + BTC/ETH context."""
    feat = compute_features(sol_df)

    # Cross-asset return lags from BTC and ETH
    for prefix, df in [("btc", btc_df), ("eth", eth_df)]:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{prefix}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{prefix}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    # SOL-BTC correlation and beta
    btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1)).reindex(feat.index)
    sol_ret = np.log(feat["close"] / feat["close"].shift(1))
    feat["sol_btc_corr"] = sol_ret.rolling(2880).corr(btc_ret).shift(1)
    cov = sol_ret.rolling(2880).cov(btc_ret)
    var = btc_ret.rolling(2880).var()
    feat["sol_btc_beta"] = (cov / (var + 1e-10)).shift(1)

    feat = feat.dropna()

    if start:
        feat = feat[feat.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        feat = feat[feat.index <= pd.Timestamp(end, tz="UTC")]

    # Align to BTC feature index (shared timeline)
    common_idx = feat.index.intersection(btc_feat_index)
    feat = feat.loc[common_idx]

    return feat


# -- Regime detection (vectorized) --------------------------------------------

def compute_regime_series(btc_df, bull_mult, sideways_mult, bear_mult, confirmation=2):
    """
    Vectorized regime detection: EMA20/EMA50 on daily-resampled BTC.
    Returns a Series of multipliers aligned to 15M timestamps.
    """
    btc = btc_df.copy()
    btc.columns = btc.columns.str.lower()

    # Resample to daily
    daily = btc["close"].resample("1D").last().dropna()
    ema20 = daily.ewm(span=20, adjust=False).mean()
    ema50 = daily.ewm(span=50, adjust=False).mean()

    # Raw regime signal per day
    spread_pct = (ema20 - ema50).abs() / ema50
    raw_regime = np.where(
        spread_pct < 0.001, sideways_mult,
        np.where(ema20 > ema50, bull_mult, bear_mult)
    )

    # Apply hysteresis (confirmation bars)
    regime_vals = np.full_like(raw_regime, sideways_mult)
    current = sideways_mult
    pending = current
    pending_count = 0
    for i in range(len(raw_regime)):
        r = raw_regime[i]
        if r != current:
            if r == pending:
                pending_count += 1
            else:
                pending = r
                pending_count = 1
            if pending_count >= confirmation:
                current = r
                pending = current
                pending_count = 0
        else:
            pending = current
            pending_count = 0
        regime_vals[i] = current

    regime_daily = pd.Series(regime_vals, index=daily.index)

    # Forward-fill to 15M resolution
    regime_15m = regime_daily.reindex(btc_df.index, method="ffill")
    return regime_15m


# -- Circuit breaker -----------------------------------------------------------

def cb_multiplier(drawdown, halt=0.30, heavy=0.20, light=0.10):
    if drawdown >= halt:
        return 0.0
    elif drawdown >= heavy:
        return 0.25
    elif drawdown >= light:
        return 0.5
    return 1.0


# -- Model loading -------------------------------------------------------------

def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def batch_predict(model, feat_df):
    cols = list(model.feature_names_in_)
    missing = [c for c in cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = feat_df[cols].values
    valid = ~np.isnan(X).any(axis=1)
    probas = np.full(len(X), np.nan)
    if valid.any():
        probas[valid] = model.predict_proba(X[valid])[:, 1]
    return probas


# -- MR signal vectorization ---------------------------------------------------

def vectorize_mr_signals(feat_df):
    """Pre-compute MR buy/sell signals for all bars. Returns (buy_mask, sell_mask, sizes, confs)."""
    ema20 = feat_df["EMA_20"].values
    ema50 = feat_df["EMA_50"].values
    rsi = feat_df["RSI_14"].values
    bb = feat_df["bb_pos"].values
    macd_h = feat_df["MACDh_12_26_9"].values

    n = len(feat_df)
    uptrend = ema20 > ema50

    # Standard entry: RSI<30 + bb<0.15 + MACD_h>0 in uptrend
    standard = uptrend & (rsi < 30) & (bb < 0.15) & (macd_h > 0)
    # Extreme: RSI<25 in uptrend (not already standard)
    extreme = uptrend & (rsi < 25) & ~standard

    buy = standard | extreme
    sizes = np.where(standard, 0.35, np.where(extreme, 0.25, 0.0))
    confs = np.where(standard, 0.60, np.where(extreme, 0.55, 0.0))

    sell = (rsi > 55) | (bb > 0.6)

    return buy, sell, sizes, confs


def vectorize_relaxed_mr_signals(feat_df):
    """Pre-compute relaxed MR buy/sell signals. Returns (buy_mask, sell_mask, sizes, confs)."""
    ema20 = feat_df["EMA_20"].values
    ema50 = feat_df["EMA_50"].values
    rsi = feat_df["RSI_14"].values
    bb = feat_df["bb_pos"].values

    regime_mult = np.where(ema20 > ema50, 1.0, 0.25)

    # Standard: RSI<35 + bb<0.25
    standard = (rsi < 35) & (bb < 0.25)
    # Deep oversold: RSI<28 (not already standard)
    deep = (rsi < 28) & ~standard

    buy = standard | deep
    sizes = np.where(buy, 0.01 * regime_mult, 0.0)
    confs = np.full(len(feat_df), 0.50)

    sell = (rsi > 50) | (bb > 0.55)

    return buy, sell, sizes, confs


# -- Combined backtest ---------------------------------------------------------

def run_combined_backtest(
    btc_feat, sol_feat,
    btc_probas, sol_probas,
    regime_mults,
    btc_threshold=0.65, sol_threshold=0.70,
    btc_exit_threshold=0.08, sol_exit_threshold=0.08,
    initial_capital=1_000_000,
    risk_per_trade=0.02,
    hard_stop_pct=0.05,
    atr_mult=10.0,
    max_single_pct=0.40,
    expected_wl=1.5,
    fee_bps=10,
    cb_halt=0.30, cb_heavy=0.20, cb_light=0.10,
):
    """
    Bar-by-bar combined backtest with two independent position slots.
    """
    fee = fee_bps / 10_000.0

    # Align timelines
    common_idx = btc_feat.index.intersection(sol_feat.index)
    btc_feat = btc_feat.loc[common_idx]
    sol_feat = sol_feat.loc[common_idx]
    btc_probas_aligned = btc_probas[np.isin(btc_feat.index, common_idx) | True][:len(common_idx)]
    sol_probas_aligned = sol_probas[:len(common_idx)]
    regime_aligned = regime_mults.reindex(common_idx, method="ffill").fillna(0.5).values

    # Pre-vectorize MR signals
    btc_mr_buy, btc_mr_sell, btc_mr_sizes, btc_mr_confs = vectorize_mr_signals(btc_feat)
    sol_mr_buy, sol_mr_sell, sol_mr_sizes, sol_mr_confs = vectorize_mr_signals(sol_feat)

    btc_rmr_buy, btc_rmr_sell, btc_rmr_sizes, btc_rmr_confs = vectorize_relaxed_mr_signals(btc_feat)
    sol_rmr_buy, sol_rmr_sell, sol_rmr_sizes, sol_rmr_confs = vectorize_relaxed_mr_signals(sol_feat)

    # Extract arrays
    btc_closes = btc_feat["close"].values
    sol_closes = sol_feat["close"].values
    btc_atrs = btc_feat["atr_proxy"].values
    sol_atrs = sol_feat["atr_proxy"].values
    timestamps = common_idx
    n = len(common_idx)

    # Portfolio state
    free_balance = initial_capital
    portfolio_hwm = initial_capital

    # Per-asset position state
    slots = {
        "BTC": {"units": 0.0, "entry_price": 0.0, "trail_stop": 0.0, "source": None},
        "SOL": {"units": 0.0, "entry_price": 0.0, "trail_stop": 0.0, "source": None},
    }

    portfolio_values = np.zeros(n)
    portfolio_values[0] = initial_capital
    returns = np.zeros(n)
    closed_trades = []
    active_days = set()
    gate_stats = {"kelly_blocked": 0, "cb_halted": 0, "regime_blocked": 0, "total_signals": 0}

    for i in range(n):
        btc_c, sol_c = btc_closes[i], sol_closes[i]
        btc_atr, sol_atr = btc_atrs[i], sol_atrs[i]
        btc_p, sol_p = btc_probas_aligned[i], sol_probas_aligned[i]
        regime_mult = regime_aligned[i]
        ts = timestamps[i]

        # -- Mark to market --
        btc_val = slots["BTC"]["units"] * btc_c
        sol_val = slots["SOL"]["units"] * sol_c
        total = free_balance + btc_val + sol_val
        portfolio_hwm = max(portfolio_hwm, total)
        dd = (portfolio_hwm - total) / portfolio_hwm if portfolio_hwm > 0 else 0.0
        cb_mult = cb_multiplier(dd, cb_halt, cb_heavy, cb_light)

        # -- Process each asset --
        for asset, c, atr, p, threshold, exit_thresh in [
            ("BTC", btc_c, btc_atr, btc_p, btc_threshold, btc_exit_threshold),
            ("SOL", sol_c, sol_atr, sol_p, sol_threshold, sol_exit_threshold),
        ]:
            slot = slots[asset]
            idx = 0 if asset == "BTC" else 1

            # -- Trailing stop update --
            if slot["units"] > 0 and not np.isnan(atr) and atr > 0:
                new_stop = c - atr_mult * atr
                slot["trail_stop"] = max(slot["trail_stop"], new_stop)

            # -- Check exits --
            just_exited = False
            if slot["units"] > 0:
                stop_hit = c <= slot["trail_stop"]

                # Source-specific exit
                if slot["source"] == "xgb":
                    sell_signal = (not np.isnan(p)) and p <= exit_thresh
                elif slot["source"] == "mr":
                    if asset == "BTC":
                        sell_signal = btc_mr_sell[i]
                    else:
                        sell_signal = sol_mr_sell[i]
                elif slot["source"] == "relaxed_mr":
                    if asset == "BTC":
                        sell_signal = btc_rmr_sell[i]
                    else:
                        sell_signal = sol_rmr_sell[i]
                else:
                    sell_signal = False

                if stop_hit or sell_signal:
                    proceeds = slot["units"] * c * (1.0 - fee)
                    net_exit = c * (1.0 - fee)
                    pnl_pct = (net_exit - slot["entry_price"]) / slot["entry_price"]

                    closed_trades.append({
                        "asset": asset,
                        "entry_bar": None,
                        "exit_bar": ts,
                        "entry_price": slot["entry_price"],
                        "exit_price": net_exit,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "stop" if stop_hit else "signal",
                        "source": slot["source"],
                    })

                    free_balance += proceeds
                    slot["units"] = 0.0
                    slot["source"] = None
                    just_exited = True
                    active_days.add(ts.date() if hasattr(ts, "date") else str(ts)[:10])

            # -- Check entries --
            if slot["units"] == 0 and not just_exited:
                # Determine signal source (cascade)
                signal_source = None
                signal_size = 0.0
                signal_conf = 0.0

                # 1. XGBoost
                if not np.isnan(p) and p >= threshold:
                    signal_source = "xgb"
                    signal_conf = p
                # 2. MR fallback
                elif asset == "BTC" and btc_mr_buy[i]:
                    signal_source = "mr"
                    signal_size = btc_mr_sizes[i]
                    signal_conf = btc_mr_confs[i]
                elif asset == "SOL" and sol_mr_buy[i]:
                    signal_source = "mr"
                    signal_size = sol_mr_sizes[i]
                    signal_conf = sol_mr_confs[i]
                # 3. Relaxed MR fallback
                elif asset == "BTC" and btc_rmr_buy[i]:
                    signal_source = "relaxed_mr"
                    signal_size = btc_rmr_sizes[i]
                    signal_conf = btc_rmr_confs[i]
                elif asset == "SOL" and sol_rmr_buy[i]:
                    signal_source = "relaxed_mr"
                    signal_size = sol_rmr_sizes[i]
                    signal_conf = sol_rmr_confs[i]

                if signal_source is None:
                    continue

                gate_stats["total_signals"] += 1

                # Regime gate
                if regime_mult < 0.10:
                    gate_stats["regime_blocked"] += 1
                    continue

                # CB gate
                if cb_mult == 0.0:
                    gate_stats["cb_halted"] += 1
                    continue

                # Kelly gate (XGBoost only)
                if signal_source == "xgb":
                    kelly = (signal_conf * expected_wl - (1.0 - signal_conf)) / expected_wl
                    if kelly <= 0:
                        gate_stats["kelly_blocked"] += 1
                        continue

                # Compute stop
                hard_stop = c * (1.0 - hard_stop_pct)
                if not np.isnan(atr) and atr > 0:
                    atr_stop = c - atr_mult * atr
                    initial_stop = max(hard_stop, atr_stop)
                else:
                    initial_stop = hard_stop

                stop_dist = min(c - initial_stop, c * hard_stop_pct)
                if stop_dist <= 0:
                    stop_dist = c * hard_stop_pct

                # Size position
                effective_mult = regime_mult * cb_mult

                if signal_source == "xgb":
                    risk_usd = total * risk_per_trade * signal_conf * effective_mult
                    qty = risk_usd / stop_dist
                    target_usd = qty * c
                elif signal_source in ("mr", "relaxed_mr"):
                    target_usd = total * signal_size * effective_mult

                usable = free_balance * 0.95
                target_usd = min(target_usd, total * max_single_pct, usable)

                if target_usd >= 10.0:
                    position_units = target_usd / c
                    entry_fee = target_usd * fee
                    free_balance -= (target_usd + entry_fee)
                    slot["units"] = position_units
                    slot["entry_price"] = c * (1.0 + fee)
                    slot["trail_stop"] = initial_stop
                    slot["source"] = signal_source
                    active_days.add(ts.date() if hasattr(ts, "date") else str(ts)[:10])

                    # Refresh total
                    btc_val = slots["BTC"]["units"] * btc_c
                    sol_val = slots["SOL"]["units"] * sol_c
                    total = free_balance + btc_val + sol_val

        # Record
        btc_val = slots["BTC"]["units"] * btc_c
        sol_val = slots["SOL"]["units"] * sol_c
        portfolio_values[i] = free_balance + btc_val + sol_val
        if i > 0:
            returns[i] = portfolio_values[i] / portfolio_values[i - 1] - 1.0

    returns_series = pd.Series(returns, index=timestamps)
    portfolio_series = pd.Series(portfolio_values, index=timestamps)
    return returns_series, portfolio_series, closed_trades, gate_stats, active_days


# -- Stats ---------------------------------------------------------------------

def compute_stats(returns, portfolio, trades, gate_stats, active_days, label=""):
    total_ret = portfolio.iloc[-1] / portfolio.iloc[0] - 1
    years = len(returns) / PERIODS_15M
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    ann_vol = returns.std() * np.sqrt(PERIODS_15M)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(PERIODS_15M) if returns.std() > 0 else 0

    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std()) * np.sqrt(PERIODS_15M) if len(downside) > 0 and downside.std() > 0 else 0

    running_max = portfolio.cummax()
    drawdowns = (portfolio - running_max) / running_max
    max_dd = drawdowns.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade breakdown
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_pnl = np.mean([t["pnl_pct"] for t in trades]) * 100 if trades else 0

    # By source
    sources = {}
    for t in trades:
        s = t["source"]
        if s not in sources:
            sources[s] = {"count": 0, "wins": 0, "pnl_sum": 0}
        sources[s]["count"] += 1
        sources[s]["pnl_sum"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            sources[s]["wins"] += 1

    # By asset
    assets = {}
    for t in trades:
        a = t["asset"]
        if a not in assets:
            assets[a] = {"count": 0, "wins": 0, "pnl_sum": 0}
        assets[a]["count"] += 1
        assets[a]["pnl_sum"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            assets[a]["wins"] += 1

    stop_exits = sum(1 for t in trades if t["exit_reason"] == "stop")
    signal_exits = sum(1 for t in trades if t["exit_reason"] == "signal")

    print(f"\n{'=' * 70}")
    print(f"  COMBINED BACKTEST — {label}")
    print(f"{'=' * 70}")
    print(f"  Initial capital   : ${portfolio.iloc[0]:,.0f}")
    print(f"  Final portfolio   : ${portfolio.iloc[-1]:,.2f}")
    print(f"  Total return      : {total_ret:+.2%}")
    print(f"  CAGR              : {cagr:+.2%}")
    print(f"  Sharpe (15M ann.) : {sharpe:.3f}")
    print(f"  Sortino (15M ann.): {sortino:.3f}")
    print(f"  Calmar            : {calmar:.3f}")
    print(f"  Max drawdown      : {max_dd:.2%}")
    print(f"  Volatility (ann.) : {ann_vol:.2%}")
    print(f"{'=' * 70}")
    print(f"  # Trades          : {n_trades}")
    print(f"    Stop exits       : {stop_exits}")
    print(f"    Signal exits     : {signal_exits}")
    print(f"  Win rate          : {win_rate:.1f}%")
    print(f"  Avg trade PnL     : {avg_pnl:+.2f}%")
    if trades:
        print(f"  Best trade        : {max(t['pnl_pct'] for t in trades):+.2%}")
        print(f"  Worst trade       : {min(t['pnl_pct'] for t in trades):+.2%}")
    print(f"{'=' * 70}")
    print(f"  By source:")
    for src, stats in sorted(sources.items()):
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        avg = stats["pnl_sum"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"    {src:12s}: {stats['count']:3d} trades | {wr:.0f}% win | {avg:+.2f}% avg")
    print(f"  By asset:")
    for asset, stats in sorted(assets.items()):
        wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        avg = stats["pnl_sum"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"    {asset:12s}: {stats['count']:3d} trades | {wr:.0f}% win | {avg:+.2f}% avg")
    print(f"{'=' * 70}")
    print(f"  Active trading days: {len(active_days)}")
    total_days = (portfolio.index[-1] - portfolio.index[0]).days
    print(f"  Calendar days      : {total_days}")
    if total_days > 0:
        print(f"  Activity rate      : {len(active_days) / total_days:.1%}")
    print(f"{'=' * 70}")
    print(f"  Gates:")
    print(f"    Total signals    : {gate_stats['total_signals']}")
    print(f"    Regime blocked   : {gate_stats['regime_blocked']}")
    print(f"    CB halted        : {gate_stats['cb_halted']}")
    print(f"    Kelly blocked    : {gate_stats['kelly_blocked']}")
    print(f"{'=' * 70}")


# -- Main ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Combined multi-asset backtest")
    p.add_argument("--btc-model", default="models/xgb_btc_15m_iter5.pkl")
    p.add_argument("--sol-model", default="models/xgb_sol_15m.pkl")
    p.add_argument("--btc", default="data/BTCUSDT_15m.parquet")
    p.add_argument("--eth", default="data/ETHUSDT_15m.parquet")
    p.add_argument("--sol", default="data/SOLUSDT_15m.parquet")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--btc-threshold", type=float, default=0.65)
    p.add_argument("--sol-threshold", type=float, default=0.70)
    p.add_argument("--exit-threshold", type=float, default=0.08,
                    help="Exit threshold for both models (default 0.08)")
    p.add_argument("--bear-mult", type=float, default=0.35)
    p.add_argument("--sideways-mult", type=float, default=0.50)
    p.add_argument("--bull-mult", type=float, default=1.00)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--fee-bps", type=int, default=10)
    p.add_argument("--atr-mult", type=float, default=10.0)
    p.add_argument("--risk-per-trade", type=float, default=0.02)
    p.add_argument("--compare", action="store_true",
                    help="Run old params vs new params side-by-side")
    return p.parse_args()


def run_one(args, label, btc_threshold, sol_threshold, exit_threshold, bear_mult,
            btc_feat, sol_feat, btc_probas, sol_probas, btc_raw):
    regime_mults = compute_regime_series(
        btc_raw, args.bull_mult, args.sideways_mult, bear_mult
    )

    returns, portfolio, trades, gates, active_days = run_combined_backtest(
        btc_feat, sol_feat, btc_probas, sol_probas, regime_mults,
        btc_threshold=btc_threshold,
        sol_threshold=sol_threshold,
        btc_exit_threshold=exit_threshold,
        sol_exit_threshold=exit_threshold,
        initial_capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        hard_stop_pct=0.05,
        atr_mult=args.atr_mult,
        fee_bps=args.fee_bps,
    )

    compute_stats(returns, portfolio, trades, gates, active_days, label)
    return returns, portfolio, trades, gates, active_days


def main():
    args = parse_args()

    print("Loading BTC features...")
    btc_feat, btc_raw, eth_raw, sol_raw = prepare_btc_features(
        args.btc, args.eth, args.sol, args.start, args.end
    )
    print(f"  BTC features: {btc_feat.shape[0]:,} bars x {btc_feat.shape[1]} cols")

    print("Loading SOL features...")
    sol_feat = prepare_sol_features(btc_raw, eth_raw, sol_raw, btc_feat.index, args.start, args.end)
    print(f"  SOL features: {sol_feat.shape[0]:,} bars x {sol_feat.shape[1]} cols")

    print("Loading models...")
    btc_model = load_model(args.btc_model)
    sol_model = load_model(args.sol_model)
    print(f"  BTC model: {len(btc_model.feature_names_in_)} features")
    print(f"  SOL model: {len(sol_model.feature_names_in_)} features")

    print("Batch predicting...")
    btc_probas = batch_predict(btc_model, btc_feat)
    sol_probas = batch_predict(sol_model, sol_feat)
    print(f"  BTC valid probas: {(~np.isnan(btc_probas)).sum():,}")
    print(f"  SOL valid probas: {(~np.isnan(sol_probas)).sum():,}")

    if args.compare:
        # Old params (current broken config)
        run_one(args, "OLD (bear=0.0x, SOL_t=0.75, exit=0.10)",
                btc_threshold=0.65, sol_threshold=0.75, exit_threshold=0.10,
                bear_mult=0.0,
                btc_feat=btc_feat, sol_feat=sol_feat,
                btc_probas=btc_probas, sol_probas=sol_probas, btc_raw=btc_raw)

        # New params (our fix)
        run_one(args, "NEW (bear=0.35x, SOL_t=0.70, exit=0.08)",
                btc_threshold=0.65, sol_threshold=0.70, exit_threshold=0.08,
                bear_mult=0.35,
                btc_feat=btc_feat, sol_feat=sol_feat,
                btc_probas=btc_probas, sol_probas=sol_probas, btc_raw=btc_raw)
    else:
        run_one(args, f"bear={args.bear_mult}x SOL_t={args.sol_threshold} exit={args.exit_threshold}",
                btc_threshold=args.btc_threshold, sol_threshold=args.sol_threshold,
                exit_threshold=args.exit_threshold, bear_mult=args.bear_mult,
                btc_feat=btc_feat, sol_feat=sol_feat,
                btc_probas=btc_probas, sol_probas=sol_probas, btc_raw=btc_raw)


if __name__ == "__main__":
    main()

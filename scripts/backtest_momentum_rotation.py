#!/usr/bin/env python3
"""
Cross-sectional momentum rotation backtest with 10-day window analysis.

Simulates: rank 20 coins → hold top 4 → inverse-vol weight → buffer zones →
4-layer crash protection → rebalance every 4H.

Reports: distribution of 10-day Sortino/Sharpe/Calmar vs BTC buy-and-hold.

Usage:
  python scripts/backtest_momentum_rotation.py
  python scripts/backtest_momentum_rotation.py --sweep-alpha
  python scripts/backtest_momentum_rotation.py --sweep-buffer
"""
import argparse
import sys
import datetime
import numpy as np
import pandas as pd
import quantstats as qs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.strategy.momentum_signals import (
    resample_to_4h, sharpe_momentum, nearness_to_high,
    residual_momentum, compute_regime_flag, adjust_weights_for_regime,
)

PERIODS_4H = 365.25 * 6  # annualization for 4H data
COINS = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","LINK","DOT",
         "LTC","UNI","NEAR","SUI","APT","PEPE","ARB","SHIB","FIL","HBAR"]
WINDOW_BARS = 60  # 10 days at 4H = 60 bars


def load_4h_data():
    """Load all 20 coins, resample to 4H, return dict."""
    print("Loading data...", flush=True)
    coin_4h = {}
    for coin in COINS:
        df = pd.read_parquet(f"data/{coin}USDT_15m.parquet")
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()
        coin_4h[coin] = resample_to_4h(df)
    return coin_4h


def compute_all_scores(coin_4h, idx, weights, btc_ret):
    """
    Compute composite scores for all coins at a given bar index.

    CONTRARIAN SCORING: momentum components are NEGATED.
    IC analysis showed momentum has negative predictive power.
    Contrarian signals (buy recent losers) have positive IC.
    """
    scores = {}
    vols = {}
    for coin in COINS:
        c4h = coin_4h[coin]
        if idx not in c4h.index:
            continue
        loc = c4h.index.get_loc(idx)
        if loc < 200:
            continue

        closes = c4h["close"].iloc[:loc + 1]
        score = 0.0

        # CONTRARIAN: negate sharpe_48h momentum
        if weights.get("contra_48h", 0) > 0:
            score += weights["contra_48h"] * (-sharpe_momentum(closes, 12, 1))

        # nearness unchanged (positive IC as-is)
        if weights.get("nearness", 0) > 0:
            score += weights["nearness"] * nearness_to_high(closes, 180)

        # CONTRARIAN: negate sharpe_168h momentum
        if weights.get("contra_168h", 0) > 0:
            score += weights["contra_168h"] * (-sharpe_momentum(closes, 42, 1))

        # CONTRARIAN: negate residual momentum
        if weights.get("contra_residual", 0) > 0:
            coin_ret = np.log(c4h["close"].iloc[:loc+1] / c4h["close"].iloc[:loc+1].shift(1)).dropna()
            common = coin_ret.index.intersection(btc_ret.index)
            common = common[common <= idx]
            if len(common) > 60:
                score += weights["contra_residual"] * (-residual_momentum(
                    coin_ret.reindex(common), btc_ret.reindex(common)
                ))

        # Volatility for inverse-vol weighting
        rets = np.log(closes / closes.shift(1)).dropna()
        vol = rets.iloc[-42:].std() if len(rets) >= 42 else rets.std()

        scores[coin] = score
        vols[coin] = max(vol, 1e-10)

    return scores, vols


def run_momentum_backtest(
    coin_4h, weights,
    start_date="2025-06-01",
    initial=1_000_000.0,
    n_holdings=4,
    sell_rank=6,
    ema_alpha=0.4,
    fee_bps=5,
    dd_flat=0.12,
    dd_half=0.08,
    dd_quarter=0.05,
):
    """Run full momentum rotation backtest. Returns (returns_series, stats_dict)."""
    btc_4h = coin_4h["BTC"]
    btc_ret = np.log(btc_4h["close"] / btc_4h["close"].shift(1)).dropna()

    start_ts = pd.Timestamp(start_date, tz="UTC")
    timeline = btc_4h.index[btc_4h.index >= start_ts]
    n = len(timeline)
    fee_rate = fee_bps / 10_000

    # State
    free = initial
    hwm = initial
    positions = {}  # coin -> {"units": float, "value": float}
    smoothed_scores = {}  # coin -> float (EMA smoothed)
    port_vals = np.zeros(n)
    port_vals[0] = initial
    rets = np.zeros(n)
    n_trades = 0
    active_dates = set()

    for i, ts in enumerate(timeline):
        # Mark to market
        port_val = free
        for coin, pos in positions.items():
            if ts in coin_4h[coin].index:
                price = coin_4h[coin].loc[ts, "close"]
                port_val += pos["units"] * price
        hwm = max(hwm, port_val)

        # Compute scores
        raw_scores, vols = compute_all_scores(coin_4h, ts, weights, btc_ret)

        # EMA smooth
        for coin, score in raw_scores.items():
            prev = smoothed_scores.get(coin, score)
            smoothed_scores[coin] = ema_alpha * score + (1 - ema_alpha) * prev

        # Rank by smoothed score
        ranked = sorted(smoothed_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_coins = [c for c, _ in ranked if c in raw_scores]

        # Crash protection
        drawdown = (hwm - port_val) / hwm if hwm > 0 else 0
        if drawdown > dd_flat:
            dd_scalar = 0.0
        elif drawdown > dd_half:
            dd_scalar = 0.50
        elif drawdown > dd_quarter:
            dd_scalar = 0.75
        else:
            dd_scalar = 1.0

        # BTC TSMOM
        btc_7d = btc_ret.loc[btc_ret.index <= ts].iloc[-42:]
        btc_ema_ret = btc_7d.ewm(span=7).mean().iloc[-1] if len(btc_7d) > 7 else 0
        if btc_ema_ret < -0.05:
            tsmom = 0.0
        elif btc_ema_ret < 0:
            tsmom = 0.5
        else:
            tsmom = 1.0

        # Dispersion
        period_rets = {}
        for coin in COINS:
            if ts in coin_4h[coin].index:
                loc = coin_4h[coin].index.get_loc(ts)
                if loc > 0:
                    period_rets[coin] = np.log(
                        coin_4h[coin]["close"].iloc[loc] /
                        coin_4h[coin]["close"].iloc[loc - 1]
                    )
        dispersion_scalar = 1.0
        if len(period_rets) >= 10:
            disp = np.std(list(period_rets.values()))
            # Simple threshold: if dispersion < 0.005 (0.5%), reduce
            if disp < 0.005:
                dispersion_scalar = 0.5

        exposure = dd_scalar * min(tsmom, dispersion_scalar)

        # Target portfolio
        if exposure == 0:
            # Go flat
            for coin in list(positions.keys()):
                if ts in coin_4h[coin].index:
                    price = coin_4h[coin].loc[ts, "close"]
                    free += positions[coin]["units"] * price * (1 - fee_rate)
                    n_trades += 1
                    active_dates.add(ts.date())
            positions.clear()
        else:
            # Determine target holdings using buffer zones
            current_held = set(positions.keys())
            target_coins = []

            for coin in ranked_coins:
                if len(target_coins) >= n_holdings:
                    break
                rank = ranked_coins.index(coin) + 1
                if coin in current_held:
                    if rank <= sell_rank:  # hold zone
                        target_coins.append(coin)
                    # else: sell (below sell_rank)
                else:
                    if rank <= n_holdings:  # buy zone
                        target_coins.append(coin)

            # Fill remaining slots from top-ranked non-held
            if len(target_coins) < n_holdings:
                for coin in ranked_coins:
                    if coin not in target_coins and len(target_coins) < n_holdings:
                        target_coins.append(coin)

            # Inverse-vol weights
            target_vols = {c: vols.get(c, 0.01) for c in target_coins}
            inv_vol = {c: 1.0 / v for c, v in target_vols.items()}
            total_inv = sum(inv_vol.values())
            target_weights = {c: (inv_vol[c] / total_inv) * exposure for c in target_coins}

            # Sell coins no longer in target
            for coin in list(positions.keys()):
                if coin not in target_coins:
                    if ts in coin_4h[coin].index:
                        price = coin_4h[coin].loc[ts, "close"]
                        free += positions[coin]["units"] * price * (1 - fee_rate)
                        n_trades += 1
                        active_dates.add(ts.date())
                    del positions[coin]

            # Buy/adjust target coins
            total_val = free + sum(
                positions[c]["units"] * coin_4h[c].loc[ts, "close"]
                for c in positions if ts in coin_4h[c].index
            )

            for coin in target_coins:
                target_usd = total_val * target_weights[coin]
                if ts not in coin_4h[coin].index:
                    continue
                price = coin_4h[coin].loc[ts, "close"]

                current_usd = 0
                if coin in positions:
                    current_usd = positions[coin]["units"] * price

                diff = target_usd - current_usd
                if abs(diff) < max(5000, total_val * 0.005):
                    continue  # min trade threshold

                if diff > 0:  # buy more
                    buy_usd = min(diff, free * 0.95)
                    if buy_usd > 10:
                        units = buy_usd / price
                        cost = buy_usd * (1 + fee_rate)
                        free -= cost
                        if coin in positions:
                            positions[coin]["units"] += units
                        else:
                            positions[coin] = {"units": units}
                        n_trades += 1
                        active_dates.add(ts.date())
                elif diff < 0:  # sell some
                    sell_usd = min(-diff, current_usd)
                    sell_units = sell_usd / price
                    positions[coin]["units"] -= sell_units
                    free += sell_usd * (1 - fee_rate)
                    if positions[coin]["units"] < 0.0001:
                        del positions[coin]
                    n_trades += 1
                    active_dates.add(ts.date())

        # Record
        port_val = free
        for coin, pos in positions.items():
            if ts in coin_4h[coin].index:
                port_val += pos["units"] * coin_4h[coin].loc[ts, "close"]
        port_vals[i] = port_val
        if i > 0:
            rets[i] = port_vals[i] / port_vals[i - 1] - 1.0

    ret_s = pd.Series(rets, index=timeline)
    total_days = (timeline[-1] - timeline[0]).days

    return ret_s, {
        "sharpe": float(qs.stats.sharpe(ret_s, periods=PERIODS_4H)),
        "sortino": float(qs.stats.sortino(ret_s, periods=PERIODS_4H)),
        "maxdd": float(qs.stats.max_drawdown(ret_s)) * 100,
        "ret": (port_vals[-1] - initial) / initial * 100,
        "trades": n_trades,
        "active_pct": len(active_dates) / max(total_days, 1) * 100,
    }


def run_10d_windows(ret_s, btc_4h, start_date="2025-06-01"):
    """Compute 10-day window statistics and compare to BTC hold."""
    start_ts = pd.Timestamp(start_date, tz="UTC")
    btc_rets = np.log(btc_4h["close"] / btc_4h["close"].shift(1)).dropna()
    btc_rets = btc_rets[btc_rets.index >= start_ts]

    strat_windows = []
    btc_windows = []

    for start in range(0, len(ret_s) - WINDOW_BARS, 6):  # step 1 day = 6 bars
        end = start + WINDOW_BARS
        w_ret = ret_s.iloc[start:end]
        std = w_ret.std()
        if std < 1e-10:
            continue

        sharpe = float(qs.stats.sharpe(w_ret, periods=PERIODS_4H))
        sortino = float(qs.stats.sortino(w_ret, periods=PERIODS_4H))
        total_ret = (1 + w_ret).prod() - 1
        maxdd = float(qs.stats.max_drawdown(w_ret)) * 100
        calmar = abs(total_ret / (maxdd / 100)) if maxdd != 0 else 0
        score = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

        strat_windows.append({
            "ret": total_ret * 100, "sharpe": sharpe, "sortino": sortino,
            "calmar": calmar, "maxdd": maxdd, "score": score,
        })

        # BTC hold for same window
        b_idx = btc_rets.index[start:end] if start + end <= len(btc_rets) else []
        if len(b_idx) >= WINDOW_BARS:
            b_ret = btc_rets.iloc[start:end]
            b_sharpe = float(qs.stats.sharpe(b_ret, periods=PERIODS_4H))
            b_sortino = float(qs.stats.sortino(b_ret, periods=PERIODS_4H))
            b_total = (1 + b_ret).prod() - 1
            b_maxdd = float(qs.stats.max_drawdown(b_ret)) * 100
            b_calmar = abs(b_total / (b_maxdd / 100)) if b_maxdd != 0 else 0
            b_score = 0.4 * b_sortino + 0.3 * b_sharpe + 0.3 * b_calmar
            btc_windows.append({"score": b_score, "ret": b_total * 100})

    return strat_windows, btc_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="JSON dict of contrarian weights, e.g. '{\"contra_48h\":0.335,\"contra_168h\":0.323,\"contra_residual\":0.259,\"nearness\":0.083}'")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--sweep-alpha", action="store_true")
    parser.add_argument("--sweep-buffer", action="store_true")
    args = parser.parse_args()

    coin_4h = load_4h_data()

    # Default contrarian weights (IC-derived from analysis)
    import json
    if args.weights:
        weights = json.loads(args.weights)
    else:
        # IC-derived weights for contrarian scoring (negated momentum components)
        weights = {
            "contra_48h": 0.335,      # NEGATED sharpe_momentum(12 bars)
            "contra_168h": 0.323,     # NEGATED sharpe_momentum(42 bars)
            "contra_residual": 0.259, # NEGATED residual_momentum
            "nearness": 0.083,        # unchanged (positive IC as-is)
        }

    print(f"Weights (contrarian): {weights}", flush=True)

    if args.sweep_alpha:
        print("\n=== EMA Alpha Sweep (Contrarian) ===", flush=True)
        for alpha in [0.2, 0.3, 0.4, 0.5, 0.6]:
            _, s = run_momentum_backtest(coin_4h, weights, args.start, ema_alpha=alpha)
            print(f"  alpha={alpha}: Sharpe={s['sharpe']:.3f} Sortino={s['sortino']:.3f} "
                  f"Trades={s['trades']} Ret={s['ret']:+.1f}% DD={s['maxdd']:.1f}%", flush=True)
        return

    if args.sweep_buffer:
        print("\n=== Buffer Zone Sweep (Contrarian) ===", flush=True)
        for sell in [5, 6, 7, 8]:
            _, s = run_momentum_backtest(coin_4h, weights, args.start, sell_rank=sell)
            print(f"  sell_rank={sell}: Sharpe={s['sharpe']:.3f} Sortino={s['sortino']:.3f} "
                  f"Trades={s['trades']} Ret={s['ret']:+.1f}% DD={s['maxdd']:.1f}%", flush=True)
        return

    # Main run
    print("\nRunning contrarian momentum rotation backtest...", flush=True)
    ret_s, stats = run_momentum_backtest(coin_4h, weights, args.start)

    print("\n" + "=" * 70)
    print(f"  CONTRARIAN MOMENTUM ROTATION BACKTEST ({args.start} onwards)")
    print("=" * 70)
    print(f"  Sharpe:  {stats['sharpe']:.3f}")
    print(f"  Sortino: {stats['sortino']:.3f}")
    print(f"  MaxDD:   {stats['maxdd']:.1f}%")
    print(f"  Return:  {stats['ret']:+.1f}%")
    print(f"  Trades:  {stats['trades']}")
    print(f"  Active:  {stats['active_pct']:.0f}%")

    # 10-day window analysis
    print("\n--- 10-Day Window Analysis ---", flush=True)
    strat_w, btc_w = run_10d_windows(ret_s, coin_4h["BTC"], args.start)

    if strat_w:
        scores = [w["score"] for w in strat_w]
        rets_list = [w["ret"] for w in strat_w]
        print(f"  Windows analyzed: {len(strat_w)}")
        print(f"  Median score:  {np.median(scores):.3f}")
        print(f"  Median return: {np.median(rets_list):+.2f}%")
        print(f"  P5 return:     {np.percentile(rets_list, 5):+.2f}%")
        print(f"  P95 return:    {np.percentile(rets_list, 95):+.2f}%")
        print(f"  % positive:    {sum(1 for r in rets_list if r > 0)/len(rets_list)*100:.0f}%")

        if btc_w:
            btc_scores = [w["score"] for w in btc_w]
            beat_btc = sum(1 for s, b in zip(scores, btc_scores) if s > b) / len(scores) * 100
            print(f"  % beat BTC:    {beat_btc:.0f}%")
            print(f"  BTC med score: {np.median(btc_scores):.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()

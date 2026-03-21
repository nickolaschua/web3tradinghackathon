#!/usr/bin/env python3
"""Quick hybrid backtest: BTC+SOL XGBoost + Relaxed MR on 20 coins."""
import sys, pickle, datetime
import numpy as np, pandas as pd, quantstats as qs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.data.features import compute_features, compute_cross_asset_features, compute_btc_context_features
from bot.strategy.relaxed_mean_reversion import RelaxedMeanReversionStrategy
from bot.strategy.base import SignalDirection

PERIODS = 35040
COINS = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","LINK","DOT",
         "LTC","UNI","NEAR","SUI","APT","PEPE","ARB","SHIB","FIL","HBAR"]

cutoff = pd.Timestamp("2024-01-01", tz="UTC")

# Load raw data
btc_raw = pd.read_parquet("data/BTCUSDT_15m.parquet")
eth_raw = pd.read_parquet("data/ETHUSDT_15m.parquet")
sol_raw = pd.read_parquet("data/SOLUSDT_15m.parquet")
for df in (btc_raw, eth_raw, sol_raw):
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str.lower()

# BTC features (iter5 style)
print("Preparing BTC features...")
btc_feat = compute_features(btc_raw)
btc_feat = compute_cross_asset_features(btc_feat, {"ETH/USD": eth_raw, "SOL/USD": sol_raw})
for a, d in [("eth", eth_raw), ("sol", sol_raw)]:
    lr = np.log(d["close"] / d["close"].shift(1))
    btc_feat[f"{a}_return_4h"] = lr.shift(16).reindex(btc_feat.index)
    btc_feat[f"{a}_return_1d"] = lr.shift(96).reindex(btc_feat.index)
btc_feat = compute_btc_context_features(btc_feat, eth_raw, sol_raw, window=2880)
btc_feat = btc_feat.dropna()
btc_feat = btc_feat[btc_feat.index >= cutoff]

# SOL features
print("Preparing SOL features...")
sol_feat = compute_features(sol_raw)
sol_feat = compute_cross_asset_features(sol_feat, {"BTC/USD": btc_raw, "ETH/USD": eth_raw})
for a, d in [("btc", btc_raw), ("eth", eth_raw)]:
    lr = np.log(d["close"] / d["close"].shift(1))
    sol_feat[f"{a}_return_4h"] = lr.shift(16).reindex(sol_feat.index)
    sol_feat[f"{a}_return_1d"] = lr.shift(96).reindex(sol_feat.index)
sr = np.log(sol_raw["close"] / sol_raw["close"].shift(1)).reindex(sol_feat.index)
br = np.log(btc_raw["close"] / btc_raw["close"].shift(1)).reindex(sol_feat.index)
sol_feat["sol_btc_corr"] = sr.rolling(2880).corr(br).shift(1)
sol_feat["sol_btc_beta"] = (sr.rolling(2880).cov(br) / (br.rolling(2880).var() + 1e-10)).shift(1)
sol_feat = sol_feat.dropna()
sol_feat = sol_feat[sol_feat.index >= cutoff]

# Models
print("Loading models...")
with open("models/xgb_btc_15m_iter5.pkl", "rb") as f:
    btc_model = pickle.load(f)
with open("models/xgb_sol_15m.pkl", "rb") as f:
    sol_model = pickle.load(f)

btc_p = btc_model.predict_proba(btc_feat[list(btc_model.feature_names_in_)])[:, 1]
sol_p = sol_model.predict_proba(sol_feat[list(sol_model.feature_names_in_)])[:, 1]

# MR features for all coins
print("Preparing MR features for 20 coins...")
mr = RelaxedMeanReversionStrategy()
coin_feats = {}
for coin in COINS:
    df = pd.read_parquet(f"data/{coin}USDT_15m.parquet")
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str.lower()
    feat = compute_features(df).dropna()
    feat = feat[feat.index >= cutoff]
    coin_feats[coin] = feat

# Common timeline
common_idx = btc_feat.index.intersection(sol_feat.index)
for coin in COINS:
    common_idx = common_idx.intersection(coin_feats[coin].index)
print(f"Common bars: {len(common_idx)}")

# Align data
btc_closes = btc_feat.reindex(common_idx)["close"].values
sol_closes = sol_feat.reindex(common_idx)["close"].values
btc_atrs = btc_feat.reindex(common_idx)["atr_proxy"].values
sol_atrs = sol_feat.reindex(common_idx)["atr_proxy"].values
btc_ps = pd.Series(btc_p, index=btc_feat.index).reindex(common_idx).fillna(0).values
sol_ps = pd.Series(sol_p, index=sol_feat.index).reindex(common_idx).fillna(0).values

# Simulation
fee_rate = 0.001
initial = 1_000_000.0
free = initial
hwm = initial
n = len(common_idx)

xgb_pos = {
    "BTC": {"units": 0.0, "ep": 0.0, "trail": 0.0},
    "SOL": {"units": 0.0, "ep": 0.0, "trail": 0.0},
}
mr_positions = {}
max_mr_positions = 1
BTC_T, SOL_T = 0.65, 0.75

port_vals = np.zeros(n)
port_vals[0] = initial
returns_arr = np.zeros(n)
trades = []
active_dates = set()

print("Running simulation...")
for i in range(n):
    ts = common_idx[i]

    # Mark to market
    xgb_val = 0.0
    if xgb_pos["BTC"]["units"] > 0:
        xgb_val += xgb_pos["BTC"]["units"] * btc_closes[i]
    if xgb_pos["SOL"]["units"] > 0:
        xgb_val += xgb_pos["SOL"]["units"] * sol_closes[i]
    mr_val = 0.0
    for c, p in mr_positions.items():
        if ts in coin_feats[c].index:
            mr_val += p["units"] * coin_feats[c].loc[ts, "close"]
    total = free + xgb_val + mr_val
    hwm = max(hwm, total)

    # XGB exits
    for name, closes, atrs, ps in [("BTC", btc_closes, btc_atrs, btc_ps),
                                     ("SOL", sol_closes, sol_atrs, sol_ps)]:
        s = xgb_pos[name]
        if s["units"] <= 0:
            continue
        c = closes[i]
        atr = atrs[i]
        if not np.isnan(atr) and atr > 0:
            new_stop = c - 10.0 * atr
            s["trail"] = max(s["trail"], new_stop)
        if c <= s["trail"] or ps[i] <= 0.10:
            proceeds = s["units"] * c * (1 - fee_rate)
            pnl = (c * (1 - fee_rate) - s["ep"]) / s["ep"] if s["ep"] > 0 else 0
            trades.append({"coin": name, "pnl": pnl, "src": "xgb"})
            active_dates.add(ts.date())
            free += proceeds
            s["units"] = 0.0

    # MR exits
    coins_to_close = []
    for coin, p in mr_positions.items():
        if ts not in coin_feats[coin].index:
            continue
        sig = mr.generate_signal(f"{coin}/USD", coin_feats[coin].loc[[ts]])
        if sig.direction == SignalDirection.SELL:
            c = coin_feats[coin].loc[ts, "close"]
            proceeds = p["units"] * c * (1 - fee_rate)
            pnl = (c * (1 - fee_rate) - p["ep"]) / p["ep"] if p["ep"] > 0 else 0
            trades.append({"coin": coin, "pnl": pnl, "src": "mr"})
            active_dates.add(ts.date())
            free += proceeds
            coins_to_close.append(coin)
    for c in coins_to_close:
        del mr_positions[c]

    # Recalculate total
    xgb_val = 0.0
    if xgb_pos["BTC"]["units"] > 0:
        xgb_val += xgb_pos["BTC"]["units"] * btc_closes[i]
    if xgb_pos["SOL"]["units"] > 0:
        xgb_val += xgb_pos["SOL"]["units"] * sol_closes[i]
    mr_val = sum(
        p["units"] * coin_feats[c].loc[ts, "close"]
        for c, p in mr_positions.items()
        if ts in coin_feats[c].index
    )
    total = free + xgb_val + mr_val

    # XGB entries
    for name, closes, atrs, ps, threshold in [
        ("BTC", btc_closes, btc_atrs, btc_ps, BTC_T),
        ("SOL", sol_closes, sol_atrs, sol_ps, SOL_T),
    ]:
        s = xgb_pos[name]
        if s["units"] > 0 or ps[i] < threshold:
            continue
        c = closes[i]
        atr = atrs[i]
        kelly = (ps[i] * 1.5 - (1 - ps[i])) / 1.5
        if kelly <= 0:
            continue
        hard_stop = c * 0.95
        atr_stop = c - 10 * atr if not np.isnan(atr) and atr > 0 else hard_stop
        initial_stop = max(hard_stop, atr_stop)
        stop_d = min(c - initial_stop, c * 0.05)
        if stop_d <= 0:
            stop_d = c * 0.05
        risk_usd = total * 0.02 * ps[i]
        target = min(risk_usd / stop_d * c, total * 0.4, free * 0.95)
        if target >= 10:
            s["units"] = target / c
            s["ep"] = c * (1 + fee_rate)
            s["trail"] = initial_stop
            free -= target * (1 + fee_rate)
            active_dates.add(ts.date())

    # MR entries (skip BTC/SOL — they have XGBoost models)
    if len(mr_positions) < max_mr_positions:
        for coin in COINS:
            if coin in ("BTC", "SOL") or coin in mr_positions:
                continue
            if len(mr_positions) >= max_mr_positions:
                break
            if ts not in coin_feats[coin].index:
                continue
            sig = mr.generate_signal(f"{coin}/USD", coin_feats[coin].loc[[ts]])
            if sig.direction == SignalDirection.BUY:
                c = coin_feats[coin].loc[ts, "close"]
                target = total * sig.size  # 0.10-0.15
                target = min(target, free * 0.95)
                if target >= 10:
                    mr_positions[coin] = {"units": target / c, "ep": c * (1 + fee_rate)}
                    free -= target * (1 + fee_rate)
                    active_dates.add(ts.date())

    # Record
    xgb_val = 0.0
    if xgb_pos["BTC"]["units"] > 0:
        xgb_val += xgb_pos["BTC"]["units"] * btc_closes[i]
    if xgb_pos["SOL"]["units"] > 0:
        xgb_val += xgb_pos["SOL"]["units"] * sol_closes[i]
    mr_val = sum(
        p["units"] * coin_feats[c].loc[ts, "close"]
        for c, p in mr_positions.items()
        if ts in coin_feats[c].index
    )
    port_vals[i] = free + xgb_val + mr_val
    if i > 0:
        returns_arr[i] = port_vals[i] / port_vals[i - 1] - 1.0

ret_s = pd.Series(returns_arr, index=common_idx)
port_s = pd.Series(port_vals, index=common_idx)

total_days = (common_idx[-1] - common_idx[0]).days
sharpe = float(qs.stats.sharpe(ret_s, periods=PERIODS))
sortino = float(qs.stats.sortino(ret_s, periods=PERIODS))
maxdd = float(qs.stats.max_drawdown(ret_s)) * 100
total_ret = (port_vals[-1] - initial) / initial * 100
n_trades = len(trades)
xgb_trades = [t for t in trades if t["src"] == "xgb"]
mr_trades_list = [t for t in trades if t["src"] == "mr"]
winners = sum(1 for t in trades if t["pnl"] > 0)

print("=" * 70)
print("  HYBRID BACKTEST: BTC(0.65) + SOL(0.75) + Relaxed MR (18 coins)")
print("=" * 70)
print(f"  Sharpe:  {sharpe:.3f}")
print(f"  Sortino: {sortino:.3f}")
print(f"  MaxDD:   {maxdd:.2f}%")
print(f"  Return:  {total_ret:+.2f}%")
print(f"  Final:   ${port_vals[-1]:,.0f}")
print(f"  Trades:  {n_trades} (XGB: {len(xgb_trades)}, MR: {len(mr_trades_list)})")
if n_trades > 0:
    print(f"  Win rate: {winners/n_trades:.1%}")
    print(f"  Avg PnL:  {sum(t['pnl'] for t in trades)/n_trades*100:+.2f}%")
print(f"\n  Active trading days: {len(active_dates)}/{total_days} ({len(active_dates)/total_days:.1%})")

# Worst 10-day window
all_cal_dates = [common_idx[0].date() + datetime.timedelta(days=d) for d in range(total_days)]
worst_10d = total_days
for si in range(len(all_cal_dates) - 10):
    window = all_cal_dates[si : si + 10]
    active_in = sum(1 for d in window if d in active_dates)
    worst_10d = min(worst_10d, active_in)
print(f"  Worst 10-day window: {worst_10d}/10 active days")
last_30 = [d for d in sorted(active_dates) if d >= (common_idx[-1] - pd.Timedelta(days=30)).date()]
print(f"  Last 30 days active: {len(last_30)}/30")

print(f"\n  XGB breakdown: {len(xgb_trades)} trades")
if xgb_trades:
    xw = sum(1 for t in xgb_trades if t["pnl"] > 0)
    print(f"    Win: {xw/len(xgb_trades):.1%}  Avg PnL: {sum(t['pnl'] for t in xgb_trades)/len(xgb_trades)*100:+.2f}%")
print(f"  MR breakdown:  {len(mr_trades_list)} trades")
if mr_trades_list:
    mw = sum(1 for t in mr_trades_list if t["pnl"] > 0)
    print(f"    Win: {mw/len(mr_trades_list):.1%}  Avg PnL: {sum(t['pnl'] for t in mr_trades_list)/len(mr_trades_list)*100:+.2f}%")

# Per-coin MR breakdown
mr_by_coin = {}
for t in mr_trades_list:
    mr_by_coin.setdefault(t["coin"], []).append(t)
print(f"\n  MR per-coin breakdown:")
for coin in sorted(mr_by_coin.keys()):
    ct = mr_by_coin[coin]
    w = sum(1 for t in ct if t["pnl"] > 0)
    avg = sum(t["pnl"] for t in ct) / len(ct) * 100
    print(f"    {coin:>5}: {len(ct):>3} trades  Win: {w/len(ct):.0%}  Avg: {avg:+.2f}%")
print("=" * 70)

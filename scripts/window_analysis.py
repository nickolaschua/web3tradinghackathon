#!/usr/bin/env python3
"""10-day sliding window analysis for competition threshold selection."""
import sys, pickle
import numpy as np, pandas as pd, quantstats as qs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.data.features import compute_features, compute_cross_asset_features, compute_btc_context_features

PERIODS = 35040
WINDOW = 960  # 10 days at 15M
STEP = 480    # 5-day sliding step
cutoff = pd.Timestamp("2024-01-01", tz="UTC")


def load_data():
    print("Loading data...", flush=True)
    btc_raw = pd.read_parquet("data/BTCUSDT_15m.parquet")
    eth_raw = pd.read_parquet("data/ETHUSDT_15m.parquet")
    sol_raw = pd.read_parquet("data/SOLUSDT_15m.parquet")
    for df in (btc_raw, eth_raw, sol_raw):
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    print("Preparing BTC features...", flush=True)
    btc_feat = compute_features(btc_raw)
    btc_feat = compute_cross_asset_features(btc_feat, {"ETH/USD": eth_raw, "SOL/USD": sol_raw})
    for a, d in [("eth", eth_raw), ("sol", sol_raw)]:
        lr = np.log(d["close"] / d["close"].shift(1))
        btc_feat[f"{a}_return_4h"] = lr.shift(16).reindex(btc_feat.index)
        btc_feat[f"{a}_return_1d"] = lr.shift(96).reindex(btc_feat.index)
    btc_feat = compute_btc_context_features(btc_feat, eth_raw, sol_raw, window=2880)
    btc_feat = btc_feat.dropna()
    btc_feat = btc_feat[btc_feat.index >= cutoff]

    print("Preparing SOL features...", flush=True)
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

    print("Loading models...", flush=True)
    with open("models/xgb_btc_15m_iter5.pkl", "rb") as f:
        btc_model = pickle.load(f)
    with open("models/xgb_sol_15m.pkl", "rb") as f:
        sol_model = pickle.load(f)

    common = btc_feat.index.intersection(sol_feat.index)
    return {
        "common": common,
        "n": len(common),
        "btc_c": btc_feat.reindex(common)["close"].values,
        "sol_c": sol_feat.reindex(common)["close"].values,
        "btc_a": btc_feat.reindex(common)["atr_proxy"].values,
        "sol_a": sol_feat.reindex(common)["atr_proxy"].values,
        "btc_p": pd.Series(
            btc_model.predict_proba(btc_feat[list(btc_model.feature_names_in_)])[:, 1],
            index=btc_feat.index
        ).reindex(common).fillna(0).values,
        "sol_p": pd.Series(
            sol_model.predict_proba(sol_feat[list(sol_model.feature_names_in_)])[:, 1],
            index=sol_feat.index
        ).reindex(common).fillna(0).values,
    }


def run_window(data, start, btc_t, sol_t, exit_t=0.10):
    end = start + WINDOW
    if end > data["n"]:
        return None
    fee = 0.001
    free = 1_000_000.0
    btc_u, btc_ep, btc_tr = 0.0, 0.0, 0.0
    sol_u, sol_ep, sol_tr = 0.0, 0.0, 0.0
    port = np.zeros(WINDOW)
    port[0] = free
    rets = np.zeros(WINDOW)
    n_trades = 0
    active_dates = set()
    common = data["common"]
    btc_c, sol_c = data["btc_c"], data["sol_c"]
    btc_a, sol_a = data["btc_a"], data["sol_a"]
    btc_p, sol_p = data["btc_p"], data["sol_p"]

    for i in range(start, end):
        j = i - start
        # Trail
        if btc_u > 0:
            a = btc_a[i]
            if not np.isnan(a) and a > 0:
                btc_tr = max(btc_tr, btc_c[i] - 10 * a)
        if sol_u > 0:
            a = sol_a[i]
            if not np.isnan(a) and a > 0:
                sol_tr = max(sol_tr, sol_c[i] - 10 * a)

        total = free + btc_u * btc_c[i] + sol_u * sol_c[i]

        # Exits
        if btc_u > 0 and (btc_c[i] <= btc_tr or btc_p[i] <= exit_t):
            free += btc_u * btc_c[i] * (1 - fee)
            btc_u = 0.0
            n_trades += 1
            active_dates.add(common[i].date())
        if sol_u > 0 and (sol_c[i] <= sol_tr or sol_p[i] <= exit_t):
            free += sol_u * sol_c[i] * (1 - fee)
            sol_u = 0.0
            n_trades += 1
            active_dates.add(common[i].date())

        total = free + btc_u * btc_c[i] + sol_u * sol_c[i]

        # Entries
        if btc_u == 0 and btc_p[i] >= btc_t:
            k = (btc_p[i] * 1.5 - (1 - btc_p[i])) / 1.5
            if k > 0:
                c = btc_c[i]
                a = btc_a[i]
                hs = c * 0.95
                ast = c - 10 * a if not np.isnan(a) and a > 0 else hs
                ist = max(hs, ast)
                sd = min(c - ist, c * 0.05)
                if sd <= 0:
                    sd = c * 0.05
                tgt = min(total * 0.02 * btc_p[i] / sd * c, total * 0.4, free * 0.95)
                if tgt >= 10:
                    btc_u = tgt / c
                    btc_ep = c * (1 + fee)
                    btc_tr = ist
                    free -= tgt * (1 + fee)
                    n_trades += 1
                    active_dates.add(common[i].date())

        if sol_u == 0 and sol_p[i] >= sol_t:
            k = (sol_p[i] * 1.5 - (1 - sol_p[i])) / 1.5
            if k > 0:
                c = sol_c[i]
                a = sol_a[i]
                hs = c * 0.95
                ast = c - 10 * a if not np.isnan(a) and a > 0 else hs
                ist = max(hs, ast)
                sd = min(c - ist, c * 0.05)
                if sd <= 0:
                    sd = c * 0.05
                tgt = min(total * 0.02 * sol_p[i] / sd * c, total * 0.4, free * 0.95)
                if tgt >= 10:
                    sol_u = tgt / c
                    sol_ep = c * (1 + fee)
                    sol_tr = ist
                    free -= tgt * (1 + fee)
                    n_trades += 1
                    active_dates.add(common[i].date())

        port[j] = free + btc_u * btc_c[i] + sol_u * sol_c[i]
        if j > 0:
            rets[j] = port[j] / port[j - 1] - 1.0

    ret_s = pd.Series(rets, index=common[start:end])
    ret_pct = (port[-1] - 1_000_000) / 1_000_000 * 100
    std = ret_s.std()
    sharpe = float(qs.stats.sharpe(ret_s, periods=PERIODS)) if std > 1e-10 else 0
    sortino = float(qs.stats.sortino(ret_s, periods=PERIODS)) if std > 1e-10 else 0
    maxdd = float(qs.stats.max_drawdown(ret_s)) * 100 if std > 1e-10 else 0
    return {
        "sharpe": sharpe, "sortino": sortino, "ret": ret_pct,
        "maxdd": maxdd, "trades": n_trades, "active_days": len(active_dates),
    }


def main():
    data = load_data()
    n = data["n"]

    configs = [
        ("BTC=0.55 SOL=0.65", 0.55, 0.65),
        ("BTC=0.58 SOL=0.68", 0.58, 0.68),
        ("BTC=0.60 SOL=0.70", 0.60, 0.70),
        ("BTC=0.62 SOL=0.72", 0.62, 0.72),
        ("BTC=0.65 SOL=0.75", 0.65, 0.75),
        ("BTC=0.68 SOL=0.75", 0.68, 0.75),
        ("BTC=0.70 SOL=0.75", 0.70, 0.75),
    ]

    print("\n" + "=" * 120, flush=True)
    print("10-DAY SLIDING WINDOW ANALYSIS — XGBoost only (MR adds active days at ~0 P&L impact)")
    print("=" * 120)
    print(f"\n  {'Config':<22} {'MedSharpe':>10} {'MedSortino':>11} {'MedRet%':>8} "
          f"{'AvgTrades':>10} {'%Pos':>6} {'%Zero':>6} {'P25Ret%':>8} {'P75Ret%':>8} "
          f"{'AvgActive':>10}", flush=True)
    print("  " + "-" * 115)

    all_config_results = []

    for label, bt, st in configs:
        windows = []
        for s in range(0, n - WINDOW, STEP):
            r = run_window(data, s, bt, st)
            if r:
                windows.append(r)

        if not windows:
            continue

        sharpes = [w["sharpe"] for w in windows]
        sortinos = [w["sortino"] for w in windows]
        rets_list = [w["ret"] for w in windows]
        trades_list = [w["trades"] for w in windows]
        active_list = [w["active_days"] for w in windows]

        med_sharpe = np.median(sharpes)
        med_sortino = np.median(sortinos)
        med_ret = np.median(rets_list)
        avg_trades = np.mean(trades_list)
        pct_pos = sum(1 for r in rets_list if r > 0.01) / len(rets_list) * 100
        pct_zero = sum(1 for r in rets_list if abs(r) < 0.01) / len(rets_list) * 100
        p25 = np.percentile(rets_list, 25)
        p75 = np.percentile(rets_list, 75)
        avg_active = np.mean(active_list)

        print(f"  {label:<22} {med_sharpe:>10.3f} {med_sortino:>11.3f} {med_ret:>+7.2f}% "
              f"{avg_trades:>10.1f} {pct_pos:>5.0f}% {pct_zero:>5.0f}% {p25:>+7.2f}% {p75:>+7.2f}% "
              f"{avg_active:>10.1f}", flush=True)

        all_config_results.append({
            "label": label, "bt": bt, "st": st,
            "med_sharpe": med_sharpe, "med_sortino": med_sortino,
            "med_ret": med_ret, "avg_trades": avg_trades,
            "pct_pos": pct_pos, "pct_zero": pct_zero,
            "p25": p25, "p75": p75, "avg_active": avg_active,
            "windows": windows,
        })

    # Best by median Sortino
    print("\n" + "=" * 120)
    print("RECOMMENDATION")
    print("=" * 120)

    best = max(all_config_results, key=lambda x: x["med_sortino"])
    most_trades = max(all_config_results, key=lambda x: x["avg_trades"])
    best_winrate = max(all_config_results, key=lambda x: x["pct_pos"])

    print(f"\n  Best median Sortino:  {best['label']} (Sortino={best['med_sortino']:.3f}, "
          f"{best['avg_trades']:.1f} trades/10d, {best['pct_pos']:.0f}% positive windows)")
    print(f"  Most trades/window:   {most_trades['label']} ({most_trades['avg_trades']:.1f} trades/10d, "
          f"Sortino={most_trades['med_sortino']:.3f})")
    print(f"  Best win rate:        {best_winrate['label']} ({best_winrate['pct_pos']:.0f}% positive, "
          f"Sortino={best_winrate['med_sortino']:.3f})")

    print("\n  Key insight: MR activity layer adds 8-9 active days at 0.01x size (~$0 impact).")
    print("  The competition P&L is driven entirely by which XGBoost threshold you choose.")
    print("  Lower threshold = more trades per window = more consistent alpha = lower variance.")
    print("  Higher threshold = fewer trades = higher per-trade quality = higher variance.")

    # Distribution analysis for top 2
    print("\n  Return distribution (top 2 configs):")
    for cfg in sorted(all_config_results, key=lambda x: x["med_sortino"], reverse=True)[:2]:
        rets_list = [w["ret"] for w in cfg["windows"]]
        print(f"\n  {cfg['label']}:")
        print(f"    Min: {min(rets_list):+.2f}%  P10: {np.percentile(rets_list, 10):+.2f}%  "
              f"P25: {np.percentile(rets_list, 25):+.2f}%  Med: {np.median(rets_list):+.2f}%  "
              f"P75: {np.percentile(rets_list, 75):+.2f}%  P90: {np.percentile(rets_list, 90):+.2f}%  "
              f"Max: {max(rets_list):+.2f}%")
        # Trades distribution
        trades_list = [w["trades"] for w in cfg["windows"]]
        print(f"    Trades: min={min(trades_list)} med={np.median(trades_list):.0f} "
              f"max={max(trades_list)} zero_trade_windows={sum(1 for t in trades_list if t==0)}")


if __name__ == "__main__":
    main()

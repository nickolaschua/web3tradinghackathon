#!/usr/bin/env python3
"""Fast MR parameter sweep using pre-computed numpy arrays (no per-bar pandas)."""
import sys, pickle, datetime
import numpy as np, pandas as pd, quantstats as qs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.data.features import compute_features, compute_cross_asset_features, compute_btc_context_features

PERIODS = 35040
COINS = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","LINK","DOT",
         "LTC","UNI","NEAR","SUI","APT","PEPE","ARB","SHIB","FIL","HBAR"]
MR_COINS = [c for c in COINS if c not in ("BTC", "SOL")]
cutoff = pd.Timestamp("2024-01-01", tz="UTC")


def load_all_data():
    """Load and align all data into numpy arrays on a common index."""
    print("Loading raw data...")
    btc_raw = pd.read_parquet("data/BTCUSDT_15m.parquet")
    eth_raw = pd.read_parquet("data/ETHUSDT_15m.parquet")
    sol_raw = pd.read_parquet("data/SOLUSDT_15m.parquet")
    for df in (btc_raw, eth_raw, sol_raw):
        df.index = pd.to_datetime(df.index); df.columns = df.columns.str.lower()

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

    print("Loading models + predicting...")
    with open("models/xgb_btc_15m_iter5.pkl", "rb") as f: bm = pickle.load(f)
    with open("models/xgb_sol_15m.pkl", "rb") as f: sm = pickle.load(f)

    # Common index
    common = btc_feat.index.intersection(sol_feat.index)
    print("Preparing MR coin arrays...")
    mr_arrays = {}
    for coin in MR_COINS:
        df = pd.read_parquet(f"data/{coin}USDT_15m.parquet")
        df.index = pd.to_datetime(df.index); df.columns = df.columns.str.lower()
        feat = compute_features(df).dropna()
        feat = feat[feat.index >= cutoff]
        common = common.intersection(feat.index)
        mr_arrays[coin] = feat

    n = len(common)
    print(f"Common bars: {n}")

    # Pre-extract all numpy arrays on common index
    data = {
        "idx": common,
        "n": n,
        "btc_c": btc_feat.reindex(common)["close"].values,
        "sol_c": sol_feat.reindex(common)["close"].values,
        "btc_a": btc_feat.reindex(common)["atr_proxy"].values,
        "sol_a": sol_feat.reindex(common)["atr_proxy"].values,
        "btc_p": pd.Series(bm.predict_proba(btc_feat[list(bm.feature_names_in_)])[:, 1],
                           index=btc_feat.index).reindex(common).fillna(0).values,
        "sol_p": pd.Series(sm.predict_proba(sol_feat[list(sm.feature_names_in_)])[:, 1],
                           index=sol_feat.index).reindex(common).fillna(0).values,
        "mr": {},
    }
    for coin in MR_COINS:
        f = mr_arrays[coin].reindex(common)
        data["mr"][coin] = {
            "close": f["close"].values,
            "rsi": f["RSI_14"].fillna(50).values,
            "bb": f["bb_pos"].fillna(0.5).values,
            "e20": f["EMA_20"].fillna(0).values,
            "e50": f["EMA_50"].fillna(0).values,
        }
    return data


def run_config(data, entry_rsi=35, entry_bb=0.25, exit_rsi=50, exit_bb=0.55,
               regime_mult=0.5, size=0.03, max_pos=1, exclude=()):
    """Run hybrid backtest with given MR params. All numpy, no pandas in loop."""
    n = data["n"]
    idx = data["idx"]
    fee = 0.001
    initial = 1_000_000.0
    free = initial
    hwm = initial

    btc_c = data["btc_c"]; sol_c = data["sol_c"]
    btc_a = data["btc_a"]; sol_a = data["sol_a"]
    btc_p = data["btc_p"]; sol_p = data["sol_p"]

    # XGB state
    btc_u, btc_ep, btc_tr = 0.0, 0.0, 0.0
    sol_u, sol_ep, sol_tr = 0.0, 0.0, 0.0

    # MR state: parallel arrays
    active_coins = [c for c in MR_COINS if c not in exclude]
    mr_u = {}  # coin -> units
    mr_ep = {}  # coin -> entry price

    port = np.zeros(n)
    port[0] = initial
    rets = np.zeros(n)
    xgb_pnls = []
    mr_pnls = []
    active_dates = set()

    deep_offset = max(entry_rsi - 7, 20)

    for i in range(n):
        # XGB trail updates
        if btc_u > 0:
            a = btc_a[i]
            if not np.isnan(a) and a > 0:
                btc_tr = max(btc_tr, btc_c[i] - 10 * a)
        if sol_u > 0:
            a = sol_a[i]
            if not np.isnan(a) and a > 0:
                sol_tr = max(sol_tr, sol_c[i] - 10 * a)

        # Mark to market
        xv = btc_u * btc_c[i] + sol_u * sol_c[i]
        mv = sum(mr_u.get(c, 0) * data["mr"][c]["close"][i] for c in mr_u)
        total = free + xv + mv
        hwm = max(hwm, total)

        # XGB exits
        if btc_u > 0 and (btc_c[i] <= btc_tr or btc_p[i] <= 0.10):
            free += btc_u * btc_c[i] * (1 - fee)
            pnl = (btc_c[i] * (1 - fee) - btc_ep) / btc_ep if btc_ep > 0 else 0
            xgb_pnls.append(pnl)
            active_dates.add(idx[i].date())
            btc_u = 0.0
        if sol_u > 0 and (sol_c[i] <= sol_tr or sol_p[i] <= 0.10):
            free += sol_u * sol_c[i] * (1 - fee)
            pnl = (sol_c[i] * (1 - fee) - sol_ep) / sol_ep if sol_ep > 0 else 0
            xgb_pnls.append(pnl)
            active_dates.add(idx[i].date())
            sol_u = 0.0

        # MR exits
        to_close = []
        for c in list(mr_u.keys()):
            rsi = data["mr"][c]["rsi"][i]
            bb = data["mr"][c]["bb"][i]
            if rsi > exit_rsi or bb > exit_bb:
                p_close = data["mr"][c]["close"][i]
                free += mr_u[c] * p_close * (1 - fee)
                pnl = (p_close * (1 - fee) - mr_ep[c]) / mr_ep[c] if mr_ep[c] > 0 else 0
                mr_pnls.append(pnl)
                active_dates.add(idx[i].date())
                to_close.append(c)
        for c in to_close:
            del mr_u[c]; del mr_ep[c]

        # Recalc
        xv = btc_u * btc_c[i] + sol_u * sol_c[i]
        mv = sum(mr_u.get(c, 0) * data["mr"][c]["close"][i] for c in mr_u)
        total = free + xv + mv

        # XGB entries
        if btc_u == 0 and btc_p[i] >= 0.65:
            k = (btc_p[i] * 1.5 - (1 - btc_p[i])) / 1.5
            if k > 0:
                c = btc_c[i]; a = btc_a[i]
                hs = c * 0.95
                ast = c - 10 * a if not np.isnan(a) and a > 0 else hs
                ist = max(hs, ast)
                sd = min(c - ist, c * 0.05)
                if sd <= 0: sd = c * 0.05
                tgt = min(total * 0.02 * btc_p[i] / sd * c, total * 0.4, free * 0.95)
                if tgt >= 10:
                    btc_u = tgt / c; btc_ep = c * (1 + fee); btc_tr = ist
                    free -= tgt * (1 + fee)
                    active_dates.add(idx[i].date())

        if sol_u == 0 and sol_p[i] >= 0.75:
            k = (sol_p[i] * 1.5 - (1 - sol_p[i])) / 1.5
            if k > 0:
                c = sol_c[i]; a = sol_a[i]
                hs = c * 0.95
                ast = c - 10 * a if not np.isnan(a) and a > 0 else hs
                ist = max(hs, ast)
                sd = min(c - ist, c * 0.05)
                if sd <= 0: sd = c * 0.05
                tgt = min(total * 0.02 * sol_p[i] / sd * c, total * 0.4, free * 0.95)
                if tgt >= 10:
                    sol_u = tgt / c; sol_ep = c * (1 + fee); sol_tr = ist
                    free -= tgt * (1 + fee)
                    active_dates.add(idx[i].date())

        # MR entries
        if len(mr_u) < max_pos:
            for c in active_coins:
                if c in mr_u or len(mr_u) >= max_pos:
                    continue
                rsi = data["mr"][c]["rsi"][i]
                bb = data["mr"][c]["bb"][i]
                e20 = data["mr"][c]["e20"][i]
                e50 = data["mr"][c]["e50"][i]

                rm = 1.0 if e20 > e50 else regime_mult
                if rm == 0:
                    continue

                is_buy = (rsi < entry_rsi and bb < entry_bb) or (rsi < deep_offset)
                if not is_buy:
                    continue

                sz = size * rm
                p_entry = data["mr"][c]["close"][i]
                tgt = min(total * sz, free * 0.95)
                if tgt >= 10:
                    mr_u[c] = tgt / p_entry
                    mr_ep[c] = p_entry * (1 + fee)
                    free -= tgt * (1 + fee)
                    active_dates.add(idx[i].date())

        xv = btc_u * btc_c[i] + sol_u * sol_c[i]
        mv = sum(mr_u.get(c, 0) * data["mr"][c]["close"][i] for c in mr_u)
        port[i] = free + xv + mv
        if i > 0:
            rets[i] = port[i] / port[i - 1] - 1.0

    ret_s = pd.Series(rets, index=idx)
    total_days = (idx[-1] - idx[0]).days

    # Worst 10-day window
    all_cal = [idx[0].date() + datetime.timedelta(days=d) for d in range(total_days)]
    worst_10 = total_days
    for si in range(len(all_cal) - 10):
        w = all_cal[si:si + 10]
        worst_10 = min(worst_10, sum(1 for d in w if d in active_dates))

    n_mr = len(mr_pnls)
    return {
        "sharpe": float(qs.stats.sharpe(ret_s, periods=PERIODS)),
        "sortino": float(qs.stats.sortino(ret_s, periods=PERIODS)),
        "maxdd": float(qs.stats.max_drawdown(ret_s)) * 100,
        "ret": (port[-1] - initial) / initial * 100,
        "trades": len(xgb_pnls) + n_mr,
        "mr_trades": n_mr,
        "active_pct": len(active_dates) / total_days * 100,
        "worst_10d": worst_10,
        "mr_pnl": sum(mr_pnls) / n_mr * 100 if n_mr else 0,
        "mr_wr": sum(1 for p in mr_pnls if p > 0) / n_mr * 100 if n_mr else 0,
    }


def main():
    data = load_all_data()
    results = []

    def test(tag, **kw):
        r = run_config(data, **kw)
        results.append({"tag": tag, **r, "kw": kw})
        print(f"  {tag:<40} Sharpe={r['sharpe']:>6.3f}  Sort={r['sortino']:>6.3f}  "
              f"MR={r['mr_pnl']:>+5.2f}%  W10={r['worst_10d']}/10  "
              f"Tr={r['trades']:>5}  Act={r['active_pct']:>4.0f}%  DD={r['maxdd']:>5.1f}%",
              flush=True)

    print("\n" + "=" * 100)
    print("PHASE 1: INDIVIDUAL SWEEPS")
    print("=" * 100)

    print("\n--- Baseline ---", flush=True)
    test("BASELINE")

    print("\n--- Entry RSI ---", flush=True)
    for v in [30, 32, 33, 35, 38, 40]:
        test(f"entry_rsi={v}", entry_rsi=v)

    print("\n--- Entry bb_pos ---", flush=True)
    for v in [0.15, 0.20, 0.25, 0.30, 0.35]:
        test(f"entry_bb={v}", entry_bb=v)

    print("\n--- Exit RSI ---", flush=True)
    for v in [42, 45, 48, 50, 55]:
        test(f"exit_rsi={v}", exit_rsi=v)

    print("\n--- Exit bb_pos ---", flush=True)
    for v in [0.40, 0.45, 0.50, 0.55, 0.60]:
        test(f"exit_bb={v}", exit_bb=v)

    print("\n--- Regime mult ---", flush=True)
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        test(f"regime={v}", regime_mult=v)

    print("\n--- Size ---", flush=True)
    for v in [0.01, 0.02, 0.03, 0.05, 0.08]:
        test(f"size={v}", size=v)

    print("\n--- Max positions ---", flush=True)
    for v in [1, 2, 3]:
        test(f"max_pos={v}", max_pos=v)

    print("\n--- Coin exclusions ---", flush=True)
    test("excl LINK/AVAX/UNI", exclude=("LINK","AVAX","UNI"))
    test("excl 5 worst", exclude=("LINK","AVAX","UNI","XRP","PEPE"))
    test("excl 7 worst", exclude=("LINK","AVAX","UNI","XRP","PEPE","APT","BNB"))

    print("\n" + "=" * 100)
    print("PHASE 2: COMBINATIONS")
    print("=" * 100, flush=True)

    test("tight+fast", entry_rsi=32, entry_bb=0.20, exit_rsi=45, exit_bb=0.50)
    test("med+fast", entry_rsi=33, entry_bb=0.25, exit_rsi=45, exit_bb=0.50)
    test("med+medfast", entry_rsi=33, entry_bb=0.20, exit_rsi=48, exit_bb=0.50)
    test("tight+excl3", entry_rsi=32, entry_bb=0.20, exclude=("LINK","AVAX","UNI"))
    test("med+excl3", entry_rsi=33, entry_bb=0.25, exclude=("LINK","AVAX","UNI"))
    test("tight+fast+excl3", entry_rsi=32, entry_bb=0.20, exit_rsi=45, exit_bb=0.50, exclude=("LINK","AVAX","UNI"))
    test("med+fast+excl3", entry_rsi=33, entry_bb=0.25, exit_rsi=45, exit_bb=0.50, exclude=("LINK","AVAX","UNI"))
    test("tight+fast+r0.25", entry_rsi=32, entry_bb=0.20, exit_rsi=45, regime_mult=0.25)
    test("med+fast+r0.25", entry_rsi=33, entry_bb=0.25, exit_rsi=45, regime_mult=0.25)
    test("med+fast+sz0.02+mp2", entry_rsi=33, entry_bb=0.25, exit_rsi=45, size=0.02, max_pos=2)
    test("tight+fast+sz0.02+mp2", entry_rsi=32, entry_bb=0.20, exit_rsi=45, size=0.02, max_pos=2)
    test("tight+fast+r0.25+excl3", entry_rsi=32, entry_bb=0.20, exit_rsi=45, exit_bb=0.50, regime_mult=0.25, exclude=("LINK","AVAX","UNI"))
    test("med+fast+sz0.02+excl3", entry_rsi=33, entry_bb=0.25, exit_rsi=45, exit_bb=0.50, size=0.02, exclude=("LINK","AVAX","UNI"))
    test("tight+fast+sz0.02+excl3", entry_rsi=32, entry_bb=0.20, exit_rsi=45, size=0.02, exclude=("LINK","AVAX","UNI"))
    test("tight+fast+r0.25+sz0.02", entry_rsi=32, entry_bb=0.20, exit_rsi=45, regime_mult=0.25, size=0.02)
    test("med+fast+r0.25+excl3", entry_rsi=33, entry_bb=0.25, exit_rsi=45, regime_mult=0.25, exclude=("LINK","AVAX","UNI"))

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("TOP 10 BY SORTINO (must pass worst_10d >= 8)")
    print("=" * 100)

    passing = [r for r in results if r["worst_10d"] >= 8]
    passing.sort(key=lambda r: r["sortino"], reverse=True)

    print(f"\n{'#':>3} {'Tag':<40} {'Sharpe':>7} {'Sortino':>8} {'MR PnL':>7} {'W10':>4} {'Tr':>5} {'Act%':>5} {'Ret%':>7} {'DD%':>6}")
    print("-" * 105)
    for i, r in enumerate(passing[:10]):
        print(f"  {i+1:>2} {r['tag']:<40} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['mr_pnl']:>+6.2f}% {r['worst_10d']:>3}/10 {r['trades']:>5} {r['active_pct']:>4.0f}% {r['ret']:>+6.1f}% {r['maxdd']:>5.1f}%")

    # Competition score
    print(f"\n{'#':>3} {'Tag':<40} {'Score':>7} {'Sortino':>8} {'Sharpe':>7} {'Calmar':>7}")
    print("-" * 80)
    for i, r in enumerate(passing[:5]):
        calmar = abs(r["ret"] / r["maxdd"]) if r["maxdd"] != 0 else 0
        score = 0.4 * r["sortino"] + 0.3 * r["sharpe"] + 0.3 * calmar
        print(f"  {i+1:>2} {r['tag']:<40} {score:>7.3f} {r['sortino']:>8.3f} {r['sharpe']:>7.3f} {calmar:>7.3f}")

    if not passing:
        print("\n  No configs pass 8/10! Top 5 by Sortino regardless:")
        results.sort(key=lambda r: r["sortino"], reverse=True)
        for i, r in enumerate(results[:5]):
            print(f"  {i+1} {r['tag']:<40} Sort={r['sortino']:.3f} W10={r['worst_10d']}/10")


if __name__ == "__main__":
    main()

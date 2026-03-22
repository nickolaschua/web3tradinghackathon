#!/usr/bin/env python3
"""Compare 5-coin (20% each) vs 3-coin (33% each) with correct allocation."""
import pickle, numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.data.features import compute_features, compute_cross_asset_features, compute_btc_context_features

CAPITAL = 1_000_000
COMM = 0.001
HARD_STOP = 0.05
ATR_MULT = 10.0
TRAIL_MULT = 10.0
RISK_PCT = 0.02
MAX_POS = 5
WIN_LOSS = 1.5
EXIT_TH = 0.08
PERIODS = 35040
CUTOFF = "2024-01-01"

def cb_mult(dd):
    if dd >= 0.30: return 0.0
    if dd >= 0.20: return 0.25
    if dd >= 0.10: return 0.5
    return 1.0

def load(coin):
    df = pd.read_parquet(f"data/{coin}USDT_15m.parquet")
    df.columns = df.columns.str.lower()
    return df

def prep_feat(coin, btc, eth, sol):
    raw = load(coin)
    feat = compute_features(raw)
    cross = {}
    if coin != "ETH": cross["ETH/USD"] = eth
    if coin != "SOL": cross["SOL/USD"] = sol
    if coin != "BTC": cross["BTC/USD"] = btc
    keys = list(cross.keys())[:2]
    feat = compute_cross_asset_features(feat, {k: cross[k] for k in keys})
    for a, df in [("eth", eth), ("sol", sol)]:
        lr = np.log(df["close"] / df["close"].shift(1))
        feat[f"{a}_return_4h"] = lr.shift(16).reindex(feat.index)
        feat[f"{a}_return_1d"] = lr.shift(96).reindex(feat.index)
    feat = compute_btc_context_features(feat, eth, sol, window=2880)
    return feat.dropna()

def run_combined(coins_cfg, all_models, all_feats, all_closes):
    """coins_cfg: {coin: (threshold, weight)}"""
    idx = None
    for coin in coins_cfg:
        f = all_feats[coin][CUTOFF:]
        idx = f.index if idx is None else idx.intersection(f.index)

    probas = {}
    for coin in coins_cfg:
        model = all_models[coin]
        fcols = list(model.feature_names_in_)
        feat = all_feats[coin].reindex(idx).copy()
        for c in fcols:
            if c not in feat.columns: feat[c] = np.nan
        probas[coin] = model.predict_proba(feat[fcols])[:, 1]

    closes = {}
    for coin in coins_cfg:
        closes[coin] = all_closes[coin].reindex(idx).ffill().values

    cash = float(CAPITAL)
    hwm = float(CAPITAL)
    positions = {}
    trades = []
    equity = []

    for i in range(len(idx)):
        pv = cash
        for coin, pos in positions.items():
            pv += pos["qty"] * closes[coin][i]
        hwm = max(hwm, pv)
        dd = (hwm - pv) / hwm if hwm > 0 else 0
        cbm = cb_mult(dd)

        # Stops
        for coin in list(positions.keys()):
            pos = positions[coin]
            price = closes[coin][i]
            nt = price - TRAIL_MULT * pos["atr"]
            if nt > pos["trail"]: pos["trail"] = nt
            if price <= pos["stop"] or price <= pos["trail"]:
                pnl = (price - pos["entry"]) * pos["qty"]
                comm = price * pos["qty"] * COMM
                cash += price * pos["qty"] - comm
                trades.append({"coin": coin, "pnl": pnl - comm, "side": "SELL", "reason": "stop"})
                del positions[coin]

        for coin, (threshold, weight) in coins_cfg.items():
            price = closes[coin][i]
            if np.isnan(price) or price <= 0: continue
            p = probas[coin][i]

            if coin in positions and p <= EXIT_TH:
                pos = positions[coin]
                pnl = (price - pos["entry"]) * pos["qty"]
                comm = price * pos["qty"] * COMM
                cash += price * pos["qty"] - comm
                trades.append({"coin": coin, "pnl": pnl - comm, "side": "SELL", "reason": "signal"})
                del positions[coin]

            if coin not in positions and p >= threshold and cbm > 0 and len(positions) < MAX_POS:
                feat_row = all_feats[coin].reindex(idx).iloc[i]
                atr = feat_row.get("atr_proxy", price * 0.02)
                if pd.isna(atr) or atr <= 0: atr = price * 0.02

                hs = price * (1 - HARD_STOP)
                ast = price - ATR_MULT * atr
                init_stop = max(hs, ast)
                sd = price - init_stop
                sd = min(sd, price * HARD_STOP)  # canonical cap
                if sd <= 0: sd = price * HARD_STOP

                kelly = (p * WIN_LOSS - (1 - p)) / WIN_LOSS
                if kelly <= 0: continue

                risk_usd = pv * RISK_PCT * p * cbm * weight
                qty = risk_usd / sd
                target = qty * price
                target = min(target, pv * 0.40, cash * 0.95)
                if target < 10: continue
                qty = target / price

                comm = target * COMM
                cash -= target + comm
                positions[coin] = {"qty": qty, "entry": price, "stop": hs, "trail": init_stop, "atr": atr}
                trades.append({"coin": coin, "pnl": -comm, "side": "BUY", "reason": "entry"})

        pv_end = cash
        for coin, pos in positions.items():
            pv_end += pos["qty"] * closes[coin][i]
        equity.append(pv_end)

    eq = pd.Series(equity, index=idx)
    rets = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    sharpe = (rets.mean() / rets.std() * np.sqrt(PERIODS)) if rets.std() > 0 else 0
    ds = rets[rets < 0].std() * np.sqrt(PERIODS)
    sortino = (rets.mean() * PERIODS / ds) if ds > 0 else 0
    max_dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    calmar = (rets.mean() * PERIODS / abs(max_dd / 100)) if max_dd != 0 else 0
    comp = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    buys = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if t["side"] == "SELL"]
    wins = len([t for t in sells if t["pnl"] > 0])
    losses = len([t for t in sells if t["pnl"] <= 0])
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    coin_counts = {}
    for t in buys:
        coin_counts[t["coin"]] = coin_counts.get(t["coin"], 0) + 1

    return {
        "eq": eq, "total_ret": total_ret, "sharpe": sharpe, "sortino": sortino,
        "calmar": calmar, "max_dd": max_dd, "comp": comp,
        "n_trades": len(buys), "wr": wr, "wins": wins, "losses": losses,
        "coin_counts": coin_counts,
    }


def print_results(name, r):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Return:    {r['total_ret']:+.2f}%")
    print(f"Sharpe:    {r['sharpe']:.3f}")
    print(f"Sortino:   {r['sortino']:.3f}")
    print(f"Calmar:    {r['calmar']:.3f}")
    print(f"MaxDD:     {r['max_dd']:.2f}%")
    print(f"CompScore: {r['comp']:.3f}")
    print(f"Trades:    {r['n_trades']} ({r['wr']:.1f}% win, {r['wins']}W/{r['losses']}L)")
    for coin, cnt in sorted(r["coin_counts"].items(), key=lambda x: -x[1]):
        print(f"  {coin}: {cnt}")

    eq = r["eq"]
    b10 = 10 * 24 * 4
    if len(eq) > b10:
        print(f"\n10-day windows:")
        for off in range(5):
            ei = len(eq) - 1 - off * b10
            si = ei - b10
            if si < 0: break
            w = eq.iloc[si:ei]
            wr2 = (w.iloc[-1] / w.iloc[0] - 1) * 100
            wrets = w.pct_change().dropna()
            wsh = (wrets.mean() / wrets.std() * np.sqrt(PERIODS)) if wrets.std() > 0 else 0
            wdd = ((w - w.cummax()) / w.cummax()).min() * 100
            print(f"  {w.index[0].strftime('%m-%d')} -> {w.index[-1].strftime('%m-%d')}: "
                  f"ret={wr2:+.2f}% sharpe={wsh:.2f} dd={wdd:.2f}%")


def main():
    print("Loading data and models...")
    btc_raw, eth_raw, sol_raw = load("BTC"), load("ETH"), load("SOL")

    all_models, all_feats, all_closes = {}, {}, {}
    for coin in ["BTC", "SOL", "ETH", "XRP", "DOGE"]:
        mp = "models/xgb_btc_15m_iter5.pkl" if coin == "BTC" else f"models/xgb_{coin.lower()}_15m.pkl"
        with open(mp, "rb") as f:
            all_models[coin] = pickle.load(f)
        all_feats[coin] = prep_feat(coin, btc_raw, eth_raw, sol_raw)
        all_closes[coin] = load(coin)["close"]
    print("Done loading.\n")

    # Config 1: 5-coin 20% each
    r5 = run_combined(
        {"BTC": (0.65, 0.20), "SOL": (0.70, 0.20), "ETH": (0.65, 0.20),
         "XRP": (0.65, 0.20), "DOGE": (0.65, 0.20)},
        all_models, all_feats, all_closes,
    )
    print_results("5-COIN (BTC+SOL+ETH+XRP+DOGE) @ 20% each", r5)

    # Config 2: 3-coin 33% each
    r3 = run_combined(
        {"BTC": (0.65, 0.333), "SOL": (0.70, 0.333), "XRP": (0.65, 0.333)},
        all_models, all_feats, all_closes,
    )
    print_results("3-COIN (BTC+SOL+XRP) @ 33% each", r3)

    # Config 3: BTC-only 100%
    r1 = run_combined(
        {"BTC": (0.65, 1.0)},
        all_models, all_feats, all_closes,
    )
    print_results("BTC-ONLY @ 100%", r1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Alpha Research Pipeline — Feature Discovery for Crypto Trading Hackathon

Usage:
    python alpha_research.py                    # full pipeline
    python alpha_research.py --skip-download    # reuse cached data
    python alpha_research.py --quick            # fewer CV folds (faster)
"""
import argparse, json, time, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")

DATA_DIR = Path("research_data")
RESULTS_DIR = Path("research_results")
OHLCV_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "PAXGUSDT"]
FUNDING_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="aucpr", early_stopping_rounds=30,
    random_state=42, n_jobs=-1, verbosity=0)
LABEL_HORIZON = 6
LABEL_THRESHOLD = 0.01
BASE_FEATURE_COLS = ["atr_proxy","RSI_14","MACD_12_26_9","MACDs_12_26_9","MACDh_12_26_9",
    "EMA_20","EMA_50","ema_slope","eth_return_lag1","eth_return_lag2","sol_return_lag1","sol_return_lag2"]

# === DATA DOWNLOAD ===
def download_binance_klines(symbol, interval="4h", start_date="2022-01-01"):
    import requests
    base_url = "https://api.binance.com/api/v3/klines"
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_rows, current = [], start_ms
    while current < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": current, "endTime": end_ms, "limit": 1000}
        for attempt in range(3):
            try:
                resp = requests.get(base_url, params=params, timeout=15); resp.raise_for_status(); batch = resp.json(); break
            except: 
                if attempt == 2: raise
                time.sleep(2 ** attempt)
        if not batch: break
        all_rows.extend(batch)
        if len(batch) < 1000: break
        current = int(batch[-1][0]) + 1; time.sleep(0.1)
    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame({"open":[float(r[1]) for r in all_rows],"high":[float(r[2]) for r in all_rows],
        "low":[float(r[3]) for r in all_rows],"close":[float(r[4]) for r in all_rows],
        "volume":[float(r[5]) for r in all_rows],"quote_volume":[float(r[7]) for r in all_rows],
        "n_trades":[int(r[8]) for r in all_rows],"taker_buy_volume":[float(r[9]) for r in all_rows]})
    df.index = pd.to_datetime([int(r[0]) for r in all_rows], unit="ms", utc=True); df.index.name = "timestamp"
    return df

def download_binance_funding_rates(symbol, start_date="2022-01-01"):
    import requests
    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_rows, current = [], start_ms
    while current < end_ms:
        params = {"symbol": symbol, "startTime": current, "endTime": end_ms, "limit": 1000}
        for attempt in range(3):
            try:
                resp = requests.get(base_url, params=params, timeout=15); resp.raise_for_status(); batch = resp.json(); break
            except:
                if attempt == 2: raise
                time.sleep(2 ** attempt)
        if not batch: break
        all_rows.extend(batch)
        if len(batch) < 1000: break
        current = int(batch[-1]["fundingTime"]) + 1; time.sleep(0.1)
    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame({"funding_rate": [float(r["fundingRate"]) for r in all_rows]})
    df.index = pd.to_datetime([int(r["fundingTime"]) for r in all_rows], unit="ms", utc=True); df.index.name = "timestamp"
    return df

def download_macro_data():
    import yfinance as yf
    result = {}
    for ticker, name in [("CL=F", "oil"), ("DX-Y.NYB", "dxy")]:
        try:
            data = yf.download(ticker, start="2022-01-01", interval="1d", progress=False)
            if len(data) > 0:
                df = data[["Close"]].copy(); df.columns = ["close"]
                if df.index.tz is None: df.index = df.index.tz_localize("UTC")
                result[name] = df; print(f"  {name}: {len(df)} daily bars")
        except Exception as e: print(f"  {name}: failed - {e}")
    return result

def download_all_data(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for symbol in OHLCV_PAIRS:
        path = data_dir / f"{symbol}_4h.parquet"
        if path.exists(): print(f"  {symbol} OHLCV: cached ({len(pd.read_parquet(path))} bars)"); continue
        print(f"  {symbol} OHLCV: downloading...", end=" ", flush=True)
        df = download_binance_klines(symbol, "4h")
        if len(df) > 0: df.to_parquet(path); print(f"{len(df)} bars")
        else: print("FAILED")
    for symbol in FUNDING_PAIRS:
        path = data_dir / f"{symbol}_funding.parquet"
        if path.exists(): print(f"  {symbol} funding: cached ({len(pd.read_parquet(path))} rows)"); continue
        print(f"  {symbol} funding: downloading...", end=" ", flush=True)
        df = download_binance_funding_rates(symbol)
        if len(df) > 0: df.to_parquet(path); print(f"{len(df)} rows")
        else: print("FAILED")
    print("  Macro (oil, DXY)..."); macro = download_macro_data()
    for name, df in macro.items(): df.to_parquet(data_dir / f"{name}_daily.parquet")

# === FEATURE ENGINEERING ===
def compute_base_features(df):
    import pandas_ta_classic as ta
    out = df.copy()
    log_ret = np.log(out["close"] / out["close"].shift(1))
    out["atr_proxy"] = log_ret.rolling(14).std() * out["close"] * 1.25
    out.ta.rsi(length=14, append=True); out.ta.macd(fast=12, slow=26, signal=9, append=True)
    out.ta.ema(length=20, append=True); out.ta.ema(length=50, append=True)
    out["ema_slope"] = (out["EMA_20"] - out["EMA_20"].shift(1)) / out["EMA_20"].shift(1)
    ohlcv = {"open","high","low","close","volume","quote_volume","n_trades","taker_buy_volume"}
    out[[c for c in out.columns if c not in ohlcv]] = out[[c for c in out.columns if c not in ohlcv]].shift(1)
    return out

def compute_cross_asset_lags(btc_df, other_dfs):
    out = btc_df.copy()
    for pair, df in other_dfs.items():
        df = df.copy(); df.columns = df.columns.str.lower()
        prefix = pair.replace("USDT", "").lower()
        log_ret = np.log(df["close"] / df["close"].shift(1))
        out[f"{prefix}_return_lag1"] = log_ret.shift(1).reindex(out.index)
        out[f"{prefix}_return_lag2"] = log_ret.shift(2).reindex(out.index)
    return out

def compute_funding_features(btc_df, funding_dfs):
    out = btc_df.copy()
    for symbol, fdf in funding_dfs.items():
        prefix = symbol.replace("USDT", "").lower()
        fr = fdf["funding_rate"].copy(); fr_4h = fr.resample("4h").ffill()
        aligned = fr_4h.reindex(out.index, method="ffill")
        out[f"{prefix}_funding_rate"] = aligned.shift(1)
        out[f"{prefix}_funding_ma8"] = aligned.rolling(8).mean().shift(1)
        rm, rs = aligned.rolling(90).mean(), aligned.rolling(90).std()
        out[f"{prefix}_funding_zscore"] = ((aligned - rm) / rs.replace(0, np.nan)).shift(1)
        out[f"{prefix}_funding_cum24h"] = aligned.rolling(6).sum().shift(1)
    return out

def compute_macro_features(btc_df, macro_dfs):
    out = btc_df.copy()
    for name, mdf in macro_dfs.items():
        close = mdf["close"].copy(); close_4h = close.resample("4h").ffill()
        aligned = close_4h.reindex(out.index, method="ffill")
        out[f"{name}_return_1d"] = aligned.pct_change(6).shift(1)
        out[f"{name}_return_5d"] = aligned.pct_change(30).shift(1)
        out[f"{name}_vol_5d"] = aligned.pct_change().rolling(30).std().shift(1)
        r1d = aligned.pct_change(6); out[f"{name}_acceleration"] = (r1d - r1d.shift(6)).shift(1)
    return out

def compute_paxg_features(btc_df, paxg_df):
    out = btc_df.copy(); paxg = paxg_df.copy(); paxg.columns = paxg.columns.str.lower()
    paxg_ret = np.log(paxg["close"] / paxg["close"].shift(1))
    btc_ret = np.log(out["close"] / out["close"].shift(1))
    out["paxg_return_lag1"] = paxg_ret.shift(1).reindex(out.index)
    out["paxg_return_lag2"] = paxg_ret.shift(2).reindex(out.index)
    out["btc_paxg_spread_lag1"] = (btc_ret - paxg_ret.reindex(out.index)).shift(1)
    out["btc_paxg_corr_20"] = btc_ret.rolling(20).corr(paxg_ret.reindex(out.index)).shift(1)
    return out

def compute_volume_features(btc_df):
    out = btc_df.copy()
    if "quote_volume" in out.columns and out["quote_volume"].sum() > 0:
        qv = out["quote_volume"]; out["volume_ratio"] = (qv / qv.rolling(20).mean()).shift(1)
        if "taker_buy_volume" in out.columns:
            out["taker_buy_ratio"] = (out["taker_buy_volume"] / out["volume"].replace(0, np.nan)).shift(1)
        out["volume_trend"] = (qv.rolling(6).mean() / qv.rolling(24).mean()).shift(1)
    return out

def compute_cross_sectional_momentum(btc_df, all_dfs):
    out = btc_df.copy()
    rets = {s: d.copy().pipe(lambda d: d.set_axis(d.columns.str.lower(), axis=1))["close"].pct_change(6) for s, d in all_dfs.items()}
    if len(rets) < 2: return out
    ret_df = pd.DataFrame(rets); ranks = ret_df.rank(axis=1, ascending=False, pct=True)
    if "BTCUSDT" in ranks.columns: out["btc_momentum_rank"] = ranks["BTCUSDT"].reindex(out.index).shift(1)
    if "BTCUSDT" in ret_df.columns:
        out["btc_excess_return_6bar"] = (ret_df["BTCUSDT"] - ret_df.median(axis=1)).reindex(out.index).shift(1)
    return out

# === CV ENGINE ===
def run_cv(X, y, n_splits=5, label=""):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=24); folds = []
    for fold, (ti, vi) in enumerate(tscv.split(X)):
        Xt, Xv, yt, yv = X.iloc[ti], X.iloc[vi], y.iloc[ti], y.iloc[vi]
        np_, nn = int(yt.sum()), len(yt)-int(yt.sum())
        if np_ == 0 or nn == 0: continue
        m = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": nn/np_})
        m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        p = m.predict_proba(Xv)[:, 1]
        if len(set(yv)) < 2: continue
        folds.append({"ap": average_precision_score(yv, p), "f1": f1_score(yv, (p>=0.5).astype(int), zero_division=0)})
    if not folds: return {"label":label,"mean_ap":0.0,"mean_f1":0.0,"n_folds":0}
    return {"label":label, "mean_ap":round(float(np.mean([f["ap"] for f in folds])),4),
            "mean_f1":round(float(np.mean([f["f1"] for f in folds])),4), "n_folds":len(folds)}

def test_feature_set(X_all, y, candidate_cols, label, n_splits=5):
    valid = X_all.index
    for col in candidate_cols:
        if col in X_all.columns: valid = valid[X_all[col].reindex(valid).notna()]
    Xv, yv = X_all.loc[valid], y.loc[valid]
    if len(Xv) < 500: return {"label":label,"candidate_cols":candidate_cols,"baseline_ap":0.0,"augmented_ap":0.0,"delta_ap":0.0,"verdict":"SKIP","n_rows":len(Xv)}
    bc = [c for c in BASE_FEATURE_COLS if c in Xv.columns]
    baseline = run_cv(Xv[bc], yv, n_splits, "base")
    ac = bc + [c for c in candidate_cols if c in Xv.columns]
    augmented = run_cv(Xv[ac], yv, n_splits, "aug")
    d = augmented["mean_ap"] - baseline["mean_ap"]
    v = "STRONG" if d > 0.015 else "MODERATE" if d > 0.005 else "WEAK" if d > 0 else "NO IMPROVEMENT"
    return {"label":label,"candidate_cols":candidate_cols,"baseline_ap":baseline["mean_ap"],
            "augmented_ap":augmented["mean_ap"],"delta_ap":round(d,4),"baseline_f1":baseline["mean_f1"],
            "augmented_f1":augmented["mean_f1"],"verdict":v,"n_rows":len(Xv)}

# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--threshold", type=float, default=LABEL_THRESHOLD)
    parser.add_argument("--horizon", type=int, default=LABEL_HORIZON)
    args = parser.parse_args()
    n_splits = 3 if args.quick else 5

    print("="*70); print("  ALPHA RESEARCH PIPELINE"); print("="*70)

    print("\n[1/6] Downloading data...")
    if not args.skip_download: download_all_data(DATA_DIR)
    else: print("  Skipped")

    print("\n[2/6] Loading data...")
    ohlcv = {}
    for s in OHLCV_PAIRS:
        p = DATA_DIR / f"{s}_4h.parquet"
        if p.exists(): ohlcv[s] = pd.read_parquet(p); print(f"  {s}: {len(ohlcv[s])} bars")
    if "BTCUSDT" not in ohlcv: print("FATAL: BTC data missing"); return
    funding = {}
    for s in FUNDING_PAIRS:
        p = DATA_DIR / f"{s}_funding.parquet"
        if p.exists(): funding[s] = pd.read_parquet(p); print(f"  {s} funding: {len(funding[s])} rows")
    macro = {}
    for n in ["oil","dxy"]:
        p = DATA_DIR / f"{n}_daily.parquet"
        if p.exists(): macro[n] = pd.read_parquet(p); print(f"  {n}: {len(macro[n])} daily bars")

    print("\n[3/6] Computing features...")
    btc = ohlcv["BTCUSDT"]; feat = compute_base_features(btc)
    feat = compute_cross_asset_lags(feat, {s: ohlcv[s] for s in ["ETHUSDT","SOLUSDT"] if s in ohlcv})
    if funding: feat = compute_funding_features(feat, funding)
    if macro: feat = compute_macro_features(feat, macro)
    if "PAXGUSDT" in ohlcv: feat = compute_paxg_features(feat, ohlcv["PAXGUSDT"])
    feat = compute_volume_features(feat)
    feat = compute_cross_sectional_momentum(feat, ohlcv)
    print(f"  Total: {feat.shape}")

    print(f"\n[4/6] Labels (horizon={args.horizon}, threshold={args.threshold})...")
    fwd_ret = feat["close"].shift(-args.horizon) / feat["close"] - 1
    labels = (fwd_ret >= args.threshold).astype(int)
    X_all = feat.iloc[:-args.horizon].copy(); y_all = labels.iloc[:-args.horizon].copy()
    bv = X_all[BASE_FEATURE_COLS].dropna().index; X_all = X_all.loc[bv]; y_all = y_all.loc[bv]
    br = y_all.mean(); print(f"  Bars: {len(X_all)} | BUY rate: {br:.1%}")
    if br < 0.10 or br > 0.50: print(f"  WARNING: BUY rate outside [10%, 50%]")

    print(f"\n[5/6] Testing feature groups ({n_splits}-fold CV)...\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); results = []
    groups = {
        "BTC Funding": [c for c in X_all.columns if c.startswith("btc_funding")],
        "ETH Funding": [c for c in X_all.columns if c.startswith("eth_funding")],
        "All Funding": [c for c in X_all.columns if "funding" in c],
        "Oil (WTI)": [c for c in X_all.columns if c.startswith("oil_")],
        "DXY": [c for c in X_all.columns if c.startswith("dxy_")],
        "Oil + DXY": [c for c in X_all.columns if c.startswith("oil_") or c.startswith("dxy_")],
        "PAXG (Gold)": [c for c in X_all.columns if "paxg" in c],
        "Volume": [c for c in X_all.columns if c in ("volume_ratio","taker_buy_ratio","volume_trend")],
        "Cross-Sect Mom": [c for c in X_all.columns if c in ("btc_momentum_rank","btc_excess_return_6bar")],
    }
    for gn, cols in groups.items():
        if not cols: print(f"  {gn}: no features, skipping"); continue
        print(f"  Testing: {gn} ({len(cols)} feat)...", end=" ", flush=True)
        r = test_feature_set(X_all, y_all, cols, gn, n_splits); results.append(r)
        print(f"dAP: {r['delta_ap']:+.4f} - {r['verdict']}")
    all_cand = list(set(c for cols in groups.values() for c in cols))
    if all_cand:
        print(f"  Testing: Kitchen Sink ({len(all_cand)} feat)...", end=" ", flush=True)
        r = test_feature_set(X_all, y_all, all_cand, "Kitchen Sink", n_splits); results.append(r)
        print(f"dAP: {r['delta_ap']:+.4f} - {r['verdict']}")

    print("\n"+"="*70); print("  RESULTS - Ranked by Delta AP"); print("="*70)
    results.sort(key=lambda r: r["delta_ap"], reverse=True)
    print(f"\n  {'Group':<25} {'dAP':>8} {'Base':>8} {'Aug':>8} {'Verdict'}")
    print("  "+"-"*65)
    for r in results:
        m = " ***" if r["delta_ap"] > 0.005 else ""
        print(f"  {r['label']:<25} {r['delta_ap']:>+8.4f} {r['baseline_ap']:>8.4f} {r['augmented_ap']:>8.4f} {r['verdict']}{m}")
    winners = [r for r in results if r["delta_ap"] > 0.005]
    if winners:
        print(f"\n  RECOMMENDED:"); 
        for w in winners: print(f"    {w['label']}: {w['candidate_cols']}")
    else: print(f"\n  No group improved AP by > 0.005")
    with open(RESULTS_DIR / "results.json", "w") as f: json.dump(results, f, indent=2, default=str)

    print("\n"+"="*70); print("  BONUS - Threshold Sweep"); print("="*70)
    bc = [c for c in BASE_FEATURE_COLS if c in X_all.columns]
    for tau in [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03]:
        ls = ((feat["close"].shift(-args.horizon)/feat["close"]-1) >= tau).astype(int)
        ys = ls.iloc[:-args.horizon].loc[bv]; brs = ys.mean()
        if brs < 0.05 or brs > 0.60: print(f"  tau={tau:.3f}: BUY={brs:.1%} SKIP"); continue
        cv = run_cv(X_all[bc], ys, n_splits, f"tau={tau}")
        print(f"  tau={tau:.3f}: BUY={brs:.1%}, AP={cv['mean_ap']:.4f}, F1={cv['mean_f1']:.4f}")
    print("\nDone.")

if __name__ == "__main__": main()

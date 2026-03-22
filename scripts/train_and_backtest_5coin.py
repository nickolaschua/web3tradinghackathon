#!/usr/bin/env python3
"""
Train XGBoost models for ETH, XRP, DOGE using the same pipeline as BTC/SOL.
Then backtest all 5 models individually and combined.

Usage:
  python scripts/train_and_backtest_5coin.py
"""
import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_features, compute_cross_asset_features, compute_btc_context_features

# ── Training config (same as train_model_15m.py) ────────────────────────────
HORIZON_15M = 16
CV_GAP = 64
CV_SPLITS = 8
PERIODS_15M = 35_040
TRAIN_CUTOFF = "2024-01-01"

FEATURE_COLS = [
    "atr_proxy", "RSI_14", "RSI_7",
    "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    "bb_width", "bb_pos",
    "volume_ratio", "candle_body",
    "eth_return_4h", "sol_return_4h",
    "eth_return_1d", "sol_return_1d",
    "eth_btc_corr", "eth_btc_beta",
]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)

# ── Backtest config ──────────────────────────────────────────────────────────
CAPITAL = 1_000_000
COMMISSION_PCT = 0.001
HARD_STOP_PCT = 0.05
ATR_STOP_MULT = 10.0
TRAILING_STOP_MULT = 10.0
RISK_PER_TRADE = 0.02
MAX_POSITIONS = 5
EXPECTED_WIN_LOSS = 1.5
CB_LIGHT = 0.10
CB_HEAVY = 0.20
CB_HALT = 0.30

COINS_TO_TRAIN = ["ETH", "XRP", "DOGE"]
ALL_COINS = ["BTC", "SOL", "ETH", "XRP", "DOGE"]

THRESHOLDS = {
    "BTC": 0.65, "SOL": 0.70,
    "ETH": 0.65, "XRP": 0.65, "DOGE": 0.65,
}
EXIT_THRESHOLD = 0.08


def load_parquet(coin: str) -> pd.DataFrame:
    path = Path(f"data/{coin}USDT_15m.parquet")
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    df = pd.read_parquet(path)
    df.columns = df.columns.str.lower()
    return df


def prepare_coin_features(coin: str, btc_raw, eth_raw, sol_raw) -> pd.DataFrame:
    """Prepare features for any coin using the same pipeline as BTC."""
    coin_raw = load_parquet(coin)

    feat = compute_features(coin_raw)

    # Cross-asset features: use the OTHER two major coins
    cross = {}
    if coin != "ETH":
        cross["ETH/USD"] = eth_raw
    if coin != "SOL":
        cross["SOL/USD"] = sol_raw
    if coin != "BTC":
        cross["BTC/USD"] = btc_raw
    # Only use first 2 to match the 2-asset pattern
    cross_keys = list(cross.keys())[:2]
    cross = {k: cross[k] for k in cross_keys}
    feat = compute_cross_asset_features(feat, cross)

    # Add 4H and 1D cross-asset lags
    for asset_name, asset_df in [("eth", eth_raw), ("sol", sol_raw)]:
        log_ret = np.log(asset_df["close"] / asset_df["close"].shift(1))
        feat[f"{asset_name}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset_name}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    # BTC context features
    feat = compute_btc_context_features(feat, eth_raw, sol_raw, window=2880)
    feat = feat.dropna()
    return feat


def compute_labels(feat_df, horizon=HORIZON_15M, tp_pct=0.005, sl_pct=0.003):
    """Triple-barrier labels (same as train_model_15m.py)."""
    closes = feat_df["close"].values
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - horizon):
        entry = closes[i]
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        for j in range(i + 1, i + horizon + 1):
            if closes[j] >= tp:
                labels[i] = 1
                break
            elif closes[j] <= sl:
                break
    return pd.Series(labels, index=feat_df.index)


def train_model(coin: str, btc_raw, eth_raw, sol_raw):
    """Train XGBoost for a single coin. Returns (model, test_metrics)."""
    print(f"\n{'='*60}")
    print(f"  TRAINING {coin}/USD")
    print(f"{'='*60}")

    feat = prepare_coin_features(coin, btc_raw, eth_raw, sol_raw)
    labels = compute_labels(feat)

    cols = [c for c in FEATURE_COLS if c in feat.columns]
    X = feat[cols].iloc[:-HORIZON_15M]
    y = labels.iloc[:-HORIZON_15M]
    valid = X.notna().all(axis=1)
    X, y = X[valid], y[valid]

    n_buy = int(y.sum())
    print(f"Data: {len(X):,} bars | BUY={n_buy:,} ({n_buy/len(X):.1%})")

    # Walk-forward CV
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)
    cv_aps = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        n_pos = int(y_tr.sum())
        if n_pos == 0:
            continue
        spw = (len(y_tr) - n_pos) / n_pos
        model = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": spw})
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        proba = model.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, proba)
        cv_aps.append(ap)
        print(f"  Fold {fold}: AP={ap:.3f}")

    mean_ap = np.mean(cv_aps) if cv_aps else 0
    print(f"  CV Mean AP: {mean_ap:.3f}")

    # Train final model on pre-2024 data
    cutoff = pd.Timestamp(TRAIN_CUTOFF, tz="UTC")
    X_train = X[X.index < cutoff]
    y_train = y[y.index < cutoff]
    X_test = X[X.index >= cutoff]
    y_test = y[y.index >= cutoff]

    n_pos = int(y_train.sum())
    spw = (len(y_train) - n_pos) / n_pos if n_pos > 0 else 1.0

    final_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = 300
    final_params["scale_pos_weight"] = spw

    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train, verbose=False)

    # Test metrics
    test_metrics = {}
    if len(X_test) > 0 and y_test.sum() > 0:
        proba_test = model.predict_proba(X_test)[:, 1]
        ap_test = average_precision_score(y_test, proba_test)
        test_metrics = {"ap": ap_test, "n_test": len(X_test)}
        print(f"  Test AP: {ap_test:.3f} ({len(X_test):,} bars)")

    # Save
    out_path = f"models/xgb_{coin.lower()}_15m.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {out_path}")

    return model, test_metrics


def get_cb_mult(drawdown):
    if drawdown >= CB_HALT: return 0.0
    if drawdown >= CB_HEAVY: return 0.25
    if drawdown >= CB_LIGHT: return 0.5
    return 1.0


def backtest_single(coin, model, feat_df, close_series, threshold, exit_th=EXIT_THRESHOLD):
    """Backtest a single coin model. Returns equity Series."""
    feature_cols = list(model.feature_names_in_)
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = np.nan

    oos = feat_df[TRAIN_CUTOFF:]
    probas = model.predict_proba(oos[feature_cols])[:, 1]
    closes = close_series.reindex(oos.index).ffill().values
    idx = oos.index

    cash = float(CAPITAL)
    hwm = float(CAPITAL)
    position = None  # {qty, entry, stop, trail, atr}
    equity = []
    trades = []

    for i in range(len(idx)):
        price = closes[i]
        if np.isnan(price) or price <= 0:
            equity.append(cash + (position["qty"] * position["entry"] if position else 0))
            continue

        pv = cash + (position["qty"] * price if position else 0)
        hwm = max(hwm, pv)
        dd = (hwm - pv) / hwm if hwm > 0 else 0
        cb = get_cb_mult(dd)

        # Check stop
        if position:
            atr = position["atr"]
            new_trail = price - TRAILING_STOP_MULT * atr
            if new_trail > position["trail"]:
                position["trail"] = new_trail

            if price <= position["stop"] or price <= position["trail"]:
                pnl = (price - position["entry"]) * position["qty"]
                comm = price * position["qty"] * COMMISSION_PCT
                cash += price * position["qty"] - comm
                trades.append({"pnl": pnl - comm, "exit": "stop"})
                position = None

        p = probas[i]

        # Exit signal
        if position and p <= exit_th:
            pnl = (price - position["entry"]) * position["qty"]
            comm = price * position["qty"] * COMMISSION_PCT
            cash += price * position["qty"] - comm
            trades.append({"pnl": pnl - comm, "exit": "signal"})
            position = None

        # Entry signal
        if position is None and p >= threshold and cb > 0:
            atr_val = oos.iloc[i].get("atr_proxy", price * 0.02)
            if pd.isna(atr_val) or atr_val <= 0:
                atr_val = price * 0.02

            hard_stop = price * (1 - HARD_STOP_PCT)
            atr_stop = price - ATR_STOP_MULT * atr_val
            init_stop = max(hard_stop, atr_stop)
            stop_dist = price - init_stop
            if stop_dist <= 0:
                stop_dist = price * HARD_STOP_PCT

            # Kelly gate
            b = EXPECTED_WIN_LOSS
            kelly = (p * b - (1 - p)) / b
            if kelly <= 0:
                equity.append(cash)
                continue

            # Position sizing: full weight for single-coin backtest
            risk_usd = pv * RISK_PER_TRADE * p * cb
            qty = risk_usd / stop_dist
            target_usd = qty * price
            target_usd = min(target_usd, pv * 0.40, cash * 0.95)
            if target_usd < 100:
                equity.append(cash)
                continue
            qty = target_usd / price

            comm = target_usd * COMMISSION_PCT
            cash -= target_usd + comm
            position = {
                "qty": qty, "entry": price,
                "stop": hard_stop, "trail": init_stop,
                "atr": atr_val,
            }
            trades.append({"pnl": -comm, "exit": "entry"})

        pv_end = cash + (position["qty"] * price if position else 0)
        equity.append(pv_end)

    eq = pd.Series(equity, index=idx)
    returns = eq.pct_change().dropna()

    total_ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    n_entries = len([t for t in trades if t["exit"] == "entry"])
    wins = len([t for t in trades if t["exit"] != "entry" and t["pnl"] > 0])
    losses = len([t for t in trades if t["exit"] != "entry" and t["pnl"] <= 0])
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    sharpe = (returns.mean() / returns.std() * np.sqrt(PERIODS_15M)) if returns.std() > 0 else 0
    sortino_d = returns[returns < 0].std() * np.sqrt(PERIODS_15M)
    sortino = (returns.mean() * PERIODS_15M / sortino_d) if sortino_d > 0 else 0
    max_dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100

    return {
        "coin": coin, "equity": eq, "returns": returns,
        "total_ret": total_ret, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "n_trades": n_entries, "win_rate": win_rate,
        "wins": wins, "losses": losses,
    }


def backtest_combined(models_dict, feats_dict, closes_dict):
    """Backtest all 5 models combined with equal weight (1/N per model)."""
    n_models = len(models_dict)
    weight_per = 1.0 / n_models

    # Get common index
    common_idx = None
    for coin in models_dict:
        feat = feats_dict[coin][TRAIN_CUTOFF:]
        if common_idx is None:
            common_idx = feat.index
        else:
            common_idx = common_idx.intersection(feat.index)

    # Precompute probas
    all_probas = {}
    for coin, model in models_dict.items():
        feature_cols = list(model.feature_names_in_)
        feat = feats_dict[coin].reindex(common_idx)
        for col in feature_cols:
            if col not in feat.columns:
                feat[col] = np.nan
        all_probas[coin] = model.predict_proba(feat[feature_cols])[:, 1]

    all_closes = {}
    for coin in models_dict:
        all_closes[coin] = closes_dict[coin].reindex(common_idx).ffill().values

    cash = float(CAPITAL)
    hwm = float(CAPITAL)
    positions = {}
    equity = []
    trades = []

    for i in range(len(common_idx)):
        # Portfolio value
        pv = cash
        for coin, pos in positions.items():
            pv += pos["qty"] * all_closes[coin][i]
        hwm = max(hwm, pv)
        dd = (hwm - pv) / hwm if hwm > 0 else 0
        cb = get_cb_mult(dd)

        # Check stops
        for coin in list(positions.keys()):
            pos = positions[coin]
            price = all_closes[coin][i]
            new_trail = price - TRAILING_STOP_MULT * pos["atr"]
            if new_trail > pos["trail"]:
                pos["trail"] = new_trail
            if price <= pos["stop"] or price <= pos["trail"]:
                pnl = (price - pos["entry"]) * pos["qty"]
                comm = price * pos["qty"] * COMMISSION_PCT
                cash += price * pos["qty"] - comm
                trades.append({"coin": coin, "pnl": pnl - comm, "side": "SELL"})
                del positions[coin]

        # Signals
        for coin in models_dict:
            price = all_closes[coin][i]
            if np.isnan(price) or price <= 0:
                continue
            p = all_probas[coin][i]
            th = THRESHOLDS[coin]

            # Exit
            if coin in positions and p <= EXIT_THRESHOLD:
                pos = positions[coin]
                pnl = (price - pos["entry"]) * pos["qty"]
                comm = price * pos["qty"] * COMMISSION_PCT
                cash += price * pos["qty"] - comm
                trades.append({"coin": coin, "pnl": pnl - comm, "side": "SELL"})
                del positions[coin]

            # Entry
            if coin not in positions and p >= th and cb > 0 and len(positions) < MAX_POSITIONS:
                feat_row = feats_dict[coin].reindex(common_idx).iloc[i]
                atr_val = feat_row.get("atr_proxy", price * 0.02)
                if pd.isna(atr_val) or atr_val <= 0:
                    atr_val = price * 0.02

                hard_stop = price * (1 - HARD_STOP_PCT)
                atr_stop = price - ATR_STOP_MULT * atr_val
                init_stop = max(hard_stop, atr_stop)
                stop_dist = price - init_stop
                if stop_dist <= 0:
                    stop_dist = price * HARD_STOP_PCT

                b = EXPECTED_WIN_LOSS
                kelly = (p * b - (1 - p)) / b
                if kelly <= 0:
                    continue

                # Equal weight per model
                risk_usd = pv * RISK_PER_TRADE * weight_per * p * cb
                qty = risk_usd / stop_dist
                target_usd = qty * price
                target_usd = min(target_usd, pv * 0.40, cash * 0.95)
                if target_usd < 100:
                    continue
                qty = target_usd / price

                comm = target_usd * COMMISSION_PCT
                cash -= target_usd + comm
                positions[coin] = {
                    "qty": qty, "entry": price,
                    "stop": hard_stop, "trail": init_stop,
                    "atr": atr_val,
                }
                trades.append({"coin": coin, "pnl": -comm, "side": "BUY"})

        pv_end = cash
        for coin, pos in positions.items():
            pv_end += pos["qty"] * all_closes[coin][i]
        equity.append(pv_end)

    eq = pd.Series(equity, index=common_idx)
    returns = eq.pct_change().dropna()

    total_ret = (eq.iloc[-1] / CAPITAL - 1) * 100
    n_entries = len([t for t in trades if t["side"] == "BUY"])
    sells = [t for t in trades if t["side"] == "SELL"]
    wins = len([t for t in sells if t["pnl"] > 0])
    losses = len([t for t in sells if t["pnl"] <= 0])
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    sharpe = (returns.mean() / returns.std() * np.sqrt(PERIODS_15M)) if returns.std() > 0 else 0
    sortino_d = returns[returns < 0].std() * np.sqrt(PERIODS_15M)
    sortino = (returns.mean() * PERIODS_15M / sortino_d) if sortino_d > 0 else 0
    max_dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    calmar = (returns.mean() * PERIODS_15M / abs(max_dd / 100)) if max_dd != 0 else 0
    comp_score = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    # Per-coin trade count
    coin_counts = {}
    for t in trades:
        if t["side"] == "BUY":
            coin_counts[t["coin"]] = coin_counts.get(t["coin"], 0) + 1

    # Active days
    buy_dates = set()
    buy_idx = 0
    for t in trades:
        if t["side"] == "BUY":
            buy_dates.add(common_idx[min(buy_idx, len(common_idx)-1)].date())
        buy_idx += 1

    return {
        "equity": eq, "returns": returns,
        "total_ret": total_ret, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "calmar": calmar, "comp_score": comp_score,
        "n_trades": n_entries, "win_rate": win_rate,
        "wins": wins, "losses": losses, "coin_counts": coin_counts,
    }


def main():
    print("Loading base data...")
    btc_raw = load_parquet("BTC")
    eth_raw = load_parquet("ETH")
    sol_raw = load_parquet("SOL")

    # ── Step 1: Train new models ─────────────────────────────────────────────
    models = {}
    feats = {}
    closes = {}

    # Load existing BTC and SOL models
    for coin, model_path in [("BTC", "models/xgb_btc_15m_iter5.pkl"), ("SOL", "models/xgb_sol_15m.pkl")]:
        print(f"\nLoading existing {coin} model: {model_path}")
        with open(model_path, "rb") as f:
            models[coin] = pickle.load(f)
        feats[coin] = prepare_coin_features(coin, btc_raw, eth_raw, sol_raw)
        closes[coin] = load_parquet(coin)["close"]

    # Train new models
    for coin in COINS_TO_TRAIN:
        model, metrics = train_model(coin, btc_raw, eth_raw, sol_raw)
        models[coin] = model
        feats[coin] = prepare_coin_features(coin, btc_raw, eth_raw, sol_raw)
        closes[coin] = load_parquet(coin)["close"]

    # ── Step 2: Individual backtests ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  INDIVIDUAL BACKTESTS (OOS 2024-2026)")
    print(f"{'='*60}")
    print(f"\n{'Coin':<8} {'Return':>10} {'Sharpe':>10} {'Sortino':>10} {'MaxDD':>10} {'Trades':>8} {'WinRate':>8}")
    print("-" * 70)

    individual_results = {}
    for coin in ALL_COINS:
        r = backtest_single(coin, models[coin], feats[coin].copy(), closes[coin],
                           threshold=THRESHOLDS[coin])
        individual_results[coin] = r
        print(f"{coin:<8} {r['total_ret']:>+9.2f}% {r['sharpe']:>10.3f} {r['sortino']:>10.3f} "
              f"{r['max_dd']:>9.2f}% {r['n_trades']:>8} {r['win_rate']:>7.1f}%")

    # ── Step 3: Combined backtest ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMBINED 5-COIN BACKTEST (OOS 2024-2026)")
    print(f"{'='*60}")

    combined = backtest_combined(models, feats, closes)

    print(f"\nTotal Return:   {combined['total_ret']:+.2f}%")
    print(f"Sharpe:         {combined['sharpe']:.3f}")
    print(f"Sortino:        {combined['sortino']:.3f}")
    print(f"Calmar:         {combined['calmar']:.3f}")
    print(f"Max Drawdown:   {combined['max_dd']:.2f}%")
    print(f"Comp Score:     {combined['comp_score']:.3f}")
    print(f"Total Trades:   {combined['n_trades']}")
    print(f"Win Rate:       {combined['win_rate']:.1f}% ({combined['wins']}W / {combined['losses']}L)")
    print(f"\nTrades per coin:")
    for coin, count in sorted(combined["coin_counts"].items(), key=lambda x: -x[1]):
        print(f"  {coin}: {count}")

    # 10-day windows
    eq = combined["equity"]
    bars_10d = 10 * 24 * 4
    if len(eq) > bars_10d:
        print(f"\n10-DAY ROLLING WINDOWS:")
        print("-" * 60)
        for offset in range(5):
            end_i = len(eq) - 1 - offset * bars_10d
            start_i = end_i - bars_10d
            if start_i < 0:
                break
            w = eq.iloc[start_i:end_i]
            w_ret = (w.iloc[-1] / w.iloc[0] - 1) * 100
            w_r = w.pct_change().dropna()
            w_sh = (w_r.mean() / w_r.std() * np.sqrt(PERIODS_15M)) if w_r.std() > 0 else 0
            w_dd = ((w - w.cummax()) / w.cummax()).min() * 100
            print(f"  {w.index[0].strftime('%Y-%m-%d')} -> {w.index[-1].strftime('%Y-%m-%d')}: "
                  f"ret={w_ret:+.2f}% sharpe={w_sh:.2f} dd={w_dd:.2f}%")


if __name__ == "__main__":
    main()

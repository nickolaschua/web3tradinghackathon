#!/usr/bin/env python3
"""
Train and backtest a 15M XGBoost model for ETH/SOL/BNB/DOGE.

Mirrors train_model_15m.py (BTC) but adapts the feature pipeline so that
the target coin's own OHLCV drives the technical indicators, and other
coins contribute cross-asset lag/correlation features.

Feature set (19 cols, parallel to BTC model):
  Standard indicators on target coin (13)
  + 4H and 1D cross-asset return lags from two other coins (4)
  + target_btc_corr, target_btc_beta (2)

Usage:
  python scripts/train_alt_15m.py --target eth
  python scripts/train_alt_15m.py --target sol
  python scripts/train_alt_15m.py --target bnb --sweep
  python scripts/train_alt_15m.py --target doge --sweep
  python scripts/train_alt_15m.py --target eth --cv-only
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_cross_asset_features, compute_features


# ── Constants (match train_model_15m.py) ───────────────────────────────────────

HORIZON_15M = 16            # 16 bars = 4H at 15M resolution
PERIODS_15M = 35_040        # 365.25 × 24 × 4
CV_GAP = 64                 # leakage guard: 4 × HORIZON_15M
CV_SPLITS = 5
CORR_WINDOW = 2880          # 30 days at 15M (matches BTC model)

TRAIN_CUTOFF = "2024-01-01"

# Feature columns: standard 13 indicators + 4 cross-asset lags + 2 context features
_STANDARD_COLS = [
    "atr_proxy", "RSI_14", "RSI_7",
    "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    "bb_width", "bb_pos",
    "volume_ratio", "candle_body",
]

FEATURE_COLS_BY_TARGET = {
    "eth": _STANDARD_COLS + [
        "btc_return_4h", "sol_return_4h",
        "btc_return_1d", "sol_return_1d",
        "eth_btc_corr", "eth_btc_beta",
    ],
    "sol": _STANDARD_COLS + [
        "btc_return_4h", "eth_return_4h",
        "btc_return_1d", "eth_return_1d",
        "sol_btc_corr", "sol_btc_beta",
    ],
}


def _get_feature_cols(target: str) -> list[str]:
    """Return feature columns for any target coin.
    ETH/SOL have hardcoded cross-asset lags. All others use ETH+SOL lags."""
    if target in FEATURE_COLS_BY_TARGET:
        return FEATURE_COLS_BY_TARGET[target]
    return _STANDARD_COLS + [
        "eth_return_4h", "sol_return_4h",
        "eth_return_1d", "sol_return_1d",
        f"{target}_btc_corr", f"{target}_btc_beta",
    ]

XGB_PARAMS = dict(
    n_estimators=300,
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


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_features(
    target: str,
    btc_path: str,
    eth_path: str,
    sol_path: str,
    target_path: str = None,
) -> pd.DataFrame:
    """
    Build feature matrix for any target coin.

    Pipeline:
    1. compute_features(target_df)              — standard indicators on target
    2. compute_cross_asset_features(feat, ...)  — lag1/lag2 from other coins
    3. Add 4H (16-bar) and 1D (96-bar) cross-asset lags
    4. Compute rolling corr / beta of target vs BTC
    5. dropna()
    """
    # Load core coins (always needed for cross-asset features)
    all_dfs = {
        "btc": pd.read_parquet(btc_path),
        "eth": pd.read_parquet(eth_path),
        "sol": pd.read_parquet(sol_path),
    }
    # Load target coin if not already in core set
    if target not in all_dfs:
        if target_path is None:
            target_path = f"data/{target.upper()}USDT_15m.parquet"
        all_dfs[target] = pd.read_parquet(target_path)

    for df in all_dfs.values():
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()

    target_df = all_dfs[target]

    # Cross-asset DataFrames: core coins except target
    _to_pair = lambda k: f"{k.upper()}/USD"
    cross_dfs = {_to_pair(k): v for k, v in all_dfs.items()
                 if k != target and k in ("btc", "eth", "sol")}

    feat = compute_features(target_df)
    feat = compute_cross_asset_features(feat, cross_dfs)

    # Determine which two coins provide the 4H/1D cross-asset lags
    feature_cols = _get_feature_cols(target)
    lag_assets = []
    for col in feature_cols:
        for asset in ("btc", "eth", "sol"):
            if col == f"{asset}_return_4h" and asset != target:
                lag_assets.append(asset)

    for asset in lag_assets:
        df = all_dfs[asset]
        log_ret = np.log(df["close"] / df["close"].shift(1))
        feat[f"{asset}_return_4h"] = log_ret.shift(16).reindex(feat.index)
        feat[f"{asset}_return_1d"] = log_ret.shift(96).reindex(feat.index)

    # Rolling correlation and beta: target vs BTC
    target_ret = np.log(target_df["close"] / target_df["close"].shift(1)).reindex(feat.index)
    btc_ret = np.log(all_dfs["btc"]["close"] / all_dfs["btc"]["close"].shift(1)).reindex(feat.index)

    corr = target_ret.rolling(CORR_WINDOW).corr(btc_ret)
    cov = target_ret.rolling(CORR_WINDOW).cov(btc_ret)
    var_btc = btc_ret.rolling(CORR_WINDOW).var()

    # Shift 1 bar (match compute_features convention — no look-ahead)
    feat[f"{target}_btc_corr"] = corr.shift(1)
    feat[f"{target}_btc_beta"] = (cov / (var_btc + 1e-10)).shift(1)

    feat = feat.dropna()
    return feat


# ── Triple-barrier labeling ────────────────────────────────────────────────────

def compute_triple_barrier_labels(
    feat_df: pd.DataFrame,
    horizon: int = HORIZON_15M,
    tp_pct: float = 0.008,
    sl_pct: float = 0.003,
) -> pd.Series:
    """
    Triple-barrier labeling (Lopez de Prado) — fully vectorized with numpy.

    Uses sliding_window_view to build a (n, horizon) matrix of forward prices,
    then finds the first bar where TP or SL is touched. ~100x faster than the
    pure-Python inner loop.
    """
    closes = feat_df["close"].values
    n = len(closes)

    # Build (n - horizon, horizon + 1) window: col 0 = entry, cols 1.. = future
    windows = np.lib.stride_tricks.sliding_window_view(closes, horizon + 1)
    entry_prices = windows[:, 0:1]          # shape (n-horizon, 1)
    future = windows[:, 1:]                 # shape (n-horizon, horizon)

    tp_hit = future >= entry_prices * (1.0 + tp_pct)   # bool (n-horizon, horizon)
    sl_hit = future <= entry_prices * (1.0 - sl_pct)   # bool (n-horizon, horizon)

    tp_any = tp_hit.any(axis=1)
    sl_any = sl_hit.any(axis=1)

    # argmax returns first True index; use horizon as sentinel for "no hit"
    tp_first = np.where(tp_any, np.argmax(tp_hit, axis=1), horizon)
    sl_first = np.where(sl_any, np.argmax(sl_hit, axis=1), horizon)

    labels_short = np.where(tp_any & (tp_first < sl_first), 1, 0).astype(np.int8)

    # Pad last `horizon` rows with 0 (no future data)
    labels = np.zeros(n, dtype=np.int8)
    labels[: len(labels_short)] = labels_short

    return pd.Series(labels, index=feat_df.index)


def prepare_training_data(
    feat_df: pd.DataFrame,
    feature_cols: list,
    horizon: int = HORIZON_15M,
    tp_pct: float = 0.008,
    sl_pct: float = 0.003,
):
    labels = compute_triple_barrier_labels(feat_df, horizon, tp_pct, sl_pct)

    cols = [c for c in feature_cols if c in feat_df.columns]
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        print(f"WARNING: FEATURE_COLS missing from data: {missing}")

    X = feat_df[cols].iloc[:-horizon]
    y = labels.iloc[:-horizon]

    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    print(f"Training data: {len(X):,} bars | Features: {X.shape[1]}")
    print(f"Date range: {X.index[0]} to {X.index[-1]}")
    n_buy = int(y.sum())
    n_total = len(y)
    print(f"Class balance: BUY={n_buy:,} ({n_buy/n_total:.1%}), NOT-BUY={n_total - n_buy:,} ({1 - n_buy/n_total:.1%})")

    return X, y


# ── Walk-forward CV ────────────────────────────────────────────────────────────

def run_walk_forward_cv(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)
    scores = []

    print(f"Walk-forward CV: {CV_SPLITS} splits, gap={CV_GAP} bars ({CV_GAP * 15 // 60}H)")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        n_pos = int(y_tr.sum())
        n_neg = int(len(y_tr) - n_pos)
        if n_pos == 0:
            print(f"Fold {fold}: SKIP — no positive labels in training fold")
            continue
        spw = n_neg / n_pos

        model = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": spw})
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        proba = model.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, proba)
        f1 = f1_score(y_va, (proba >= 0.5).astype(int), zero_division=0)
        best_iter = model.best_iteration if hasattr(model, "best_iteration") else "N/A"

        scores.append({"fold": fold, "ap": ap, "f1": f1, "best_iter": best_iter,
                       "n_train": len(X_tr), "n_val": len(X_va)})
        print(f"Fold {fold}: AP={ap:.3f}  F1={f1:.3f}  best_iter={best_iter}  "
              f"train={len(X_tr):,}  val={len(X_va):,}")

    if scores:
        mean_ap = sum(s["ap"] for s in scores) / len(scores)
        mean_f1 = sum(s["f1"] for s in scores) / len(scores)
        print(f"\nCV Summary: Mean AP={mean_ap:.3f}  Mean F1={mean_f1:.3f}  ({len(scores)} folds)")

    return scores


# ── Final model ────────────────────────────────────────────────────────────────

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_cutoff: str = TRAIN_CUTOFF,
    n_estimators: int = 300,
):
    cutoff = pd.Timestamp(test_cutoff, tz="UTC")

    X_train_val = X[X.index < cutoff]
    y_train_val = y[y.index < cutoff]
    X_test = X[X.index >= cutoff]
    y_test = y[y.index >= cutoff]

    print(f"Train+Val: {len(X_train_val):,} bars ({X_train_val.index[0].date()} to {X_train_val.index[-1].date()})")
    print(f"Test:      {len(X_test):,} bars ({X_test.index[0].date()} to {X_test.index[-1].date()})")

    if len(X_test) == 0:
        print("WARNING: No test data after cutoff.")

    n_pos = int(y_train_val.sum())
    n_neg = int(len(y_train_val) - n_pos)
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Train+Val class balance: BUY={n_pos:,} ({n_pos/len(y_train_val):.1%}), scale_pos_weight={spw:.2f}")

    final_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = n_estimators
    final_params["scale_pos_weight"] = spw

    print(f"\nTraining final model (n_estimators={n_estimators}, no early stopping)...")
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train_val, y_train_val, verbose=False)

    if len(X_test) > 0 and y_test.sum() > 0:
        proba_test = model.predict_proba(X_test)[:, 1]
        ap_test = average_precision_score(y_test, proba_test)
        f1_test = f1_score(y_test, (proba_test >= 0.5).astype(int), zero_division=0)
        n_buy_test = int(y_test.sum())
        print(f"\nTest set evaluation:")
        print(f"  AP (AUC-PR): {ap_test:.3f}")
        print(f"  F1 (thresh=0.5): {f1_test:.3f}")
        print(f"  Test bars: {len(X_test):,} | BUY signals: {n_buy_test} ({n_buy_test/len(X_test):.1%})")

    importances = pd.Series(
        model.feature_importances_, index=model.feature_names_in_
    ).sort_values(ascending=False)
    print(f"\nTop 5 feature importances:")
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    return model


# ── OOS Backtest ───────────────────────────────────────────────────────────────

def _cb_multiplier(dd: float, halt=0.30, heavy=0.20, light=0.10) -> float:
    if dd >= halt:
        return 0.0
    if dd >= heavy:
        return 0.25
    if dd >= light:
        return 0.50
    return 1.0


def run_backtest(
    feat_df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float,
    initial_capital: float = 10_000.0,
    risk_per_trade_pct: float = 0.02,
    hard_stop_pct: float = 0.05,
    atr_stop_multiplier: float = 10.0,
    max_single_position_pct: float = 0.40,
    expected_win_loss_ratio: float = 1.5,
    fee_bps: int = 10,
) -> tuple:
    """
    XGBoost-only bar-by-bar backtest with live-bot risk stack.
    Mirrors backtest_15m.py run_backtest() but without MR/pairs overlays.
    """
    fee_rate = fee_bps / 10_000.0
    sell_threshold = 1.0 - threshold

    closes = feat_df["close"].values
    atrs = feat_df["atr_proxy"].values
    timestamps = feat_df.index
    n = len(closes)

    free_balance = initial_capital
    portfolio_hwm = initial_capital
    position_units = 0.0
    entry_effective_price = 0.0
    trail_stop = 0.0
    entry_bar_ts = None

    portfolio_values = np.zeros(n)
    portfolio_values[0] = initial_capital
    returns = np.zeros(n)
    closed_trades = []
    gate_stats = {"kelly_blocked": 0, "cb_halted": 0, "cb_reduced": 0}

    for i in range(n):
        c = closes[i]
        atr = atrs[i]
        p = probas[i]
        ts = timestamps[i]

        # Update trailing stop
        if position_units > 0 and not np.isnan(atr) and atr > 0:
            new_atr_stop = c - atr_stop_multiplier * atr
            trail_stop = max(trail_stop, new_atr_stop)

        position_value = position_units * c
        total_portfolio = free_balance + position_value
        portfolio_hwm = max(portfolio_hwm, total_portfolio)
        drawdown = (portfolio_hwm - total_portfolio) / portfolio_hwm if portfolio_hwm > 0 else 0.0
        cb_mult = _cb_multiplier(drawdown)

        just_exited = False

        # Exit
        if position_units > 0:
            stop_hit = c <= trail_stop
            sell_signal = p <= sell_threshold

            if stop_hit or sell_signal:
                proceeds = position_units * c * (1.0 - fee_rate)
                net_exit = c * (1.0 - fee_rate)
                pnl_pct = (net_exit - entry_effective_price) / entry_effective_price

                closed_trades.append({
                    "entry_bar": entry_bar_ts,
                    "exit_bar": ts,
                    "entry_price": entry_effective_price,
                    "exit_price": net_exit,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "stop" if stop_hit else "signal",
                })

                free_balance += proceeds
                position_units = 0.0
                just_exited = True
                total_portfolio = free_balance

        # Entry
        if position_units == 0 and not just_exited and p >= threshold:
            kelly = (p * expected_win_loss_ratio - (1.0 - p)) / expected_win_loss_ratio
            if kelly <= 0:
                gate_stats["kelly_blocked"] += 1
            elif cb_mult == 0.0:
                gate_stats["cb_halted"] += 1
            else:
                if cb_mult < 1.0:
                    gate_stats["cb_reduced"] += 1

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

                risk_usd = total_portfolio * risk_per_trade_pct * p * cb_mult
                quantity = risk_usd / stop_distance
                target_usd = quantity * c

                usable = free_balance * 0.95
                target_usd = min(target_usd, total_portfolio * max_single_position_pct, usable)

                if target_usd >= 10.0:
                    position_units = target_usd / c
                    entry_fee = target_usd * fee_rate
                    free_balance -= (target_usd + entry_fee)
                    entry_effective_price = c * (1.0 + fee_rate)
                    trail_stop = initial_stop
                    entry_bar_ts = ts

        portfolio_values[i] = free_balance + (position_units * c)
        if i > 0:
            returns[i] = portfolio_values[i] / portfolio_values[i - 1] - 1.0

    return (
        pd.Series(returns, index=timestamps),
        pd.Series(portfolio_values, index=timestamps),
        closed_trades,
        gate_stats,
    )


def compute_stats(returns, portfolio, closed_trades, initial_capital):
    n = len(closed_trades)
    if n > 0:
        winners = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
        win_rate = winners / n
        avg_pnl = sum(t["pnl_pct"] for t in closed_trades) / n
        stop_exits = sum(1 for t in closed_trades if t["exit_reason"] == "stop")
    else:
        win_rate = avg_pnl = 0.0
        stop_exits = 0

    return {
        "sharpe": float(qs.stats.sharpe(returns, periods=PERIODS_15M)),
        "sortino": float(qs.stats.sortino(returns, periods=PERIODS_15M)),
        "max_drawdown_pct": float(qs.stats.max_drawdown(returns)) * 100,
        "total_return_pct": (portfolio.iloc[-1] - initial_capital) / initial_capital * 100,
        "n_trades": n,
        "stop_exits": stop_exits,
        "win_rate_pct": win_rate * 100,
        "avg_pnl_pct": avg_pnl * 100,
    }


def run_threshold_sweep(feat_oos, probas_oos, initial_capital):
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    results = []
    for t in thresholds:
        ret, port, trades, gates = run_backtest(feat_oos, probas_oos, t, initial_capital=initial_capital)
        stats = compute_stats(ret, port, trades, initial_capital)
        results.append({
            "threshold": t,
            "sharpe": round(stats["sharpe"], 3),
            "sortino": round(stats["sortino"], 3),
            "n_trades": stats["n_trades"],
            "total_return_pct": round(stats["total_return_pct"], 2),
            "win_rate_pct": round(stats["win_rate_pct"], 1),
        })
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results


# ── Save model ─────────────────────────────────────────────────────────────────

def save_model(model, output_path: str) -> None:
    assert hasattr(model, "predict_proba"), "Model must have predict_proba"
    assert hasattr(model, "feature_names_in_"), "Model must have feature_names_in_"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    with open(output_path, "rb") as f:
        reloaded = pickle.load(f)
    assert list(reloaded.feature_names_in_) == list(model.feature_names_in_)

    print(f"\nSaved: {output_path}")
    print(f"Feature columns ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")
    print("Round-trip pickle verification: OK")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train 15M XGBoost model for ETH/SOL/BNB/DOGE"
    )
    p.add_argument("--target", required=True,
                   help="Target coin ticker (e.g. eth, sol, bnb, doge, link, avax, ada, ...)")
    p.add_argument("--btc", default="data/BTCUSDT_15m.parquet")
    p.add_argument("--eth", default="data/ETHUSDT_15m.parquet")
    p.add_argument("--sol", default="data/SOLUSDT_15m.parquet")
    p.add_argument("--target-data", default=None,
                   help="Path to target coin's 15m parquet (default: data/{TARGET}USDT_15m.parquet)")
    p.add_argument("--horizon", type=int, default=HORIZON_15M)
    p.add_argument("--tp-pct", type=float, default=0.008,
                   help="Take-profit %% for triple-barrier label (default 0.8%%)")
    p.add_argument("--sl-pct", type=float, default=0.003,
                   help="Stop-loss %% for triple-barrier label (default 0.3%%)")
    p.add_argument("--test-cutoff", default=TRAIN_CUTOFF)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--cv-only", action="store_true",
                   help="Run walk-forward CV only, skip final training and backtest")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep thresholds 0.50-0.85 on OOS data")
    p.add_argument("--threshold", type=float, default=0.65,
                   help="P(BUY) threshold for OOS backtest report (default 0.65)")
    p.add_argument("--output", default=None,
                   help="Override output path (default: models/xgb_{target}_15m.pkl)")
    return p.parse_args()


def main():
    args = parse_args()
    target = args.target.lower()
    output_path = args.output or f"models/xgb_{target}_15m.pkl"
    feature_cols = _get_feature_cols(target)

    print(f"=== Training XGBoost 15M model for {target.upper()}/USD ===\n")

    print("Step 1: Loading and computing features...")
    target_path = args.target_data or f"data/{target.upper()}USDT_15m.parquet"
    feat = prepare_features(target, args.btc, args.eth, args.sol,
                            target_path=target_path)
    print(f"  Feature matrix: {feat.shape[0]:,} bars x {feat.shape[1]} columns")
    print(f"  Date range: {feat.index[0]} to {feat.index[-1]}")

    print(f"\nStep 2: Preparing training data (TP={args.tp_pct:.1%}, SL={args.sl_pct:.1%})...")
    X, y = prepare_training_data(feat, feature_cols, args.horizon, args.tp_pct, args.sl_pct)

    print("\nStep 3: Walk-forward CV...")
    run_walk_forward_cv(X, y)

    if args.cv_only:
        print("\nCV complete. Re-run without --cv-only to train and save.")
        return

    print("\nStep 4: Training final model...")
    model = train_final_model(X, y, args.test_cutoff, args.n_estimators)

    print("\nStep 5: Saving model...")
    save_model(model, output_path)

    # OOS backtest
    print("\nStep 6: OOS backtest (2024-01-01 onwards)...")
    cutoff_ts = pd.Timestamp(args.test_cutoff, tz="UTC")
    feat_oos = feat[feat.index >= cutoff_ts]
    if feat_oos.empty:
        print("  No OOS data — skipping backtest.")
        return

    print(f"  OOS bars: {len(feat_oos):,} ({feat_oos.index[0].date()} to {feat_oos.index[-1].date()})")

    feature_cols_model = list(model.feature_names_in_)
    probas_oos = model.predict_proba(feat_oos[feature_cols_model])[:, 1]
    initial_capital = 10_000.0

    if args.sweep:
        print("\n  Threshold sweep (OOS):")
        results = run_threshold_sweep(feat_oos, probas_oos, initial_capital)
        header = f"{'Threshold':>10}  {'Sharpe':>7}  {'Sortino':>8}  {'Trades':>7}  {'Return%':>8}  {'Win%':>6}"
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        for r in results:
            print(f"  {r['threshold']:>10.2f}  {r['sharpe']:>7.3f}  {r['sortino']:>8.3f}  "
                  f"{r['n_trades']:>7}  {r['total_return_pct']:>8.2f}  {r['win_rate_pct']:>6.1f}")
    else:
        ret, port, trades, gates = run_backtest(
            feat_oos, probas_oos, args.threshold, initial_capital=initial_capital
        )
        stats = compute_stats(ret, port, trades, initial_capital)
        sep = "=" * 55
        print(f"\n  {sep}")
        print(f"  OOS BACKTEST: {target.upper()}/USD  (threshold={args.threshold})")
        print(f"  {sep}")
        print(f"  Sharpe        : {stats['sharpe']:.3f}")
        print(f"  Sortino       : {stats['sortino']:.3f}")
        print(f"  Max drawdown  : {stats['max_drawdown_pct']:.2f}%")
        print(f"  Total return  : {stats['total_return_pct']:+.2f}%")
        print(f"  # Trades      : {stats['n_trades']}")
        print(f"    Stop exits  : {stats['stop_exits']}")
        print(f"  Win rate      : {stats['win_rate_pct']:.1f}%")
        print(f"  Avg trade PnL : {stats['avg_pnl_pct']:+.2f}%")
        print(f"  {sep}")
        print(f"  Kelly blocked : {gates['kelly_blocked']}")
        print(f"  CB halted     : {gates['cb_halted']}")
        print(f"  CB reduced    : {gates['cb_reduced']}")
        print(f"  {sep}")

    print(f"\nDone. To run full backtest: python scripts/backtest_15m.py --model {output_path}")


if __name__ == "__main__":
    main()

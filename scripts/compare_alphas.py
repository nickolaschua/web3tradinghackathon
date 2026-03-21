#!/usr/bin/env python3
"""
Comprehensive Alpha Strategy Comparison
Tests multiple alpha strategies and ranks by composite score: 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar

Strategies tested:
1. Oil + DXY (macro only)
2. Funding + Oil + DXY (funding + macro)
3. Base indicators only (technical baseline)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs
import xgboost as xgb
import yaml
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# Add project root so shared framework modules can be imported
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bot.data.features import compute_cross_asset_features, compute_features
from bot.execution.portfolio import PortfolioAllocator
from bot.execution.risk import RiskDecision, RiskManager

# Configuration
DATA_DIR = Path("research_data")
RESULTS_DIR = Path("research_results")
CONFIG_PATH = Path("bot/config/config.yaml")
PERIODS_4H = 2190  # Annualization for 4H crypto bars

# Feature sets
BASE_FEATURES = [
    "atr_proxy", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    "eth_return_lag1", "eth_return_lag2",
    "sol_return_lag1", "sol_return_lag2"
]

OIL_DXY_FEATURES = [
    "oil_return_1d", "oil_return_5d", "oil_vol_5d", "oil_acceleration",
    "dxy_return_1d", "dxy_return_5d", "dxy_vol_5d", "dxy_acceleration",
]

FUNDING_FEATURES = [
    "btc_funding_rate", "btc_funding_ma8", "btc_funding_zscore", "btc_funding_cum24h",
    "eth_funding_rate", "eth_funding_ma8", "eth_funding_zscore", "eth_funding_cum24h",
    "sol_funding_rate", "sol_funding_ma8", "sol_funding_zscore", "sol_funding_cum24h",
]

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def load_framework_config():
    """Load bot config used by shared risk + portfolio framework."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all_data():
    """Load OHLCV, funding, and macro data."""
    # BTC OHLCV
    btc = pd.read_parquet(DATA_DIR / "BTCUSDT_4h.parquet")
    btc.columns = btc.columns.str.lower()

    # ETH, SOL
    eth = pd.read_parquet(DATA_DIR / "ETHUSDT_4h.parquet") if (DATA_DIR / "ETHUSDT_4h.parquet").exists() else None
    sol = pd.read_parquet(DATA_DIR / "SOLUSDT_4h.parquet") if (DATA_DIR / "SOLUSDT_4h.parquet").exists() else None

    # Funding
    btc_funding = pd.read_parquet(DATA_DIR / "BTCUSDT_funding.parquet") if (DATA_DIR / "BTCUSDT_funding.parquet").exists() else None
    eth_funding = pd.read_parquet(DATA_DIR / "ETHUSDT_funding.parquet") if (DATA_DIR / "ETHUSDT_funding.parquet").exists() else None
    sol_funding = pd.read_parquet(DATA_DIR / "SOLUSDT_funding.parquet") if (DATA_DIR / "SOLUSDT_funding.parquet").exists() else None

    # Macro
    oil = pd.read_parquet(DATA_DIR / "oil_daily.parquet") if (DATA_DIR / "oil_daily.parquet").exists() else None
    dxy = pd.read_parquet(DATA_DIR / "dxy_daily.parquet") if (DATA_DIR / "dxy_daily.parquet").exists() else None

    return {
        "btc": btc, "eth": eth, "sol": sol,
        "btc_funding": btc_funding, "eth_funding": eth_funding, "sol_funding": sol_funding,
        "oil": oil, "dxy": dxy,
    }


def compute_base_features(btc_df, eth_df=None, sol_df=None):
    """Compute base features through the shared framework pipeline."""
    out = compute_features(btc_df.copy())

    others = {}
    if eth_df is not None:
        others["ETH/USD"] = eth_df
    if sol_df is not None:
        others["SOL/USD"] = sol_df

    out = compute_cross_asset_features(out, others)
    return out


def compute_funding_features(btc_df, btc_funding, eth_funding=None, sol_funding=None):
    """Compute funding rate features."""
    out = btc_df.copy()

    for symbol, fdf, prefix in [
        ("BTC", btc_funding, "btc"),
        ("ETH", eth_funding, "eth"),
        ("SOL", sol_funding, "sol"),
    ]:
        if fdf is None:
            continue

        fr = fdf["funding_rate"].copy()
        fr_4h = fr.resample("4h").ffill()
        aligned = fr_4h.reindex(out.index, method="ffill")

        out[f"{prefix}_funding_rate"] = aligned.shift(1)
        out[f"{prefix}_funding_ma8"] = aligned.rolling(8).mean().shift(1)

        rm = aligned.rolling(90).mean()
        rs = aligned.rolling(90).std()
        out[f"{prefix}_funding_zscore"] = ((aligned - rm) / rs.replace(0, np.nan)).shift(1)
        out[f"{prefix}_funding_cum24h"] = aligned.rolling(6).sum().shift(1)

    return out


def compute_macro_features(btc_df, oil_df, dxy_df):
    """Compute Oil + DXY macro features."""
    out = btc_df.copy()

    for name, mdf in [("oil", oil_df), ("dxy", dxy_df)]:
        if mdf is None:
            continue

        close = mdf["close"].copy()
        close_4h = close.resample("4h").ffill()
        aligned = close_4h.reindex(out.index, method="ffill")

        out[f"{name}_return_1d"] = aligned.pct_change(6).shift(1)
        out[f"{name}_return_5d"] = aligned.pct_change(30).shift(1)
        out[f"{name}_vol_5d"] = aligned.pct_change().rolling(30).std().shift(1)

        r1d = aligned.pct_change(6)
        out[f"{name}_acceleration"] = (r1d - r1d.shift(6)).shift(1)

    return out


def train_and_backtest(
    X_train,
    y_train,
    X_test,
    y_test,
    test_close,
    test_atr,
    portfolio_weight,
    framework_config,
    use_macro_filter=False,
    oil_filter=None,
    dxy_filter=None,
    threshold=0.6,
    fee_bps=10,
    initial_capital=10_000.0,
):
    """
    Train XGBoost model and run fee-adjusted backtest.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        test_close: Test set close prices
        threshold: Probability threshold for BUY signal
        fee_bps: Transaction fee in basis points

    Returns:
        dict with model, returns, trades, stats
    """
    # Train model
    np_pos = int(y_train.sum())
    np_neg = len(y_train) - np_pos

    if np_pos == 0 or np_neg == 0:
        return None

    model = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": np_neg / np_pos})
    model.fit(X_train, y_train, verbose=False)

    # Backtest on test set using shared RiskManager semantics
    risk_manager = RiskManager(config=framework_config)
    risk_manager.initialize_hwm(initial_capital)

    fee_rate = fee_bps / 10_000
    pair = "BTC/USD"
    free_balance = initial_capital
    position_units = 0.0
    entry_price = None
    entry_bar = None

    returns = []
    timestamps = []
    closed_trades = []
    common_idx = X_test.index.intersection(test_close.index).intersection(test_atr.index)
    X_test_aligned = X_test.loc[common_idx].dropna()
    close_aligned = test_close.loc[X_test_aligned.index]
    atr_aligned = test_atr.loc[X_test_aligned.index]
    oil_aligned = oil_filter.loc[X_test_aligned.index] if (use_macro_filter and oil_filter is not None) else None
    dxy_aligned = dxy_filter.loc[X_test_aligned.index] if (use_macro_filter and dxy_filter is not None) else None
    probas = model.predict_proba(X_test_aligned)[:, 1]

    prev_portfolio = initial_capital
    for i, (idx, row) in enumerate(X_test_aligned.iterrows()):
        close = close_aligned.loc[idx]
        atr = atr_aligned.loc[idx]
        proba = probas[i]
        just_exited = False

        # Check stop-based exits first (matches orchestration flow)
        if position_units > 0:
            stop_result = risk_manager.check_stops(pair, close, atr)
            sell_signal = proba <= (1.0 - threshold)
            if stop_result.should_exit or sell_signal:
                exit_price = close * (1 - fee_rate)
                proceeds = position_units * close * (1 - fee_rate)
                pnl_pct = (exit_price - entry_price) / entry_price
                closed_trades.append(
                    {
                        "entry_bar": entry_bar,
                        "exit_bar": idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "stop" if stop_result.should_exit else "signal",
                    }
                )
                free_balance += proceeds
                position_units = 0.0
                entry_price = None
                entry_bar = None
                risk_manager.record_exit(pair)
                just_exited = True

        # Mark-to-market portfolio and update circuit breaker
        total_portfolio = free_balance + position_units * close
        risk_manager.check_circuit_breaker(total_portfolio)

        # Entry path via shared sizing gates
        if (position_units == 0.0) and (not just_exited) and (proba >= threshold):
            if use_macro_filter:
                if (oil_aligned is None) or (dxy_aligned is None):
                    continue
                # Risk-on macro gate: oil momentum up and DXY momentum down.
                # This is a filter only (signal model remains funding/feature driven).
                oil_1d = oil_aligned.loc[idx]
                dxy_1d = dxy_aligned.loc[idx]
                if pd.isna(oil_1d) or pd.isna(dxy_1d) or not (oil_1d > 0 and dxy_1d < 0):
                    continue

            sizing = risk_manager.size_new_position(
                pair=pair,
                current_price=close,
                current_atr=atr,
                free_balance_usd=free_balance,
                open_positions={},
                regime_multiplier=1.0,
                confidence=float(proba),
                portfolio_weight=portfolio_weight,
            )
            if sizing.decision == RiskDecision.APPROVED and sizing.approved_usd_value >= 10.0:
                target_usd = sizing.approved_usd_value
                position_units = target_usd / close
                fee_cost = target_usd * fee_rate
                free_balance -= (target_usd + fee_cost)
                entry_price = close * (1 + fee_rate)
                entry_bar = idx
                risk_manager.record_entry(pair, entry_price, sizing.trailing_stop_price)

        end_portfolio = free_balance + position_units * close
        bar_return = (end_portfolio / prev_portfolio - 1.0) if prev_portfolio > 0 else 0.0
        returns.append(bar_return)
        timestamps.append(idx)
        prev_portfolio = end_portfolio

    returns_series = pd.Series(returns, index=pd.DatetimeIndex(timestamps))
    stats = compute_backtest_stats(returns_series, closed_trades)

    return {
        "model": model,
        "returns": returns_series,
        "trades": closed_trades,
        "stats": stats,
    }


def compute_backtest_stats(returns: pd.Series, closed_trades: list) -> dict:
    """Compute Sharpe, Sortino, Calmar, and composite score."""
    returns = returns[returns.index.notna()]

    # Trade stats
    n_trades = len(closed_trades)
    if n_trades > 0:
        winners = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
        trade_win_rate = winners / n_trades
        avg_trade_pnl = sum(t["pnl_pct"] for t in closed_trades) / n_trades
    else:
        trade_win_rate = avg_trade_pnl = 0.0

    # Risk-adjusted metrics
    sharpe = float(qs.stats.sharpe(returns, periods=PERIODS_4H))
    sortino = float(qs.stats.sortino(returns, periods=PERIODS_4H))
    calmar = float(qs.stats.calmar(returns))

    # Composite score: 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar
    composite_score = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    return {
        "total_return_pct": float((1 + returns).prod() - 1) * 100,
        "cagr_pct": float(qs.stats.cagr(returns, periods=PERIODS_4H)) * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "composite_score": composite_score,
        "max_drawdown_pct": float(qs.stats.max_drawdown(returns)) * 100,
        "volatility_ann_pct": float(qs.stats.volatility(returns, periods=PERIODS_4H)) * 100,
        "n_trades": n_trades,
        "trade_win_rate_pct": trade_win_rate * 100,
        "avg_trade_pnl_pct": avg_trade_pnl * 100,
        "n_bars": len(returns),
    }


def run_walk_forward_test(
    X_full,
    y_full,
    close_full,
    atr_full,
    allocator_prices,
    feature_name,
    use_macro_filter=False,
    macro_filter_df=None,
    threshold=0.6,
    fee_bps=10,
    n_splits=3,
):
    """
    Walk-forward (time-series cross-validation) backtest.

    Returns:
        dict with aggregated stats and per-fold results
    """
    framework_config = load_framework_config()
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=24)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_full)):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
        test_close = close_full.iloc[test_idx]
        test_atr = atr_full.iloc[test_idx]
        test_oil = None
        test_dxy = None
        if use_macro_filter and macro_filter_df is not None:
            test_oil = macro_filter_df["oil_return_1d"].iloc[test_idx]
            test_dxy = macro_filter_df["dxy_return_1d"].iloc[test_idx]

        # Shared PortfolioAllocator path (weights influence RiskManager sizing)
        portfolio_weight = 1.0
        if allocator_prices is not None and not allocator_prices.empty:
            portfolio_allocator = PortfolioAllocator(config=framework_config)
            train_prices = allocator_prices.iloc[train_idx]
            price_history = {}
            for col, pair in [
                ("btc_close", "BTC/USD"),
                ("eth_close", "ETH/USD"),
                ("sol_close", "SOL/USD"),
            ]:
                if col in train_prices.columns:
                    close_series = train_prices[col].dropna()
                    if not close_series.empty:
                        price_history[pair] = pd.DataFrame({"close": close_series})

            portfolio_allocator.compute_weights(price_history)
            portfolio_weight = portfolio_allocator.get_pair_weight(
                "BTC/USD",
                n_active_pairs=max(len(price_history), 1),
            )

        result = train_and_backtest(
            X_train,
            y_train,
            X_test,
            y_test,
            test_close,
            test_atr,
            portfolio_weight,
            framework_config,
            use_macro_filter,
            test_oil,
            test_dxy,
            threshold,
            fee_bps,
        )
        if result is None:
            continue

        fold_results.append({
            "fold": fold,
            "stats": result["stats"],
            "n_trades": result["stats"]["n_trades"],
        })

    # Aggregate stats
    if not fold_results:
        return None

    agg_stats = {
        "sharpe": np.mean([f["stats"]["sharpe"] for f in fold_results]),
        "sortino": np.mean([f["stats"]["sortino"] for f in fold_results]),
        "calmar": np.mean([f["stats"]["calmar"] for f in fold_results]),
        "composite_score": np.mean([f["stats"]["composite_score"] for f in fold_results]),
        "total_return_pct": np.mean([f["stats"]["total_return_pct"] for f in fold_results]),
        "max_drawdown_pct": np.mean([f["stats"]["max_drawdown_pct"] for f in fold_results]),
        "n_trades_total": sum([f["n_trades"] for f in fold_results]),
        "n_folds": len(fold_results),
    }

    return {
        "feature_set": feature_name,
        "agg_stats": agg_stats,
        "fold_results": fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare alpha strategies")
    parser.add_argument("--start", default="2024-01-01", help="Start date for test period")
    parser.add_argument("--threshold", type=float, default=0.6, help="Signal threshold")
    parser.add_argument("--fee-bps", type=int, default=10, help="Fee in basis points")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of CV folds")
    args = parser.parse_args()

    print("=" * 80)
    print("  ALPHA STRATEGY COMPARISON")
    print("  Composite Score = 0.4*Sortino + 0.3*Sharpe + 0.3*Calmar")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    data = load_all_data()
    btc = data["btc"]
    print(f"  BTC: {len(btc)} bars")

    # Compute all features
    print("\n[2/4] Computing features...")
    feat = compute_base_features(btc, data["eth"], data["sol"])
    feat = compute_macro_features(feat, data["oil"], data["dxy"])
    feat = compute_funding_features(feat, data["btc_funding"], data["eth_funding"], data["sol_funding"])
    print(f"  Total features: {feat.shape}")

    # Create labels (6-bar forward return > 1%)
    print("\n[3/4] Creating labels...")
    horizon = 6
    threshold_label = 0.01
    fwd_ret = feat["close"].shift(-horizon) / feat["close"] - 1
    labels = (fwd_ret >= threshold_label).astype(int)

    X_all = feat.iloc[:-horizon].copy()
    y_all = labels.iloc[:-horizon].copy()
    close_all = feat["close"].iloc[:-horizon].copy()

    # Keep raw closes for shared portfolio allocator in fold-level sizing
    X_all["btc_close"] = feat["close"].iloc[:-horizon]
    if data["eth"] is not None:
        X_all["eth_close"] = data["eth"]["close"].reindex(X_all.index)
    if data["sol"] is not None:
        X_all["sol_close"] = data["sol"]["close"].reindex(X_all.index)

    # Filter to start date
    start_date = pd.Timestamp(args.start, tz="UTC")
    X_all = X_all[X_all.index >= start_date]
    y_all = y_all[y_all.index >= start_date]
    close_all = close_all[close_all.index >= start_date]

    print(f"  Test period: {X_all.index[0]} to {X_all.index[-1]}")
    print(f"  Bars: {len(X_all)} | BUY rate: {y_all.mean():.1%}")

    # Test strategies
    print(f"\n[4/4] Testing alpha strategies ({args.n_splits}-fold CV)...")
    print(f"  Threshold: {args.threshold} | Fee: {args.fee_bps} bps\n")

    strategies = [
        ("Base Indicators Only", BASE_FEATURES),
        ("Base Indicators + Macro Filter", BASE_FEATURES),
        ("Oil + DXY", BASE_FEATURES + OIL_DXY_FEATURES),
        ("Funding Only", FUNDING_FEATURES),
        ("Funding Only + Macro Filter", FUNDING_FEATURES),
        ("Funding + Oil + DXY", BASE_FEATURES + OIL_DXY_FEATURES + FUNDING_FEATURES),
    ]

    results = []
    for strategy_name, feature_cols in strategies:
        print(f"  Testing: {strategy_name} ({len(feature_cols)} features)...", end=" ", flush=True)

        # Filter to available features and drop NaN
        available_cols = [c for c in feature_cols if c in X_all.columns]
        X_strat = X_all[available_cols].copy()
        valid_idx = X_strat.dropna().index
        X_strat = X_strat.loc[valid_idx]
        y_strat = y_all.loc[valid_idx]
        close_strat = close_all.loc[valid_idx]
        atr_strat = X_all["atr_proxy"].loc[valid_idx]
        alloc_cols = [c for c in ("btc_close", "eth_close", "sol_close") if c in X_all.columns]
        alloc_prices = X_all.loc[valid_idx, alloc_cols]
        macro_filter_df = None
        use_macro_filter = "Macro Filter" in strategy_name
        if use_macro_filter:
            if {"oil_return_1d", "dxy_return_1d"}.issubset(X_all.columns):
                macro_filter_df = X_all.loc[valid_idx, ["oil_return_1d", "dxy_return_1d"]]
            else:
                print("SKIP (macro filter features unavailable)")
                continue

        if len(X_strat) < 500:
            print("SKIP (insufficient data)")
            continue

        result = run_walk_forward_test(
            X_strat,
            y_strat,
            close_strat,
            atr_strat,
            alloc_prices,
            strategy_name,
            use_macro_filter,
            macro_filter_df,
            args.threshold,
            args.fee_bps,
            args.n_splits,
        )
        if result is None:
            print("SKIP (training failed)")
            continue

        results.append(result)
        print(f"DONE (Composite: {result['agg_stats']['composite_score']:.3f})")

    # Print comparison report
    print("\n" + "=" * 80)
    print("  RESULTS - Strategy Comparison")
    print("=" * 80)
    print(f"\n  {'Strategy':<30} {'Composite':>10} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'Trades':>8} {'Return%':>9}")
    print("  " + "-" * 98)

    results.sort(key=lambda r: r["agg_stats"]["composite_score"], reverse=True)
    for r in results:
        s = r["agg_stats"]
        marker = " 🏆" if r is results[0] else ""
        print(f"  {r['feature_set']:<30} {s['composite_score']:>10.3f} {s['sharpe']:>8.3f} "
              f"{s['sortino']:>8.3f} {s['calmar']:>8.3f} {s['n_trades_total']:>8} "
              f"{s['total_return_pct']:>8.2f}%{marker}")

    # Winner details
    if results:
        print("\n" + "=" * 80)
        print("  WINNER DETAILS")
        print("=" * 80)
        winner = results[0]
        s = winner["agg_stats"]
        print(f"\n  Strategy: {winner['feature_set']}")
        print(f"  Composite Score: {s['composite_score']:.3f}")
        print(f"    ↳ Sortino (40%):  {s['sortino']:.3f} × 0.40 = {s['sortino']*0.4:.3f}")
        print(f"    ↳ Sharpe  (30%):  {s['sharpe']:.3f} × 0.30 = {s['sharpe']*0.3:.3f}")
        print(f"    ↳ Calmar  (30%):  {s['calmar']:.3f} × 0.30 = {s['calmar']*0.3:.3f}")
        print(f"\n  Total Return: {s['total_return_pct']:+.2f}%")
        print(f"  Max Drawdown: {s['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades: {s['n_trades_total']} ({s['n_folds']} folds)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "alpha_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_DIR / 'alpha_comparison.json'}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

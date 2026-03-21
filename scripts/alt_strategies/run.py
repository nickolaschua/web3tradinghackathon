#!/usr/bin/env python3
"""
Run alternative strategy backtests with configurable parameters.

Usage:
    # Run all strategies on full OOS period
    python scripts/alt_strategies/run.py --strategy all

    # Run XSM (ML-enhanced) with model
    python scripts/alt_strategies/run.py --strategy xsm --xsm-model models/xgb_xsm_top5_h96.pkl

    # Run XSM rule-based (no ML)
    python scripts/alt_strategies/run.py --strategy xsm --rule-based

    # Test on 10-day windows
    python scripts/alt_strategies/run.py --strategy all --windows 2025-02-15,2025-01-20,2024-08-05

    # Sweep XSM top-K
    python scripts/alt_strategies/run.py --strategy xsm --sweep-top-k 3,5,7,10

    # Disable circuit breaker (better for 10-day tests)
    python scripts/alt_strategies/run.py --strategy all --no-cb

    # Custom spread pair
    python scripts/alt_strategies/run.py --strategy spread --spread-a BTC/USD --spread-b PAXG/USD
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import (
    compute_btc_context_features,
    compute_cross_asset_features,
    compute_features,
)
from scripts.alt_strategies.engine import (
    BacktestConfig,
    BacktestEngine,
    compute_report,
    print_report,
)
from scripts.alt_strategies.strategies import (
    CrossSectionalMomentum,
    RegimeFilteredXGB,
    SpreadReversion,
    VolatilityBreakout,
    XSM_FEATURE_COLS,
    build_xsm_panel,
)

DATA_DIR = project_root / "data"
MODEL_DIR = project_root / "models"
RESULTS_DIR = project_root / "research_results"

_BINANCE_TO_ROOSTOO: dict[str, str] = {
    "BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD", "BNBUSDT": "BNB/USD",
    "SOLUSDT": "SOL/USD", "ADAUSDT": "ADA/USD", "AVAXUSDT": "AVAX/USD",
    "DOGEUSDT": "DOGE/USD", "LINKUSDT": "LINK/USD", "DOTUSDT": "DOT/USD",
    "UNIUSDT": "UNI/USD", "XRPUSDT": "XRP/USD", "LTCUSDT": "LTC/USD",
    "AAVEUSDT": "AAVE/USD", "CRVUSDT": "CRV/USD", "NEARUSDT": "NEAR/USD",
    "FILUSDT": "FIL/USD", "FETUSDT": "FET/USD", "HBARUSDT": "HBAR/USD",
    "ZECUSDT": "ZEC/USD", "ZENUSDT": "ZEN/USD", "CAKEUSDT": "CAKE/USD",
    "PAXGUSDT": "PAXG/USD", "XLMUSDT": "XLM/USD", "TRXUSDT": "TRX/USD",
    "CFXUSDT": "CFX/USD", "SHIBUSDT": "SHIB/USD", "ICPUSDT": "ICP/USD",
    "APTUSDT": "APT/USD", "ARBUSDT": "ARB/USD", "SUIUSDT": "SUI/USD",
    "FLOKIUSDT": "FLOKI/USD", "PEPEUSDT": "PEPE/USD",
    "PENDLEUSDT": "PENDLE/USD", "WLDUSDT": "WLD/USD", "SEIUSDT": "SEI/USD",
    "BONKUSDT": "BONK/USD", "WIFUSDT": "WIF/USD", "ENAUSDT": "ENA/USD",
    "TAOUSDT": "TAO/USD",
}

OOS_START = "2024-01-01"

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_data() -> dict[str, pd.DataFrame]:
    data = {}
    for path in sorted(DATA_DIR.glob("*_15m.parquet")):
        sym = path.stem.replace("_15m", "")
        pair = _BINANCE_TO_ROOSTOO.get(sym)
        if pair is None:
            continue
        df = pd.read_parquet(path)
        df.columns = df.columns.str.lower()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        data[pair] = df
    print(f"Loaded {len(data)} coins")
    return data


def compute_all_features(
    raw_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Compute features for all coins, with BTC getting full context features."""
    btc_raw = raw_data.get("BTC/USD")
    eth_raw = raw_data.get("ETH/USD")
    sol_raw = raw_data.get("SOL/USD")

    features = {}
    for pair, df in raw_data.items():
        feat = compute_features(df)

        if pair == "BTC/USD" and eth_raw is not None and sol_raw is not None:
            feat = compute_cross_asset_features(feat, {"ETH/USD": eth_raw, "SOL/USD": sol_raw})
            feat = compute_btc_context_features(feat, eth_raw, sol_raw, window=2880)
        elif btc_raw is not None and eth_raw is not None:
            feat = compute_cross_asset_features(feat, {"BTC/USD": btc_raw, "ETH/USD": eth_raw})

        # Trailing returns needed by strategies (XSM ranking, volatility breakout, etc.)
        close = feat["close"]
        feat["return_1h"] = close.pct_change(4)
        feat["return_4h"] = close.pct_change(16)
        feat["return_1d"] = close.pct_change(96)
        feat["return_3d"] = close.pct_change(288)

        feat = feat.dropna(subset=["RSI_14", "atr_proxy", "EMA_20", "EMA_50"])
        if len(feat) > 500:
            features[pair] = feat

    print(f"Computed features for {len(features)} coins")
    return features


# ── BTC XGBoost Probability Map ───────────────────────────────────────────────

def build_btc_proba_map(
    btc_features: pd.DataFrame,
    model_path: str = "models/xgb_btc_15m_iter5.pkl",
    oos_start: str = OOS_START,
) -> dict:
    """Pre-compute BTC XGBoost P(BUY) for all OOS bars. Returns {timestamp: float}."""
    path = project_root / model_path
    if not path.exists():
        print(f"  BTC model not found at {path}, skipping regime strategy")
        return {}

    with open(path, "rb") as f:
        model = pickle.load(f)

    oos_ts = pd.Timestamp(oos_start, tz="UTC")
    oos = btc_features[btc_features.index >= oos_ts].copy()

    feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else model.get_booster().feature_names
    for col in feature_names:
        if col not in oos.columns:
            oos[col] = np.nan

    X = oos[list(feature_names)]
    probas = model.predict_proba(X)[:, 1]

    proba_map = dict(zip(oos.index, probas))
    print(f"  BTC proba map: {len(proba_map):,} bars, "
          f"mean P={np.mean(probas):.3f}, >0.60: {(probas > 0.60).sum()}")
    return proba_map


# ── XSM Probability Map ──────────────────────────────────────────────────────

def build_xsm_proba_map(
    all_features: dict[str, pd.DataFrame],
    model_path: str,
    oos_start: str = OOS_START,
) -> dict[str, dict]:
    """Pre-compute XSM probabilities for all coins at all OOS timestamps."""
    path = project_root / model_path if not Path(model_path).is_absolute() else Path(model_path)
    if not path.exists():
        print(f"  XSM model not found at {path}")
        return {}

    with open(path, "rb") as f:
        model = pickle.load(f)

    oos_ts = pd.Timestamp(oos_start, tz="UTC")
    btc_close = all_features.get("BTC/USD", pd.DataFrame()).get("close")

    print("  Building XSM panel for prediction...")
    # Only compute panel for OOS period to save memory
    oos_features = {}
    for pair, feat_df in all_features.items():
        # Include some pre-OOS data for trailing returns warmup
        warmup_start = oos_ts - pd.Timedelta(days=10)
        oos = feat_df[feat_df.index >= warmup_start]
        if len(oos) > 100:
            oos_features[pair] = oos

    panel = build_xsm_panel(oos_features, btc_close)
    panel = panel[panel.index.get_level_values(0) >= oos_ts]

    # Predict
    feature_cols = [c for c in XSM_FEATURE_COLS if c in panel.columns]
    X = panel[feature_cols].copy()
    for col in XSM_FEATURE_COLS:
        if col not in X.columns:
            X[col] = np.nan

    X = X[XSM_FEATURE_COLS]
    valid_mask = X.notna().sum(axis=1) >= 10
    X_valid = X.loc[valid_mask]

    probas = model.predict_proba(X_valid)[:, 1]

    # Build lookup: {pair: {timestamp: proba}}
    proba_map: dict[str, dict] = {}
    for (ts, pair), p in zip(X_valid.index, probas):
        proba_map.setdefault(pair, {})[ts] = float(p)

    n_pairs = len(proba_map)
    n_total = sum(len(v) for v in proba_map.values())
    print(f"  XSM proba map: {n_pairs} coins, {n_total:,} total predictions, "
          f"mean P={np.mean(probas):.3f}")
    return proba_map


# ── Strategy Builders ─────────────────────────────────────────────────────────

def build_strategies(
    args,
    all_features: dict[str, pd.DataFrame],
) -> list:
    """Build strategy instances based on CLI args."""
    strategies = []
    strategy_names = args.strategy.split(",") if args.strategy != "all" else [
        "xsm", "spread", "regime", "vol_breakout",
    ]

    if "xsm" in strategy_names:
        xsm_proba_map = None
        if not args.rule_based and args.xsm_model:
            xsm_proba_map = build_xsm_proba_map(all_features, args.xsm_model, args.oos_start)
            if not xsm_proba_map:
                print("  Falling back to rule-based XSM")

        strategies.append(CrossSectionalMomentum(
            xsm_proba_map=xsm_proba_map,
            top_k=args.top_k,
            buy_threshold=args.xsm_threshold,
            exit_threshold=args.xsm_exit_threshold,
            rebalance_bars=args.rebalance_bars,
            excluded_pairs={"PAXG/USD"},
        ))
        mode = "ML-enhanced" if xsm_proba_map else "rule-based"
        print(f"  + XSM ({mode}, top-{args.top_k})")

    if "spread" in strategy_names:
        strategies.append(SpreadReversion(
            pair_a=args.spread_a,
            pair_b=args.spread_b,
            lookback=args.spread_lookback,
            entry_z=args.spread_entry_z,
            exit_z=args.spread_exit_z,
        ))
        print(f"  + Spread Reversion ({args.spread_a}/{args.spread_b})")

    if "regime" in strategy_names:
        btc_proba_map = {}
        if "BTC/USD" in all_features:
            btc_proba_map = build_btc_proba_map(all_features["BTC/USD"], oos_start=args.oos_start)

        high_beta_pairs = [
            "SOL/USD", "DOGE/USD", "AVAX/USD", "NEAR/USD",
            "SUI/USD", "PEPE/USD", "FLOKI/USD", "BONK/USD",
            "WIF/USD", "FET/USD", "ARB/USD", "APT/USD",
        ]
        strategies.append(RegimeFilteredXGB(
            btc_proba_map=btc_proba_map,
            regime_threshold=args.regime_threshold,
            exit_threshold=args.regime_exit_threshold,
            top_k=args.regime_top_k,
            target_pairs=high_beta_pairs,
        ))
        print(f"  + Regime-Filtered XGB (threshold={args.regime_threshold})")

    if "vol_breakout" in strategy_names:
        strategies.append(VolatilityBreakout(
            expansion_factor=args.vol_expansion,
            lookback_bars=args.vol_lookback,
            volume_threshold=args.vol_volume_threshold,
        ))
        print(f"  + Volatility Breakout (expansion={args.vol_expansion}x)")

    return strategies


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run alt strategy backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # General
    parser.add_argument("--strategy", type=str, default="all",
                        help="Strategy to run: xsm, spread, regime, vol_breakout, all, or comma-separated")
    parser.add_argument("--oos-start", type=str, default=OOS_START)
    parser.add_argument("--windows", type=str, default=None,
                        help="Comma-separated window start dates for 10-day tests")
    parser.add_argument("--window-days", type=int, default=10)
    parser.add_argument("--no-cb", action="store_true", help="Disable circuit breaker")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)

    # XSM
    parser.add_argument("--xsm-model", type=str, default=None,
                        help="Path to trained XSM model (.pkl)")
    parser.add_argument("--rule-based", action="store_true", help="Use rule-based XSM (no ML)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--xsm-threshold", type=float, default=0.50)
    parser.add_argument("--xsm-exit-threshold", type=float, default=0.30)
    parser.add_argument("--rebalance-bars", type=int, default=96)
    parser.add_argument("--sweep-top-k", type=str, default=None,
                        help="Comma-separated top-K values to sweep")

    # Spread
    parser.add_argument("--spread-a", type=str, default="BTC/USD")
    parser.add_argument("--spread-b", type=str, default="PAXG/USD")
    parser.add_argument("--spread-lookback", type=int, default=672)
    parser.add_argument("--spread-entry-z", type=float, default=2.0)
    parser.add_argument("--spread-exit-z", type=float, default=0.5)

    # Regime
    parser.add_argument("--regime-threshold", type=float, default=0.60)
    parser.add_argument("--regime-exit-threshold", type=float, default=0.40)
    parser.add_argument("--regime-top-k", type=int, default=3)

    # Vol Breakout
    parser.add_argument("--vol-expansion", type=float, default=1.5)
    parser.add_argument("--vol-lookback", type=int, default=96)
    parser.add_argument("--vol-volume-threshold", type=float, default=1.3)

    args = parser.parse_args()
    t0 = time.time()

    # 1. Load data
    print("=" * 60)
    print("  ALT STRATEGIES BACKTEST")
    print("=" * 60)
    raw_data = load_all_data()

    # 2. Compute features
    print("\nComputing features...")
    all_features = compute_all_features(raw_data)

    # 3. Build OOS index
    oos_ts = pd.Timestamp(args.oos_start, tz="UTC")
    oos_features = {}
    for pair, feat_df in all_features.items():
        oos = feat_df[feat_df.index >= oos_ts]
        if len(oos) > 100:
            oos_features[pair] = oos

    common_index = pd.DatetimeIndex(sorted(
        set().union(*(set(df.index) for df in oos_features.values()))
    ))
    print(f"\nOOS: {common_index[0]} to {common_index[-1]}")
    print(f"     {len(common_index):,} bars, {len(oos_features)} coins")

    # 4. Config
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        enable_cb=not args.no_cb,
    )

    # 5. Sweep or single run
    if args.sweep_top_k:
        sweep_values = [int(k) for k in args.sweep_top_k.split(",")]
        print(f"\n── Sweeping top-K: {sweep_values} ──")
        all_reports = []

        for k in sweep_values:
            print(f"\n  top-K = {k}")
            args.top_k = k
            strategies = build_strategies(args, all_features)
            engine = BacktestEngine(config, strategies)

            if args.windows:
                starts = args.windows.split(",")
                results = engine.run_windows(oos_features, starts, args.window_days)
            else:
                result = engine.run(oos_features, common_index, label=f"top-{k}")
                results = [result]

            for r in results:
                report = compute_report(r, config.initial_capital)
                report["sweep_param"] = f"top_k={k}"
                all_reports.append(report)
                print_report(report)

        # Summary table
        print("\n" + "=" * 60)
        print("  SWEEP SUMMARY")
        print("=" * 60)
        print(f"  {'Config':20s} {'Trades':>7s} {'Return':>8s} {'Sharpe':>8s} "
              f"{'Sortino':>8s} {'MaxDD':>8s} {'Active':>8s}")
        for r in all_reports:
            print(f"  {r.get('sweep_param',''):20s} {r['n_trades']:>7d} "
                  f"{r['total_return_pct']:>+7.2f}% {r['sharpe']:>8.3f} "
                  f"{r['sortino']:>8.3f} {r['max_drawdown_pct']:>7.2f}% "
                  f"{r['active_days']:>4d}/{r['total_days']}")

    else:
        # Single run
        print("\nBuilding strategies...")
        strategies = build_strategies(args, all_features)

        if not strategies:
            print("No strategies selected!")
            sys.exit(1)

        engine = BacktestEngine(config, strategies)

        if args.windows:
            starts = args.windows.split(",")
            print(f"\nRunning {len(starts)} window(s) of {args.window_days} days...")
            results = engine.run_windows(oos_features, starts, args.window_days)
            all_reports = []
            for r in results:
                report = compute_report(r, config.initial_capital)
                all_reports.append(report)
                print_report(report)

            if len(all_reports) > 1:
                print("\n" + "=" * 60)
                print("  WINDOW SUMMARY")
                print("=" * 60)
                print(f"  {'Window':30s} {'Trades':>7s} {'Return':>8s} {'Sharpe':>8s} "
                      f"{'Active':>8s} {'8/10':>5s}")
                for r in all_reports:
                    print(f"  {r['label']:30s} {r['n_trades']:>7d} "
                          f"{r['total_return_pct']:>+7.2f}% {r['sharpe']:>8.3f} "
                          f"{r['active_days']:>4d}/{r['total_days']}   "
                          f"{'PASS' if r['meets_8_of_10'] else 'FAIL'}")

        else:
            print(f"\nRunning full OOS backtest ({args.oos_start} to present)...")
            result = engine.run(oos_features, common_index)
            report = compute_report(result, config.initial_capital)
            print_report(report)

            # Save results
            RESULTS_DIR.mkdir(exist_ok=True)
            strat_name = args.strategy.replace(",", "_")
            out_path = RESULTS_DIR / f"alt_{strat_name}_backtest.json"
            serializable = {k: v for k, v in report.items()
                           if not isinstance(v, (np.floating, np.integer))}
            with open(out_path, "w") as f:
                json.dump(serializable, f, indent=2, default=str)
            print(f"\nResults saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()

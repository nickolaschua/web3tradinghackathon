"""
Iteration 4 — Feature Ablation: drop weakest baseline features, add cross-sectional rank.

Strategy:
  1. Load current best model → dump ALL 14 feature importances (we only logged top 5)
  2. Identify the N weakest features by importance
  3. Replace them with ret_168bar_rank + ret_168bar_zscore (IC=0.09, best in iter 2)
  4. Walk-forward CV comparison across 4 configs

Configs:
  A: Current 14 features (control)
  B: Drop bottom 3 weakest + add rank168 x2  → 13 features
  C: Drop bottom 4 weakest + add rank168 x2  → 12 features
  D: Drop EMA_50 only    + add rank168 x2  → 15 features
     (EMA_50 is the reason rank was rejected in iter 2 — explicit redundancy test)

Cross-sectional rank requires the full 67-coin universe (all data/*_4h.parquet files).
Rank is computed at 168-bar (28-day) lookback only — that was the strongest IC (0.09).

Data discipline:
  - Walk-forward CV uses train+val (< 2024-01-01) — same as all prior training scripts
  - Test set (>= 2024-01-01) is LOCKED — only touched by --save-best final evaluation
  - No IC re-testing needed: rank features already passed IC in iter 2

Usage:
    python scripts/train_ablation_rank.py            # CV only, all 4 configs
    python scripts/train_ablation_rank.py --save-best  # also train + save best final model
"""
from __future__ import annotations

import argparse
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

from bot.data.features import (
    compute_btc_context_features,
    compute_cross_asset_features,
    compute_features,
)
from bot.data.universe_features import compute_cross_sectional_ranks

DATA_DIR = project_root / "data"
CURRENT_MODEL = project_root / "models" / "xgb_btc_4h_lead_lag.pkl"

CURRENT_14 = [
    "atr_proxy", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "EMA_20", "EMA_50", "ema_slope",
    "eth_return_lag1", "eth_return_lag2", "sol_return_lag1", "sol_return_lag2",
    "eth_btc_corr", "eth_btc_beta",
]

RANK_FEATURES = ["ret_168bar_rank", "ret_168bar_zscore"]

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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_current_importances() -> pd.Series:
    """Return all 14 feature importances from the current best model, sorted desc."""
    with open(CURRENT_MODEL, "rb") as f:
        model = pickle.load(f)
    return pd.Series(
        model.feature_importances_, index=model.feature_names_in_
    ).sort_values(ascending=False)


def load_universe() -> dict[str, pd.DataFrame]:
    """Load all available 4H parquets as the coin universe for rank computation."""
    coin_dfs: dict[str, pd.DataFrame] = {}
    for path in sorted(DATA_DIR.glob("*_4h.parquet")):
        symbol = path.stem.replace("_4h", "")
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            df.columns = df.columns.str.lower()
            if "close" in df.columns and len(df) >= 200:
                coin_dfs[symbol] = df
        except Exception as exc:
            print(f"  WARNING: could not load {path.name}: {exc}")
    return coin_dfs


def build_feature_matrix(universe_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build BTC feature matrix: current 14 features + rank features.

    Does NOT call dropna() globally — each experiment config handles its own NaN
    via prepare_training_data, so configs with/without rank features get fair
    sample sizes (rank features have 168-bar extra warmup).
    """
    btc = universe_dfs.get("BTCUSDT")
    eth = universe_dfs.get("ETHUSDT")
    sol = universe_dfs.get("SOLUSDT")

    if btc is None or eth is None or sol is None:
        raise RuntimeError("BTCUSDT, ETHUSDT, or SOLUSDT not found in universe_dfs")

    # Standard features (14)
    feat = compute_features(btc.copy())
    feat = compute_cross_asset_features(feat, {"ETH/USD": eth, "SOL/USD": sol})
    feat = compute_btc_context_features(feat, eth, sol)

    # Cross-sectional rank (168-bar only — strongest IC in iter 2)
    ranked_dfs = compute_cross_sectional_ranks(universe_dfs, lookbacks=[168])
    btc_ranked = ranked_dfs.get("BTCUSDT")
    if btc_ranked is not None:
        for col in RANK_FEATURES:
            if col in btc_ranked.columns:
                feat[col] = btc_ranked[col].reindex(feat.index)
            else:
                print(f"  WARNING: rank column {col} not found in BTCUSDT ranked df")
    else:
        print("  WARNING: BTCUSDT not found in ranked_dfs — rank features will be missing")

    return feat


# ── Training helpers ──────────────────────────────────────────────────────────

def prepare_training_data(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int = 6,
    threshold: float = 0.00015,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X, y for a given feature set. Handles NaN per-config."""
    fwd_ret = feat_df["close"].shift(-horizon) / feat_df["close"] - 1
    labels = (fwd_ret >= threshold).astype(int)

    available = [c for c in feature_cols if c in feat_df.columns]
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        print(f"    Missing features (skipped): {missing}")

    X = feat_df[available].iloc[:-horizon]
    y = labels.iloc[:-horizon]

    valid = X.notna().all(axis=1)
    return X[valid], y[valid]


def run_cv(X: pd.DataFrame, y: pd.Series, label: str = "") -> float:
    """Walk-forward CV on train+val (< 2024-01-01). Returns mean AP."""
    # Restrict to train+val split — same boundary as all prior scripts
    cutoff = pd.Timestamp("2024-01-01", tz="UTC")
    X = X[X.index < cutoff]
    y = y[y.index < cutoff]

    tscv = TimeSeriesSplit(n_splits=5, gap=24)
    aps: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        n_pos = int(y_tr.sum())
        if n_pos == 0:
            print(f"    [{label}] Fold {fold}: SKIP — no positive labels")
            continue

        spw = (len(y_tr) - n_pos) / n_pos
        model = xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": spw})
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        proba = model.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, proba)
        aps.append(ap)
        print(f"    [{label}] Fold {fold}: AP={ap:.3f}  n_train={len(X_tr)}  n_val={len(X_va)}")

    mean_ap = float(np.mean(aps)) if aps else float("nan")
    print(f"    [{label}] Mean AP = {mean_ap:.3f}  ({len(aps)} folds)")
    return mean_ap


def train_final_and_save(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path,
    horizon: int = 6,
    threshold: float = 0.00015,
    test_cutoff: str = "2024-01-01",
) -> float:
    """Train final model on train+val, evaluate on locked test set, save pkl."""
    X, y = prepare_training_data(feat_df, feature_cols, horizon, threshold)
    cutoff = pd.Timestamp(test_cutoff, tz="UTC")

    X_tv, y_tv = X[X.index < cutoff], y[y.index < cutoff]
    X_te, y_te = X[X.index >= cutoff], y[y.index >= cutoff]

    n_pos = int(y_tv.sum())
    spw = (len(y_tv) - n_pos) / n_pos if n_pos > 0 else 1.0

    params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    params.update({"n_estimators": 300, "scale_pos_weight": spw})

    print(f"\n  Training final model ({len(X_tv)} train+val bars, {len(feature_cols)} features)...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_tv, y_tv, verbose=False)

    ap_test = float("nan")
    if len(X_te) > 0 and y_te.sum() > 0:
        proba = model.predict_proba(X_te)[:, 1]
        ap_test = average_precision_score(y_te, proba)
        f1_test = f1_score(y_te, (proba >= 0.5).astype(int), zero_division=0)
        print(f"  Test set: AP={ap_test:.3f}  F1={f1_test:.3f}  ({len(X_te)} bars)")

    importances = pd.Series(
        model.feature_importances_, index=model.feature_names_in_
    ).sort_values(ascending=False)
    print("  Feature importances:")
    for feat, imp in importances.items():
        print(f"    {feat:<25} {imp:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {output_path}")

    return ap_test


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 65)
    print("Iteration 4: Ablation — Drop Weak Features + Cross-Sectional Rank")
    print("=" * 65)

    # ── Step 1: Current model importances (all 14) ────────────────────────────
    print("\nStep 1: Current model — all 14 feature importances")
    importances = load_current_importances()
    for i, (feat, imp) in enumerate(importances.items(), 1):
        print(f"  {i:>2}. {feat:<25} {imp:.4f}")

    bottom_3 = list(importances.tail(3).index)
    bottom_4 = list(importances.tail(4).index)
    print(f"\n  Bottom 3 (to drop in Config B): {bottom_3}")
    print(f"  Bottom 4 (to drop in Config C): {bottom_4}")

    # ── Step 2: Load universe + build feature matrix ──────────────────────────
    print(f"\nStep 2: Loading universe from {DATA_DIR}...")
    universe_dfs = load_universe()
    print(f"  Loaded {len(universe_dfs)} coins")

    print("\nStep 3: Building BTC feature matrix (14 features + rank features)...")
    feat = build_feature_matrix(universe_dfs)
    rank_present = all(c in feat.columns for c in RANK_FEATURES)
    print(f"  Feature matrix shape: {feat.shape}  |  Rank features present: {rank_present}")
    print(f"  Date range: {feat.index[0].date()} — {feat.index[-1].date()}")

    # ── Step 3: Define configs ────────────────────────────────────────────────
    configs: dict[str, list[str]] = {
        "A: current 14":         CURRENT_14,
        "B: drop bottom-3+rank": [f for f in CURRENT_14 if f not in bottom_3] + RANK_FEATURES,
        "C: drop bottom-4+rank": [f for f in CURRENT_14 if f not in bottom_4] + RANK_FEATURES,
        "D: drop EMA_50+rank":   [f for f in CURRENT_14 if f != "EMA_50"] + RANK_FEATURES,
    }

    # ── Step 4: Walk-forward CV ───────────────────────────────────────────────
    print("\nStep 4: Walk-forward CV (5 folds, train+val < 2024-01-01)")
    print("-" * 65)
    cv_results: dict[str, float] = {}

    for name, cols in configs.items():
        available = [c for c in cols if c in feat.columns]
        print(f"\n  Config {name} — {len(available)} features:")
        print(f"    {available}")
        X, y = prepare_training_data(feat, available)
        print(f"    Training bars: {len(X)}  |  BUY rate: {y.mean():.1%}")
        cv_results[name] = run_cv(X, y, label=name.split(":")[0].strip())

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    baseline_ap = cv_results["A: current 14"]
    print(f"{'Config':<35} | {'Features':>8} | {'Mean AP':>8} | {'vs A':>7}")
    print("-" * 65)
    for name, ap in cv_results.items():
        n_feat = len([c for c in configs[name] if c in feat.columns])
        delta = ap - baseline_ap
        delta_str = f"{delta:+.3f}" if name != "A: current 14" else "  (base)"
        print(f"  {name:<33} | {n_feat:>8} | {ap:>8.3f} | {delta_str:>7}")

    best_name = max(cv_results, key=cv_results.get)
    best_ap = cv_results[best_name]
    best_delta = best_ap - baseline_ap

    print()
    if best_delta > 0.002:
        print(f"VERDICT: {best_name} improves by +{best_delta:.3f} — worth training final model")
        print("  Run with --save-best to train on full train+val and evaluate on test set")
    elif best_delta > 0:
        print(f"VERDICT: marginal improvement ({best_delta:+.3f}) — within CV noise, keep current model")
    else:
        print(f"VERDICT: no improvement ({best_delta:+.3f}) — cross-sectional rank redundant with current features")
        print("  Consistent with Iteration 2 finding: EMA_50 already encodes the trend information.")

    # ── Step 6: Optional — train + save best final model ─────────────────────
    if args.save_best:
        if best_delta <= 0.002:
            print(f"\n--save-best: best config only {best_delta:+.3f} vs baseline — skipping save")
        else:
            best_cols = [c for c in configs[best_name] if c in feat.columns]
            out_path = project_root / "models" / "xgb_btc_4h_ablation.pkl"
            print(f"\n--save-best: training final model for {best_name}...")
            ap_test = train_final_and_save(feat, best_cols, out_path)
            print(f"\nFinal model test AP: {ap_test:.3f}  (current best: 0.531)")
            print(f"Saved: {out_path}")

    print("\nNext: paste results into research/iteration_log.md (Iteration 4 section)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iteration 4: Feature ablation — drop weak baseline features, add cross-sectional rank"
    )
    p.add_argument(
        "--save-best",
        action="store_true",
        help="If best config improves CV AP by >0.002, train final model and save pkl",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()

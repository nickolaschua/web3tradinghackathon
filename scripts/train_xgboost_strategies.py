#!/usr/bin/env python3
"""
Train XGBoost models for all trading strategies.

Creates ML models that predict future returns, then use predictions as trading signals.

Strategies:
1. Mean Reversion - predict reversals from oversold/overbought
2. Momentum - predict trend continuation
3. Volatility Breakout - predict expansion after compression
4. RSI Divergence - predict reversals from divergences
5. Always In Market - predict general trend direction

Usage:
  python scripts/train_xgboost_strategies.py --strategy all
  python scripts/train_xgboost_strategies.py --strategy mean_reversion
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features_15m import prepare_15m_features

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def create_labels_mean_reversion(df: pd.DataFrame, forward_bars: int = 4) -> pd.Series:
    """
    Label mean reversion opportunities.

    BUY (1) when: oversold conditions + positive future return
    SELL (0) when: overbought conditions or negative future return
    """
    # Future return
    future_return = df["close"].pct_change(forward_bars).shift(-forward_bars)

    # Oversold conditions
    rsi = df.get("rsi", 50)
    bb_pos = df.get("bb_pos", 0.5)

    oversold = (rsi < 30) | (bb_pos < 0.2)
    overbought = (rsi > 70) | (bb_pos > 0.8)

    # Label: BUY if oversold AND positive future return
    labels = np.where(
        oversold & (future_return > 0.003),  # >0.3% gain
        1,
        0
    )

    return pd.Series(labels, index=df.index)


def create_labels_momentum(df: pd.DataFrame, forward_bars: int = 4) -> pd.Series:
    """
    Label momentum continuation.

    BUY (1) when: positive momentum + continues upward
    """
    future_return = df["close"].pct_change(forward_bars).shift(-forward_bars)

    # Current momentum
    momentum_12 = df.get("btc_momentum_12", 0)
    ema_fast = df.get("ema_20", 0)
    ema_slow = df.get("ema_50", 0)

    bullish_momentum = (momentum_12 > 0.003) & (ema_fast > ema_slow)

    labels = np.where(
        bullish_momentum & (future_return > 0.003),
        1,
        0
    )

    return pd.Series(labels, index=df.index)


def create_labels_volatility_breakout(df: pd.DataFrame, forward_bars: int = 4) -> pd.Series:
    """
    Label volatility breakout opportunities.

    BUY (1) when: low volatility expanding + positive breakout
    """
    future_return = df["close"].pct_change(forward_bars).shift(-forward_bars)

    bb_width = df.get("bb_width", 0.1)
    volume_ratio = df.get("volume_ratio", 1.0)

    # Volatility compression then expansion
    bb_width_sma = bb_width.rolling(20).mean()
    compressed = bb_width < bb_width_sma * 0.8
    volume_spike = volume_ratio > 1.5

    labels = np.where(
        compressed & volume_spike & (future_return > 0.005),  # >0.5% gain
        1,
        0
    )

    return pd.Series(labels, index=df.index)


def create_labels_rsi_divergence(df: pd.DataFrame, forward_bars: int = 8) -> pd.Series:
    """
    Label RSI divergence reversals.

    BUY (1) when: price makes lower low but RSI doesn't (bullish divergence)
    """
    future_return = df["close"].pct_change(forward_bars).shift(-forward_bars)

    rsi = df.get("rsi", 50)
    close = df["close"]

    # Detect divergences (simplified - real divergence detection is complex)
    # Look for oversold RSI with positive future return
    oversold_divergence = (rsi < 40) & (rsi.diff() > 0) & (close.diff() < 0)

    labels = np.where(
        oversold_divergence & (future_return > 0.005),
        1,
        0
    )

    return pd.Series(labels, index=df.index)


def create_labels_trend_following(df: pd.DataFrame, forward_bars: int = 4) -> pd.Series:
    """
    Label trend-following (always in market) opportunities.

    BUY (1) when: uptrend continues
    """
    future_return = df["close"].pct_change(forward_bars).shift(-forward_bars)

    ema_20 = df.get("ema_20", 0)
    ema_50 = df.get("ema_50", 0)

    uptrend = ema_20 > ema_50

    labels = np.where(
        uptrend & (future_return > 0.002),  # >0.2% gain
        1,
        0
    )

    return pd.Series(labels, index=df.index)


def prepare_training_data(df: pd.DataFrame, strategy_name: str):
    """Prepare features and labels for training."""

    # Select features (exclude OHLCV and label columns)
    feature_cols = [
        "rsi", "rsi_7", "macd", "macd_hist", "macd_signal",
        "ema_20", "ema_50", "bb_upper", "bb_lower", "bb_width", "bb_pos",
        "volume_ma_20", "volume_ratio", "btc_return", "returns_std_20",
        "btc_momentum_12", "eth_return", "eth_momentum_12", "btc_eth_corr",
        "sol_return", "sol_momentum_12", "btc_sol_corr", "btc_funding_zscore",
        "atr_proxy"
    ]

    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    X = df[available_features].copy()

    # Create labels based on strategy
    if strategy_name == "mean_reversion":
        y = create_labels_mean_reversion(df)
    elif strategy_name == "momentum":
        y = create_labels_momentum(df)
    elif strategy_name == "volatility_breakout":
        y = create_labels_volatility_breakout(df)
    elif strategy_name == "rsi_divergence":
        y = create_labels_rsi_divergence(df)
    elif strategy_name == "trend_following":
        y = create_labels_trend_following(df)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Drop rows with NaN
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    return X, y, available_features


def train_xgboost_model(X_train, y_train, X_val, y_val, strategy_name: str):
    """Train XGBoost classifier with validation."""

    print(f"\nTraining {strategy_name} model...")
    print(f"  Training samples: {len(X_train)} (BUY: {y_train.sum()}, {y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"  Validation samples: {len(X_val)} (BUY: {y_val.sum()}, {y_val.sum()/len(y_val)*100:.1f}%)")

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if y_train.sum() > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=20,
        tree_method='hist'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Validation metrics
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    print(f"\n  Validation Performance:")
    print(f"  AUC: {roc_auc_score(y_val, y_proba):.4f}")
    print(f"  Accuracy: {(y_pred == y_val).mean():.4f}")
    print(f"  Predicted BUY: {y_pred.sum()} ({y_pred.sum()/len(y_pred)*100:.1f}%)")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost models for trading strategies")
    parser.add_argument(
        "--strategy",
        choices=["mean_reversion", "momentum", "volatility_breakout", "rsi_divergence", "trend_following", "all"],
        required=True,
        help="Strategy to train (or 'all' for all strategies)"
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date for training data")
    parser.add_argument("--end", default="2026-03-01", help="End date for training data")
    args = parser.parse_args()

    print("="*70)
    print("  XGBoost Strategy Training Pipeline")
    print("="*70)

    # Load data
    print("\nLoading 15m features...")
    df = prepare_15m_features(
        btc_path="research_data/BTCUSDT_15m.parquet",
        eth_path="research_data/ETHUSDT_15m.parquet",
        sol_path="research_data/SOLUSDT_15m.parquet",
        funding_path="research_data/BTCUSDT_funding.parquet",
        start=args.start,
        end=args.end,
    )

    print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Define strategies to train
    strategies = []
    if args.strategy == "all":
        strategies = ["mean_reversion", "momentum", "volatility_breakout", "rsi_divergence", "trend_following"]
    else:
        strategies = [args.strategy]

    # Train each strategy
    for strategy_name in strategies:
        print(f"\n{'='*70}")
        print(f"  Training: {strategy_name}")
        print(f"{'='*70}")

        # Prepare data
        X, y, feature_names = prepare_training_data(df, strategy_name)

        print(f"\nFeatures used: {len(feature_names)}")
        print(f"Total samples: {len(X)}")

        # Time-based split: 60% train, 20% val, 20% test
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        print(f"\nTrain period: {X_train.index[0]} to {X_train.index[-1]}")
        print(f"Val period: {X_val.index[0]} to {X_val.index[-1]}")
        print(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")

        # Train model
        model = train_xgboost_model(X_train, y_train, X_val, y_val, strategy_name)

        # Test set performance
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n  Test Set Performance:")
        print(f"  AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
        print(f"  Accuracy: {(y_test_pred == y_test).mean():.4f}")
        print(f"  Predicted BUY: {y_test_pred.sum()} ({y_test_pred.sum()/len(y_test_pred)*100:.1f}%)")

        # Save model
        model_path = MODELS_DIR / f"xgb_{strategy_name}_15m.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_names": feature_names,
                "strategy": strategy_name
            }, f)

        print(f"\n  ✅ Model saved to: {model_path}")

        # Feature importance
        importance = model.feature_importances_
        top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  Top 10 Features:")
        for feat, imp in top_features:
            print(f"    {feat}: {imp:.4f}")

    print(f"\n{'='*70}")
    print(f"  ✅ Training Complete! Models saved in {MODELS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

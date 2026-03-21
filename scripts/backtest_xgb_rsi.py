#!/usr/bin/env python3
"""Test: Teammate's XGBoost + RSI Divergence overlay"""
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_features
from bot.execution.risk import RiskManager, RiskDecision
from bot.strategy.base import TradingSignal, SignalDirection
from bot.strategy.rsi_divergence import RSIDivergenceStrategy

print("Loading 15m BTC data...")
df = pd.read_parquet(project_root / "research_data" / "BTCUSDT_15m.parquet")
df = df.loc["2024-01-01":"2026-03-01"]
print(f"Data: {len(df)} bars")

print("Computing features...")
features = compute_features(df)

print("Loading XGBoost model...")
with open(project_root / "models" / "xgb_btc_15m.pkl", "rb") as f:
    xgb_model = pickle.load(f)
feature_names = list(xgb_model.feature_names_in_)

rsi_strat = RSIDivergenceStrategy()
risk_mgr = RiskManager()

print("Backtesting...\n")
cash, position, entry_price = 100_000.0, 0.0, None
equity_curve, trades, days_with_trades = [], [], set()
xgb_threshold, xgb_exit = 0.65, 0.12

for idx in features.index:
    curr = features.loc[:idx]
    latest, close = curr.iloc[-1], curr.iloc[-1]["close"]
    atr = latest.get("atr_proxy", close * 0.02)
    equity_curve.append(cash + position * close)
    
    if position > 0 and (stop := risk_mgr.check_stops("BTC/USD", close, atr)).should_exit:
        trades.append({"pnl": (close - entry_price) / entry_price, "src": "stop"})
        cash += position * close * 0.9998
        position, entry_price = 0.0, None
        risk_mgr.record_exit("BTC/USD")
        days_with_trades.add(idx.date())
        continue
    
    # Signals
    xgb_sig = rsi_sig = SignalDirection.HOLD
    X = curr.iloc[[-1]][[c for c in feature_names if c in curr.columns]]
    for c in feature_names:
        if c not in X.columns:
            X[c] = float("nan")
    X = X[feature_names]
    proba = float(xgb_model.predict_proba(X)[0, 1])
    
    if position == 0:
        if proba >= xgb_threshold:
            xgb_sig = SignalDirection.BUY
        if rsi_strat.generate_signal("BTC/USD", curr).direction == SignalDirection.BUY:
            rsi_sig = SignalDirection.BUY
    else:
        if proba < xgb_exit:
            xgb_sig = SignalDirection.SELL
        if rsi_strat.generate_signal("BTC/USD", curr).direction == SignalDirection.SELL:
            rsi_sig = SignalDirection.SELL
    
    # Execute
    if position == 0 and (xgb_sig == SignalDirection.BUY or rsi_sig == SignalDirection.BUY):
        src = "both" if xgb_sig == rsi_sig == SignalDirection.BUY else ("xgb" if xgb_sig == SignalDirection.BUY else "rsi")
        sizing = risk_mgr.size_new_position("BTC/USD", close, atr, cash, {}, 1.0, 0.7, 1.0)
        if sizing.decision == RiskDecision.APPROVED and sizing.approved_quantity * close * 1.0002 <= cash:
            position = sizing.approved_quantity
            cash -= position * close * 1.0002
            entry_price = close * 1.0002
            risk_mgr.record_entry("BTC/USD", entry_price, sizing.trailing_stop_price)
            days_with_trades.add(idx.date())
    elif position > 0 and (xgb_sig == SignalDirection.SELL or rsi_sig == SignalDirection.SELL):
        src = "both" if xgb_sig == rsi_sig == SignalDirection.SELL else ("xgb" if xgb_sig == SignalDirection.SELL else "rsi")
        trades.append({"pnl": (close - entry_price) / entry_price, "src": src})
        cash += position * close * 0.9998
        position, entry_price = 0.0, None
        risk_mgr.record_exit("BTC/USD")
        days_with_trades.add(idx.date())

if position > 0:
    cash += position * features.iloc[-1]["close"]

ret = cash / 100_000 - 1
eq = pd.Series(equity_curve)
rets = eq.pct_change().dropna()
sharpe = (rets.mean() / rets.std()) * np.sqrt(252*96) if rets.std() > 0 else 0
down = rets[rets < 0]
sortino = (rets.mean() / down.std()) * np.sqrt(252*96) if len(down) > 0 and down.std() > 0 else 0
dd = ((eq - eq.cummax()) / eq.cummax()).min()
calmar = ret / abs(dd) if dd != 0 else 0
comp = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
cov = len(days_with_trades) / (features.index[-1] - features.index[0]).days
wins = [t for t in trades if t["pnl"] > 0]

print("="*70)
print("  XGBoost + RSI Divergence Results")
print("="*70)
print(f"Return: {ret:.2%} | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f}")
print(f"Calmar: {calmar:.3f} | Composite: {comp:.3f}")
print(f"Max DD: {dd:.2%} | Win Rate: {len(wins)/len(trades):.1%}")
print(f"Trades: {len(trades)} | Coverage: {cov:.1%}")
print("="*70)
print(f"Sources: XGB={len([t for t in trades if t['src']=='xgb'])} "
      f"RSI={len([t for t in trades if t['src']=='rsi'])} "
      f"Both={len([t for t in trades if t['src']=='both'])} "
      f"Stop={len([t for t in trades if t['src']=='stop'])}")

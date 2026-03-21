#!/usr/bin/env python3
"""XGBoost + RSI Divergence overlay test"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.data.features import compute_features
from bot.strategy.rsi_divergence import RSIDivergenceStrategy
from bot.strategy.base import SignalDirection

# Load data
btc = pd.read_parquet(project_root / "research_data" / "BTCUSDT_15m.parquet")
btc.columns = btc.columns.str.lower()
feat = compute_features(btc).loc["2024-01-01":"2026-03-01"]

# Rename columns for RSI strategy compatibility
feat = feat.rename(columns={"RSI_14": "rsi", "EMA_20": "ema_20", "EMA_50": "ema_50"})

print(f"Data: {len(feat)} bars from {feat.index[0]} to {feat.index[-1]}")

# Load model
with open(project_root / "models" / "xgb_btc_15m.pkl", "rb") as f:
    model = pickle.load(f)

# Get probabilities
X = feat[[c for c in model.feature_names_in_ if c in feat.columns]]
for c in model.feature_names_in_:
    if c not in X.columns:
        X[c] = np.nan
probas = model.predict_proba(X[list(model.feature_names_in_)])[:, 1]

rsi_strat = RSIDivergenceStrategy()

# Params
threshold, exit_threshold, atr_mult, fee_rate = 0.65, 0.15, 10.0, 0.001
closes, atrs = feat["close"].values, feat["atr_proxy"].values
cash, position, entry_price, trail_stop, entry_source = 100_000.0, 0.0, 0.0, 0.0, None
portfolio_values, trades, days_with_trades = np.zeros(len(closes)), [], set()
portfolio_values[0] = cash

print("Backtesting XGBoost + RSI Divergence...\n")

for i in range(len(closes)):
    c, atr, p, ts = closes[i], atrs[i], probas[i], feat.index[i]
    portfolio_values[i] = cash + position * c
    
    if position > 0 and not np.isnan(atr) and atr > 0:
        trail_stop = max(trail_stop, c - atr_mult * atr)
    
    if position > 0:
        stop_hit = c <= trail_stop
        rsi_sig = rsi_strat.generate_signal("BTC/USD", feat.iloc[:i+1])
        sig_exit = (p <= exit_threshold if entry_source == "xgb" else 
                    rsi_sig.direction == SignalDirection.SELL)
        
        if stop_hit or sig_exit:
            trades.append({"pnl": (c - entry_price) / entry_price, "source": entry_source,
                          "exit": "stop" if stop_hit else "signal"})
            cash += position * c * (1 - fee_rate)
            position, entry_price, trail_stop, entry_source = 0.0, 0.0, 0.0, None
            days_with_trades.add(ts.date())
    
    if position == 0:
        xgb_buy = p >= threshold
        rsi_sig = rsi_strat.generate_signal("BTC/USD", feat.iloc[:i+1])
        rsi_buy = rsi_sig.direction == SignalDirection.BUY
        
        if xgb_buy or rsi_buy:
            units = (portfolio_values[i] * 0.25) / c
            cost = units * c * (1 + fee_rate)
            if cost <= cash:
                position, cash = units, cash - cost
                entry_price = c * (1 + fee_rate)
                trail_stop = c - atr_mult * atr if not np.isnan(atr) else c * 0.95
                entry_source = "both" if (xgb_buy and rsi_buy) else ("xgb" if xgb_buy else "rsi")
                days_with_trades.add(ts.date())

if position > 0:
    cash += position * closes[-1]

ret = cash / 100_000 - 1
eq = pd.Series(portfolio_values)
rets = eq.pct_change().dropna()
sharpe = (rets.mean() / rets.std()) * np.sqrt(35040) if rets.std() > 0 else 0
down = rets[rets < 0]
sortino = (rets.mean() / down.std()) * np.sqrt(35040) if len(down) > 0 and down.std() > 0 else 0
dd = ((eq - eq.cummax()) / eq.cummax()).min()
comp = 0.4 * sortino + 0.3 * sharpe + 0.3 * (ret / abs(dd) if dd != 0 else 0)
cov = len(days_with_trades) / (feat.index[-1] - feat.index[0]).days
wins = [t for t in trades if t["pnl"] > 0]

print("="*70)
print("  XGBoost + RSI Divergence Results")
print("="*70)
print(f"Return:     {ret:>8.2%}    |  Sharpe:   {sharpe:>8.3f}")
print(f"Sortino:    {sortino:>8.3f}    |  Composite:{comp:>8.3f}")
print(f"Max DD:     {dd:>8.2%}    |  Calmar:   {ret/abs(dd) if dd!=0 else 0:>8.3f}")
print(f"Trades:     {len(trades):>8}    |  Win Rate: {len(wins)/len(trades) if trades else 0:>8.1%}")
print(f"Coverage:   {cov:>8.1%}")
print("="*70)
print(f"\nSignal Breakdown:")
print(f"  XGB only:  {len([t for t in trades if t['source']=='xgb'])}")
print(f"  RSI only:  {len([t for t in trades if t['source']=='rsi'])}")
print(f"  Both:      {len([t for t in trades if t['source']=='both'])}")

# Backtest Results — Strategy Overlay Comparison

**Date:** 2026-03-21
**Model:** `models/xgb_btc_15m.pkl` (17 features, 15M BTC)
**Config:** threshold=0.70, ATR-mult=10, risk=2%/trade, fee=10bps, initial capital=$10,000
**Period:** 2024-01-01 → 2026-03-20 (77,665 bars)

---

## Summary Table

| Strategy | Return | CAGR | Sharpe | Sortino | Max DD | Trades | Win% | Avg PnL |
|---|---|---|---|---|---|---|---|---|
| XGBoost-only (baseline) | **+16.33%** | +7.06% | **1.141** | 1.649 | -6.00% | 73 | 60.3% | +0.61% |
| +MR overlay | **+17.45%** | +7.53% | **1.192** | 1.723 | -5.74% | 88 | 62.5% | +0.55% |
| +Pairs only | **-14.34%** | -6.74% | **-1.260** | -1.776 | -16.88% | 711 | 25.0% | -0.12% |
| +MR + Pairs | **-12.53%** | -5.86% | **-0.939** | -1.338 | -15.12% | 714 | 25.5% | -0.12% |
| +All (mr+pairs+unlock) | **-12.53%** | -5.86% | **-0.939** | -1.338 | -15.12% | 714 | 25.5% | -0.12% |

---

## Findings

### MR Overlay: Marginal improvement — keep as optional
- +1.12% more return, Sharpe 1.141 → 1.192, max DD slightly better (-5.74% vs -6.00%)
- Adds 15 trades (88 vs 73), win rate improves 60.3% → 62.5%
- MR acts as a fallback when XGBoost confidence is below threshold — it catches oversold
  setups that XGBoost misses. Modest but consistent benefit.
- **Verdict: safe to enable (`--strategies mr`) for the competition.**

### Pairs Strategy: BROKEN on 15M timeframe — do not use
- 711 trades vs 73 baseline (10x overtrade), -14.34% return, 25% win rate
- Root cause: `PairsTradingStrategy` was calibrated for **4H data**:
  - `MIN_BARS=100` → at 15M = only 25 hours of history for cointegration test
  - `RETEST_INTERVAL=42` → reruns Engle-Granger every 10.5 hours
  - `pairs_lookback=500` → 500 × 15min = 5.2 days of data
  - At these window sizes, spurious cointegration fires constantly (p<0.10 threshold)
  - Spread reverts quickly at 5-day windows → high-frequency entry/exit churn
- **Verdict: disable for competition. Only valid on 4H backtest.**

### Unlock Screen: No-op (as expected)
- `UNLOCK_EXCLUSIONS` is empty — unlock screen has zero effect until manually populated.
- `--strategies all` == `--strategies mr,pairs` in current results.
- **Action required before competition:** Check tokenomist.ai, update `bot/config/unlock_screen.py`.

---

## Recommendation for Competition

**Use `--strategies mr`** (MeanReversion overlay only).

```bash
python scripts/backtest_15m.py \
  --model models/xgb_btc_15m.pkl \
  --threshold 0.70 \
  --atr-mult 10 \
  --strategies mr
```

Do NOT enable `pairs` at 15M. If pairs trading is desired, it needs a separate 4H backtest
with recalibrated parameters (larger `pairs_lookback`, larger `MIN_BARS`, longer `RETEST_INTERVAL`).

---

## Stop Exits vs Signal Exits

| Strategy | Stop exits | Signal exits | Ratio |
|---|---|---|---|
| XGBoost-only | 27 | 46 | 37% stopped out |
| +MR overlay | 29 | 59 | 33% stopped out |
| +Pairs only | 29 | 682 | 4% stopped out |
| +MR + Pairs | 31 | 683 | 4% stopped out |

The pairs configs are overwhelmingly signal-exit driven — the spread reverts within the ATR
stop window on nearly every trade, but in the wrong direction (25% win rate).

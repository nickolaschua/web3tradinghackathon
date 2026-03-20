# Issue 29: Bot Effectively Single-Asset for First ~6 Days After Startup

## Layer
Layer 3 — Data Pipeline / Layer 8 — Orchestration

## Description
On first startup, `LiveFetcher._buffers` is seeded with historical data only for
`feature_pairs` (BTC/USD, ETH/USD, SOL/USD). The other 36 `tradeable_pairs` start
with zero bars in `_buffers`.

In `main.py`, the cycle skips pairs with insufficient warmup:
```python
if len(live_fetcher._buffers.get(pair, [])) < config.get("warmup_bars", 35):
    continue
```

At 4H candle frequency:
- 35 bars × 4 hours = 140 hours = ~5.8 days to accumulate warmup bars
- During this window, only BTC/USD, ETH/USD, SOL/USD can trade

Given Issue 20 (only BTC/USD is tradeable on Roostoo), this is currently moot. But if
the bot is deployed against an exchange with a full universe, the warmup gap means the
bot has an artificial single-asset window where the portfolio allocation logic doesn't
yet have enough history to be meaningful.

## Impact
**Low** (given current single-pair constraint) to **Medium** (if multi-exchange deployment).

## Fix Required
Document the warmup behavior explicitly. If deploying to multi-asset exchange:
1. Pre-seed all tradeable pairs from Binance historical data (same as feature_pairs seeding).
2. Reduce `warmup_bars` to 15 (minimum for ATR proxy, RSI not yet valid but better than
   nothing) for symbols where full history is unavailable.
3. Or: require all symbols to be pre-seeded before bot starts trading.

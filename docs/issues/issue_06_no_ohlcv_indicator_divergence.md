# Issue 06: No OHLCV Endpoint — Indicators Permanently Diverged from Backtest

## Layer
Layer 2 — Data Pipeline / Layer 3 — Feature Engineering (architectural)

## Description
The Roostoo API has no OHLCV/candlestick endpoint. The only price data available is `LastPrice` from `/v3/mock-ticker`. All live synthetic candles are constructed as H=L=O=C=LastPrice with volume=0.

This means every indicator that uses high, low, or volume will compute different values in live trading than in backtesting:

- **ATR**: Uses high-low range. With H=L=LastPrice, ATR will be near-zero in live data, causing ATR trailing stops to be set extremely tight (or collapse to the hard stop).
- **ADX**: Uses true range (requires high/low). Will be meaningless on flat candles.
- **OBV (On-Balance Volume)**: Volume is always 0, so OBV will never move.
- **Bollinger Bands**: Width based on close-to-close changes only — will work, but narrower than on true OHLCV data.

## Impact
**Critical** — the entire stop-loss system is ATR-based. If live ATR is near-zero, initial stop prices will be nearly at the current price, causing immediate stop-outs on every position. The ATR multiplier and lookback period calibrated against historical OHLCV data are meaningless on synthetic flat-candle data.

## Fix Required
1. **ATR**: Use a rolling close-to-close standard deviation as a proxy ATR for the live system, calibrated so it produces stop distances similar to historical ATR. Or source OHLCV from Binance WebSocket in parallel (outside Roostoo API).
2. **ADX**: Disable ADX as a signal filter in live trading, or use momentum-only signals that don't require ATR/ADX.
3. **OBV**: Disable OBV features entirely for live trading.
4. **Document divergence**: Add explicit config flags for `live_mode=true` to switch to close-only indicators.

# Issue 20: Multi-Asset Architecture Documented But Roostoo Only Supports BTC/USD

## Layer
Layer 0 — External Dependencies / Architecture

## Description
The documentation describes downloading data for BTCUSDT, ETHUSDT, SOLUSDT, and BNBUSDT from Binance, and the feature engineering layer has cross-asset features (`btc_return_lag1`, `eth_return_lag1`, etc.) designed for multi-pair trading.

However, the Roostoo API reference (`docs/13_roostoo_api_reference.md`) confirms the mock exchange only supports `BTC/USD`. There is no `ETH/USD`, `SOL/USD`, or `BNB/USD` trading pair available.

This means:
1. All ETH/SOL/BNB data downloaded from Binance can only be used as feature inputs (cross-asset signals), not as tradeable pairs
2. The `pairs` config list should contain only `["BTC/USD"]`
3. Position sizing for multiple concurrent positions in different coins is moot — BTC is the only tradeable asset
4. `max_positions = 3` is irrelevant unless the intent is to enter/exit BTC multiple times concurrently (not meaningful)

## Impact
**Medium** — wastes complexity. Multi-asset architecture adds code paths that will never execute. Simplify: BTC/USD only, use ETH/SOL/BNB only as feature inputs.

## Fix Required
Clarify in `config.yaml`: `tradeable_pairs: ["BTC/USD"]`, `feature_pairs: ["BTC/USD", "ETH/USD", "SOL/USD"]`.
Set `max_positions: 1` since only one pair is tradeable.

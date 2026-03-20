# Issue 27: No Order Confirmation Loop — Phantom Positions on Silent Fill Failures

## Layer
Layer 7 — Order Management / Layer 6 — Risk Management

## Description
After placing a buy order via the Roostoo API, `main.py` immediately calls:
```python
risk_manager.record_entry(pair, fill_price, sizing.trailing_stop_price)
```

This records the position in `RiskManager._entry_prices` and `_trailing_stops` without
confirming the order actually executed on the exchange side.

Failure modes:
1. **Partial fill**: Order for 0.1 BTC placed, only 0.05 BTC fills due to order book depth.
   Risk manager thinks it holds 0.1 BTC. Stop triggers a sell of 0.1 BTC — but only
   0.05 BTC is actually held. Roostoo returns an error or partial fill on the close order.

2. **Silent API failure**: Roostoo returns HTTP 200 but the order is not in the book
   (API bug, network partition after request is sent but before processing).
   Risk manager records a phantom entry. Stop eventually fires and a sell order fails
   because there's no position to close.

3. **Duplicate entry on restart**: If bot crashes after placing an order but before the
   `record_entry()` call completes, on restart the position exists on Roostoo but not
   in the state file. The bot may re-enter the same pair, doubling exposure.

## Impact
**High** — Any divergence between `RiskManager` state and actual Roostoo positions
causes incorrect stop/exit behavior. In a competition, a phantom position could prevent
new entries (max_positions check uses `_entry_prices` count).

## Fix Required
1. After placing a BUY order, poll `GET /api/v3/order` or equivalent to confirm fill status.
2. Only call `record_entry()` after confirming the order is filled.
3. Add a startup reconciliation step: compare `RiskManager._entry_prices` to actual
   open positions from Roostoo and resolve discrepancies.
   (See also Issue 16: startup_reconciliation_unimplemented)

# Issue 13: `fill_price` Fallback Chain Uses `quantity` as Price

## Layer
Layer 7 — Order Management (`execution/oms.py`)

## Description
In the OMS order-processing logic, the fill price is derived as:
```python
fill_price = order.filled_price or price or quantity
```

This fallback chain has a critical bug: if `filled_price` is `0.0` (falsy in Python), it falls through to `price`. If `price` is also `0.0`, it falls through to `quantity`. Using `quantity` (e.g. 0.000125 BTC) as a price (e.g. $0.000125 instead of $84,000) will silently record a catastrophically wrong entry price, which will:

1. Set a hard stop at `entry_price * 0.92 = $0.000115` — never triggered
2. Set ATR trailing stop near $0 — never triggered
3. Report absurd PnL in logs and Telegram alerts

## Code Location
`execution/oms.py` — order processing, fill_price assignment

## Fix Required
Replace the dangerous `or` chain with explicit None checks:
```python
if order.filled_price is not None and order.filled_price > 0:
    fill_price = order.filled_price
elif price is not None and price > 0:
    fill_price = price
else:
    logger.error(f"Cannot determine fill price for order {order.order_id}")
    raise ValueError("fill_price unknown")
```

## Impact
**Critical** — silent wrong entry price causes stop-loss system to be completely bypassed. Risk management becomes ineffective.

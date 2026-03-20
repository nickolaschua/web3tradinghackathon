# Gap 09: Minimum Order Size and Quantity Precision for Roostoo

## Why This Is a Gap
`ExchangeInfoCache` rounds quantities to exchange-specified precision, but I don't know what that precision is for the Roostoo mock exchange. An incorrect rounding could cause order rejections.

## What I Need to Know

1. **What is the minimum order quantity for BTC/USD on Roostoo?**
   - The API reference hints at "minimum trade size" but doesn't specify the value
   - Is it 0.001 BTC? 0.0001 BTC? Something else?

2. **What is the quantity precision (decimal places)?**
   - Standard: 4-6 decimal places for BTC
   - If the system calculates 0.1234567 BTC, what does it get rounded to?

3. **What is the price precision?**
   - For market orders, price isn't submitted. But for any future limit order implementation, knowing price precision matters.

4. **What happens when a calculated quantity rounds to zero?**
   - E.g. if minimum position size is $100 USD at $84,000 BTC price = 0.00000119 BTC
   - If minimum quantity is 0.001 BTC, this order is rejected
   - The current `size_new_position()` checks for `usable_balance < $100` but doesn't verify minimum BTC quantity

5. **What is the exact error response when quantity is below minimum?**
   - Is it `{"Success": false, "ErrMsg": "quantity too small"}` or something else?
   - Needed to handle the error gracefully rather than retrying indefinitely

## Where to Find This
- Roostoo API documentation / ExchangeInfo endpoint (`/v3/exchange_info` if it exists)
- Empirical testing: submit a very small order and observe the error

## Priority
**Medium** — affects order submission reliability. System will work at typical position sizes but may fail at edge cases.

---

## Research Findings (API Docs + Inference, 2026-03-12)

### Context7 Limitation

Roostoo-specific exchange info is not in Context7. Findings below are based on
the Roostoo API reference in `docs/13_roostoo_api_reference.md` and standard
mock exchange conventions.

### Inferred Values from API Documentation

From `docs/13_roostoo_api_reference.md`:
- The API accepts `Quantity` as a float parameter
- No `/v3/exchange_info` endpoint is documented for Roostoo
- The `ExchangeInfoCache` in the codebase round quantities but with no
  hardcoded precision — it appears to rely on an API response that may not exist

### Standard BTC Mock Exchange Conventions

Most BTC/USD mock exchanges use:
- **Minimum quantity**: 0.001 BTC (≈$84 at $84,000/BTC)
- **Quantity precision**: 4 decimal places (0.0001 BTC granularity)
- **Price precision**: 2 decimal places ($0.01 USD)

These are the Binance testnet defaults and are widely used as mock exchange
defaults. The Roostoo mock exchange likely follows the same conventions.

### Defensive Implementation

Until confirmed empirically, use these conservative defaults:

```python
# In config.yaml or ExchangeInfoCache defaults:
MIN_QUANTITY_BTC = 0.001          # 0.001 BTC minimum order
QUANTITY_PRECISION = 4            # Round to 4 decimal places
MIN_POSITION_USD = 100.0          # Minimum position in USD

# In size_new_position():
def size_new_position(usable_balance: float, price: float, ...) -> float:
    if usable_balance < MIN_POSITION_USD:
        return 0.0

    raw_qty = (usable_balance * position_pct) / price
    qty = round(raw_qty, QUANTITY_PRECISION)

    # NEW: verify minimum BTC quantity
    if qty < MIN_QUANTITY_BTC:
        logger.warning(
            f"Calculated quantity {qty} BTC below minimum {MIN_QUANTITY_BTC} BTC "
            f"(${qty * price:.2f} USD). Skipping order."
        )
        return 0.0

    return qty
```

### Error Response Handling

If an order is rejected for quantity reasons, handle gracefully:
```python
def submit_order(symbol, side, quantity):
    resp = client.new_order(symbol, side, quantity)
    if not resp.get("Success"):
        err = resp.get("ErrMsg", "")
        if "quantity" in err.lower() or "size" in err.lower():
            logger.error(f"Order rejected: quantity issue — {err}")
            # Do NOT retry — the quantity will be the same
            return None
        elif "balance" in err.lower():
            logger.error(f"Order rejected: insufficient balance — {err}")
            return None
        else:
            logger.error(f"Order rejected: {err}")
            return None
    return resp
```

### Empirical Test Plan

Before competition starts, submit a test order:
```python
# Submit minimum viable order to discover actual minimum
test_orders = [0.001, 0.0005, 0.0001]
for qty in test_orders:
    resp = client.new_order("BTC/USD", "BUY", qty)
    print(f"qty={qty}: Success={resp['Success']}, ErrMsg={resp.get('ErrMsg')}")
```

Record the results and update `MIN_QUANTITY_BTC` in `config.yaml`.

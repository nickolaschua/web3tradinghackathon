# Gap 04: Roostoo Market Order Fill Behavior and Price Slippage

## Why This Is a Gap
The API reference says market orders fill "at current market price" and the fill is synchronous (the response includes fill price). But several behaviors need verification:

## What I Need to Know

1. **Does the fill price exactly equal `LastPrice` at time of submission?**
   - Or does Roostoo simulate slippage?
   - If there is simulated slippage, how large? Fixed bps? Volume-weighted?

2. **What happens if `LastPrice` changes between request construction and API receipt?**
   - The network round-trip is ~50-200ms. If BTC moves $100 during that time, does the fill reflect the new price?

3. **What is the `FilledPrice` field in the order response?**
   - Is it always populated, or only for fully-filled orders?
   - Can an order be partially filled?

4. **What does `OrderStatus` look like in the response?**
   - Valid values: "PENDING", "FILLED", "CANCELLED", "REJECTED"?
   - Can a market order be rejected without filling?

5. **Commission deduction mechanism:**
   - Is commission deducted from the received asset (buy BTC → get slightly less BTC) or from USD balance?
   - Does the `Free` balance in `/v3/balance` reflect post-commission values?

## Why This Matters
- If fill price ≠ LastPrice, the OMS's entry price recording will be wrong
- If market orders can be partially filled, the OMS needs partial-fill handling
- Commission deduction affects position size calculations

## Where to Find This
- `docs/13_roostoo_api_reference.md` has partial information
- Test against the live Roostoo endpoint with a minimum-size order

## Priority
**High** — affects OMS correctness and stop-loss price accuracy.

---

## Research Findings (API Docs + Domain Knowledge, 2026-03-12)

### Context7 Limitation

Context7 does not contain Roostoo-specific fill behavior documentation.
Findings below are based on `docs/13_roostoo_api_reference.md` and standard
mock-exchange conventions.

### From Roostoo API Reference (`docs/13_roostoo_api_reference.md`)

The `/v3/new-order` endpoint returns synchronously with:
```json
{
  "Success": true,
  "ErrMsg": "",
  "FilledPrice": 84123.50,
  "FilledQuantity": 0.1,
  "OrderId": "abc123"
}
```

Key observations:
- `FilledPrice` IS populated in the synchronous response
- Fill appears to happen at the current `LastPrice` at the time of server receipt
- No evidence of simulated slippage in the API docs

### Inferred Behavior (Mock Exchange Standard)

Roostoo is a **mock exchange** designed for competition use. Mock exchanges
typically:
1. Fill market orders at `LastPrice` with **zero slippage**
2. Execute orders synchronously (no PENDING state for market orders)
3. Do not partially fill orders (quantity is either fully filled or rejected)
4. Deduct commission from USD balance, not from BTC quantity

### Commission Rate

From API docs: 0.1% per trade (0.001 fee rate). This means:
- BUY 0.1 BTC at $84,000: commission = $84,000 × 0.1 × 0.001 = $8.40
- Net USD balance decreases by: $8,400 (principal) + $8.40 (fee) = $8,408.40
- BTC received: exactly 0.1 BTC

### Order Rejection Conditions

Market orders can be rejected if:
- Insufficient USD balance for BUY
- Insufficient BTC balance for SELL
- Quantity below minimum threshold (see Gap 09)
- Quantity precision exceeds maximum decimal places

### Defensive Coding Recommendations

Given the synchronous fill model, the OMS should:

```python
def parse_order_response(resp: dict, fallback_price: float) -> OrderResult:
    if not resp.get("Success"):
        raise OrderRejected(resp.get("ErrMsg", "unknown"))

    # FilledPrice should always be present for successful market orders
    fill_price = resp.get("FilledPrice")
    if fill_price is None or fill_price <= 0:
        # Defensive fallback — should not happen on Roostoo mock exchange
        fill_price = fallback_price
        logger.warning("FilledPrice missing from response, using LastPrice fallback")

    return OrderResult(
        fill_price=fill_price,
        quantity=resp["FilledQuantity"],
        order_id=resp["OrderId"]
    )
```

Do NOT use `resp.get("FilledPrice") or fallback_price` — this triggers when
`FilledPrice = 0.0` (which would be a legitimate rejection price, not missing).
Use explicit `is None` check. This fixes Issue 13.

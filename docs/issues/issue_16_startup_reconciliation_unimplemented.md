# Issue 16: Startup Reconciliation Against Exchange Not Implemented

## Layer
Layer 7 — Order Management (`execution/oms.py`)
Layer 8 — State Persistence (`persistence/state.py`)

## Description
The `startup()` function in Layer 8 calls `client.get_balance()` and `client.get_open_orders()` to reconcile live state against persisted state. The docs describe this reconciliation in detail, but the actual reconciliation logic is marked with a comment `# (Full reconciliation implementation in Layer 7)` — and Layer 7 does not contain this implementation.

Specifically, the following reconciliation steps are missing:
1. Compare `_positions` quantities against live balance holdings
2. Compare persisted pending order IDs against live `/v3/pending_orders`
3. Detect and resolve discrepancies (position exists in state but not on exchange, or vice versa)
4. Log all discrepancies at WARNING level and send Telegram alert

Without this, the bot starts trading with potentially incorrect position state after any restart.

## Code Location
`persistence/state.py` → `startup()` function — reconciliation block is a placeholder comment
`execution/oms.py` → no reconciliation method defined

## Fix Required
Implement `oms.reconcile_with_exchange(live_balance, live_pending_orders)` that:
1. Checks each known position against the live balance
2. For discrepancies: trusts the exchange (live > persisted state)
3. Sends Telegram WARN for any discrepancy found

## Impact
**High** — after a crash mid-trade, the bot may not know whether an order was filled. Without reconciliation, it may try to open a duplicate position or ignore a position it actually holds.

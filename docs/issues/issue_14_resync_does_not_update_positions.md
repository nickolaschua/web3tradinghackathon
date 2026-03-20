# Issue 14: `_resync_from_exchange()` Does Not Update `_positions` Dict

## Layer
Layer 7 — Order Management (`execution/oms.py`)

## Description
`_resync_from_exchange()` is called to reconcile the OMS's in-memory position state against the live Roostoo balance. The method fetches the live balance but does not write the results back into `self._positions`. It only reads and logs, leaving `_positions` unchanged.

This means if the OMS has drifted from reality (e.g. due to a crash, manual intervention, or missed fill confirmation), `_resync_from_exchange()` will detect the discrepancy and log it, but the in-memory state will remain wrong for the rest of the session.

## Code Location
`execution/oms.py` → `_resync_from_exchange()` method

## Fix Required
After fetching live balances and detecting a discrepancy, update `_positions` to reflect the live state:
```python
# After detecting discrepancy:
self._positions[pair] = Position(
    pair=pair,
    quantity=live_quantity,
    entry_price=self._positions[pair].entry_price,  # keep known entry
    ...
)
```

## Impact
**High** — after a crash and restart, the position state may be wrong throughout the trading session. Stop-loss checks and sizing calculations will use stale position data.

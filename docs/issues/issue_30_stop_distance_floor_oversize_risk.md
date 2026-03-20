# Issue 30: Stop Distance Floor Guard Can Produce Larger Positions Than Intended

## Layer
Layer 6 — Risk Management

## Description
When `atr_stop > price` (e.g., ATR is larger than the current price — pathological but
possible with corrupt data), `initial_stop = max(hard_stop, atr_stop)` becomes negative.
The code detects this:

```python
stop_distance = current_price - initial_stop
if stop_distance <= 0:
    stop_distance = current_price * hard_stop_pct  # fallback to 5% of price
```

This fallback sets `stop_distance = current_price × 0.05`.

The position is then sized as:
```python
quantity = risk_usd / stop_distance
target_usd = quantity * current_price
```

With `stop_distance = price × 0.05`:
- `quantity = risk_usd / (price × 0.05)`
- `target_usd = quantity × price = risk_usd / 0.05 = 20 × risk_usd`

For `risk_usd = portfolio × 0.02 × 1.0 × 0.7 × 1.0 = $140` (on $10K portfolio):
- `target_usd = 20 × $140 = $2,800`

This is then correctly **capped** by:
```python
target_usd = min(target_usd, total_portfolio * max_single_pct, usable_balance)
# = min($2800, $4000, $9500) = $2800
```

The cap saves us in this case. However, if `risk_per_trade_pct` is larger or the formula
changes in the future, the floor guard can produce uncapped oversize before the cap applies.

## Impact
**Low** — Concentration cap (`max_single_position_pct = 0.40`) prevents the worst case.
The floor guard behavior is tested and passes. This is a documentation/awareness issue
rather than an active bug.

## Fix Required
Add a comment in `size_new_position()` explaining why the floor guard works safely in
conjunction with the concentration cap. Consider logging a warning when the floor guard
triggers, as it indicates corrupt/unexpected ATR data that should be investigated.

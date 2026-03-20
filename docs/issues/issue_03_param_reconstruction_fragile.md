# Issue 03: Param Dict Reconstruction from `total_params` String is Fragile

## Layer
Layer 1 — API Client (`api/client.py`)

## Description
`get_balance()` and `get_pending_count()` reconstruct a params dictionary from the `total_params` query string using `split("=", 1)`. This is fragile because URL-encoded values that contain `=` characters (e.g. base64-encoded values) will be incorrectly split, and the reconstruction assumes the query string format is exactly `key=value&key=value`.

The root cause is that the signing method returns only the signed string rather than returning both the string and the original dict, forcing callers to reconstruct what they started with.

## Code Location
`api/client.py` → `get_balance()`, `get_pending_count()`, `_sign()` method signature

## Fix Required
Refactor `_sign()` to accept a dict and return both the signed string and the original dict (or just sign and send without reconstruction):
```python
# Instead of reconstructing, just pass params directly to _request
def get_balance(self):
    params = {"timestamp": self._get_timestamp()}
    return self._request("GET", "/v3/balance", params)
```

## Impact
**Low** — currently only affects endpoints with simple params (timestamp only), so the bug doesn't manifest. Will break if any endpoint is added with complex parameter values.

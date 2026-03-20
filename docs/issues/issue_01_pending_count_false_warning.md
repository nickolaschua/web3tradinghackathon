# Issue 01: `pending_count` Logs False WARNING on Zero Orders

## Layer
Layer 1 — API Client (`api/client.py`)

## Description
`get_pending_count()` calls the Roostoo `/v3/pending_count` endpoint. The API returns `{"Success": false, "ErrMsg": "no pending order"}` when there are zero pending orders — this is a normal operating condition, not an error.

The current `_request()` method logs `Success=false` responses at WARNING level. This means every polling cycle with no open orders will fire a spurious WARNING log entry and potentially a Telegram WARN alert, creating alert fatigue that will cause real warnings to be ignored.

## Code Location
`api/client.py` → `_request()` method

## Reproduction
Any polling cycle where no pending orders exist will trigger: `WARNING: API returned Success=false: no pending order`

## Fix Required
Special-case `pending_count` to treat `Success=false` with `ErrMsg="no pending order"` as a valid empty result (return `{"Success": true, "Data": {"pending_count": 0}}`), OR suppress the WARNING log for known non-error `Success=false` responses.

## Impact
**High** — fires on every poll cycle, creates alert fatigue, may cause real errors to be dismissed.

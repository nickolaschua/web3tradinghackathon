# Issue 15: Multiple Methods Called Across Layers That Are Not Defined

## Layer
Multiple layers — cross-cutting interface mismatch

## Description
Several methods are called by one layer but not defined in the layer that is supposed to implement them. These will all raise `AttributeError` at runtime:

| Caller | Method Called | Should Be Defined In | Defined? |
|--------|--------------|----------------------|----------|
| `monitoring/healthcheck.py` | `oms.get_all_positions()` | `execution/oms.py` | Not documented |
| `persistence/state.py` | `oms.dump_state()` | `execution/oms.py` | Not documented |
| `persistence/state.py` | `oms.load_state()` | `execution/oms.py` | Not documented |
| `persistence/state.py` | `risk_manager.dump_state()` | `execution/risk.py` | Not defined (Issue 12) |
| `persistence/state.py` | `risk_manager.load_state()` | `execution/risk.py` | Not defined (Issue 12) |
| `persistence/state.py` (main loop) | `fetcher.get_candle_boundaries()` | `data/live_fetcher.py` | Not documented |
| `main.py` (Layer 10) | `fetcher.get_latest_price()` | `data/live_fetcher.py` | Not documented |

## Impact
**Critical** — each of these will cause an `AttributeError` crash the first time the relevant code path is executed. The state persistence system and monitoring will fail completely.

## Fix Required
All missing methods must be implemented. Define a complete interface contract for each class before writing `main.py`:
- `OMS`: `get_all_positions()`, `dump_state()`, `load_state()`
- `RiskManager`: `dump_state()`, `load_state()` (see Issue 12)
- `LiveFetcher`: `get_candle_boundaries()`, `get_latest_price()`

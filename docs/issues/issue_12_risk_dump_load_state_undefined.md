# Issue 12: `RiskManager.dump_state()` and `load_state()` Not Defined

## Layer
Layer 6 — Risk Management (`execution/risk.py`)
Layer 8 — State Persistence (`persistence/state.py`)

## Description
The `startup()` function in Layer 8 calls `risk_manager.load_state(state.get("risk", {}))` and the main loop integration calls `risk_manager.dump_state()` to save state. Neither of these methods is defined in the `RiskManager` class in Layer 6.

`RiskManager` has `record_entry()`, `record_exit()`, `initialize_hwm()`, `check_stops()`, `check_circuit_breaker()`, and `size_new_position()` — but no serialisation methods.

## Code Location
`execution/risk.py` → `RiskManager` class (missing methods)
`persistence/state.py` → `startup()` (calls undefined methods)

## What They Need to Do
```python
def dump_state(self) -> dict:
    return {
        "trailing_stops": self._trailing_stops,
        "entry_prices": self._entry_prices,
        "portfolio_hwm": self._portfolio_hwm,
        "circuit_breaker_active": self._circuit_breaker_active,
    }

def load_state(self, state: dict):
    self._trailing_stops = state.get("trailing_stops", {})
    self._entry_prices = state.get("entry_prices", {})
    self._portfolio_hwm = state.get("portfolio_hwm", 0.0)
    self._circuit_breaker_active = state.get("circuit_breaker_active", False)
```

## Impact
**Critical** — on restart, `load_state()` call will raise `AttributeError`, crashing startup. Stop levels and circuit breaker state will not survive restarts.

# Issue 10: `TradingSignal.pair` Always Empty String

## Layer
Layer 5 — Strategy Engine (`strategy/base.py`, `strategy/momentum.py`, `strategy/mean_reversion.py`)

## Description
`TradingSignal` is defined with `pair: str = ""`. Both `MomentumStrategy.generate_signal()` and `MeanReversionStrategy.generate_signal()` return `TradingSignal` instances without setting the `pair` field. The base class has no enforcement.

When the orchestration layer (Layer 10) receives a LONG signal and tries to route it to the OMS to place an order, `signal.pair` will be `""`, causing the OMS to attempt to place an order for a blank pair — which will either fail at exchange validation or place against the wrong instrument.

## Code Location
`strategy/base.py` → `TradingSignal` dataclass
`strategy/momentum.py` → return statements
`strategy/mean_reversion.py` → return statements

## Fix Required
Either:
1. Add `pair` as a required positional argument to `TradingSignal` (remove default value so `TradingSignal()` without pair fails at construction), or
2. Have the strategy `generate_signal(pair, features)` method automatically set `self.pair` on the returned signal.

## Impact
**Critical** — every trade signal will have `pair=""`, causing all order submissions to fail.

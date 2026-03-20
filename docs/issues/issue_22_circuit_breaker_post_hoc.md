# Issue 22: Circuit Breaker Is Post-Hoc — Fires After 30% Portfolio Loss

## Layer
Layer 6 — Risk Management

## Description
The circuit breaker activates at **30% drawdown from the high-water mark**. This is a reactive
guard, not a predictive one. In a fast crypto crash, the full 30% can be lost before the CB
ever fires. The tiered multipliers (0.5x at 10%, 0.25x at 20%) reduce *new* position sizes
during the drawdown, but they do not reduce the size of *existing* positions.

Concrete worst case:
- Portfolio at HWM = $10,000
- Bot opens a full-size position at 0% drawdown (1.0x)
- Price crashes 30% in 2H — hard stop fires at -5%, position exits at $9,500
- Another entry fires on a bounce signal at -5% drawdown → still 1.0x size
- That position also stops out — now at $9,025
- Repeat pattern could bring portfolio to CB threshold over several small-loss trades

The CB at 30% halts trading but cannot recover capital already lost.

## Impact
**Medium** — The tiered multipliers do progressively reduce risk as drawdown deepens, which
is correct behavior. The concern is that the 30% full-halt threshold is quite deep for a
competition where total capital is fixed.

## Fix Required
Consider tightening the halt threshold to 15-20% for competition use.
Document in `config.yaml` that `circuit_breaker.halt_threshold` should be calibrated
to competition bankroll, not live-trading standards.

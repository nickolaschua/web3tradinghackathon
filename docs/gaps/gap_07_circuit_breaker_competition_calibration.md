# Gap 07: Circuit Breaker Threshold Calibration for Competition Context

## Why This Is a Gap
The circuit breaker fires at 30% drawdown from the high-water mark and only resets when the portfolio fully recovers to the previous HWM. In a competition context, this design may be too conservative or too aggressive depending on the competition duration and scoring.

## What I Need to Know

1. **Is "requires full HWM recovery" appropriate for a competition?**
   - In a 2-week competition, if the bot loses 30% and the market continues rallying, waiting for full recovery before resuming means missing the recovery itself
   - Should the circuit breaker have a partial recovery threshold (e.g. 50% recovery from the drawdown)?

2. **Is 30% the right trigger threshold?**
   - For $50k capital: 30% = $15k loss trigger → portfolio at $35k
   - With max position size ~17% ($8.5k), a single 30% ATR stop exit would lose at most 2.5% of portfolio ($1.25k) — far from triggering the circuit breaker
   - The 30% threshold only triggers after multiple consecutive maximum losses — is this realistic?
   - Alternative: 20% threshold (more conservative, fires sooner)

3. **Should the circuit breaker be "hard" (no trades) or "soft" (reduced size only)?**
   - Hard (current design): no new positions until recovery
   - Soft: reduce regime multiplier to 0.25 while in drawdown mode — still trades but very small
   - Which performs better in competition scenarios where a bear regime causes a 30% drawdown?

4. **Does the competition require the bot to be actively trading?**
   - If minimum trade frequency is required, a hard circuit breaker that lasts days could cause disqualification

## Priority
**Medium** — affects worst-case loss scenarios but normal operation continues as designed.

---

## Research Findings (Domain Knowledge + Competition Design, 2026-03-12)

### Circuit Breaker Math

With $50,000 initial capital and 17% max position size:
- Max single position: $8,500
- Position entry uses ATR-based stop; typical ATR stop = 2x ATR below entry
- For BTC at $84,000 with close-to-close ATR ≈ $1,500 (4H):
  - Stop distance = 2 × $1,500 = $3,000 per BTC
  - Position size = $8,500 / $84,000 ≈ 0.101 BTC
  - Max loss per trade = 0.101 × $3,000 = $303 ≈ 0.6% of portfolio

**To reach 30% drawdown ($15,000 loss) from max-size consecutive losses**:
$15,000 / $303 ≈ 50 consecutive maximum-loss trades — essentially impossible
in normal operation.

**More realistic trigger path**: A large adverse BTC move ($5,000+) in a single
4H candle before the stop is checked (stop checked only every 60 seconds, not
continuously). A 10% BTC crash in one candle on a $8,500 position = $850 loss
(1.7% of portfolio). Still needs ~18 such events to trigger 30% CB.

**Conclusion**: 30% CB threshold is appropriate as a catastrophic-failure
safeguard, not a regular operational tool. It should remain at 30%.

### Hard vs Soft Circuit Breaker

**For a competition**: soft circuit breaker is better because:
1. Avoids missing recovery rallies after a drawdown
2. Prevents potential disqualification from trade-frequency requirements
3. Reduces size proportionally rather than going flat

Recommended implementation:
```python
def get_circuit_breaker_multiplier(current_equity: float, hwm: float) -> float:
    drawdown = (hwm - current_equity) / hwm
    if drawdown > 0.30:
        return 0.0    # Hard stop at 30% — complete halt
    elif drawdown > 0.20:
        return 0.25   # Soft: trade at 25% normal size
    elif drawdown > 0.10:
        return 0.50   # Soft: trade at 50% normal size
    else:
        return 1.0    # Full size
```

This tiered approach avoids the cliff-edge of the current binary CB design.

### HWM Recovery Design

**Replace** "requires full HWM recovery" with "75% recovery" rule:
```python
# Current (problematic): only reset CB when equity >= hwm
# New: reset CB when equity recovers 75% of the drawdown
def should_reset_circuit_breaker(
    current_equity: float,
    cb_trigger_equity: float,  # equity at which CB fired
    hwm: float
) -> bool:
    recovery_threshold = cb_trigger_equity + 0.75 * (hwm - cb_trigger_equity)
    return current_equity >= recovery_threshold
```

This means: if HWM was $60k and CB fired at $42k (30% drawdown), the CB resets
when equity reaches $42k + 0.75×($60k-$42k) = $42k + $13.5k = $55.5k.
The bot resumes trading before full recovery to the $60k HWM.

### Competition Duration Sensitivity

| Duration | CB threshold | HWM recovery rule |
|----------|-------------|-------------------|
| < 1 week | 20% | 50% recovery |
| 1-2 weeks | 25% | 75% recovery |
| > 2 weeks | 30% | 90% recovery |

Shorter competitions should use more aggressive CB because missing even 2 days
of recovery trading is a large fraction of the competition duration.

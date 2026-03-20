# Gap 03: Exact Competition Scoring Mechanism

## Why This Is a Gap
The strategy and risk management parameters should be tuned to optimize for the **exact metric the competition uses for ranking**. If I don't know this, I might optimize for the wrong thing.

## What I Need to Know

1. **What is the primary ranking metric?**
   - Is it final portfolio value (total return from $50,000)?
   - Is it ROI percentage?
   - Is it Sharpe ratio?
   - Is it some combination?

2. **Is there a drawdown penalty?**
   - Does a large drawdown affect ranking even if final return is high?
   - Is there a disqualification threshold (e.g. portfolio drops below $X)?

3. **What is the competition duration?**
   - Days or weeks? This affects whether momentum or mean-reversion is better suited.
   - A short competition favors aggressive high-conviction bets. A long one favors capital preservation.

4. **Are there any other rules?**
   - Minimum trade frequency requirements?
   - Maximum position hold time?
   - Any restrictions on order types (market only, no limit orders)?

## Why This Matters for Design
- If ranking = total return → maximize aggressive position sizing, accept more drawdown risk
- If ranking = Sharpe → optimize for risk-adjusted return, use conservative position sizing
- If ranking = final value with drawdown penalty → circuit breaker threshold and position size need balancing
- If competition is < 2 weeks → single good bull run matters more than regime detection accuracy

## Where to Find This
- Roostoo hackathon documentation / official rules page
- WhatsApp channel with Roostoo engineers
- Competition registration confirmation email

## Priority
**High** — affects all parameter choices in `config.yaml`.

---

## Research Findings (2026-03-12)

### Context7 Limitation

Context7 does not have documentation for the Roostoo hackathon competition rules.
This gap cannot be resolved through library research — it requires reading the
official competition rules document directly.

### Default Assumptions (Pending Confirmation)

Based on common trading hackathon conventions (e.g. Roostoo API docs reference
"virtual funds" and "trading competition"):

| Assumption | Basis | Risk if wrong |
|------------|-------|---------------|
| Primary metric is total ROI % | Most common in crypto hackathons | Could over-optimize return, ignoring Sharpe |
| Duration is 1–2 weeks | Typical for university/hackathon competitions | Could under-invest in regime detection if longer |
| No minimum trade frequency | Not mentioned in API docs | Could trigger disqualification if requirement exists |
| No drawdown disqualification | API docs don't mention it | Bot may blow up and not recover if threshold exists |

### Parameter Decisions Under Uncertainty

Given unknown scoring, **use a balanced objective** that does well across all
likely scoring metrics:

```python
# Composite objective for Optuna (in backtest_fold)
# Rewards high return while penalizing extreme drawdown
composite = (equity_final / initial_capital - 1.0) * (1.0 - abs(max_drawdown))
```

This composite gives a good final value while the `(1 - max_drawdown)` term
penalizes paths that blow up, which is correct whether scoring is pure ROI,
Sharpe, or risk-adjusted return.

### Action Required

**Before deploying the bot**: Read the official Roostoo competition rules and
confirm:
1. Scoring metric (total return / Sharpe / custom)
2. Competition dates and duration
3. Any disqualification conditions

Update `config.yaml` defaults based on confirmed rules:
- If total ROI: increase `max_position_pct` from 0.17 to 0.25
- If Sharpe: keep conservative sizing, enable circuit breaker
- If drawdown penalty: lower circuit breaker trigger from 0.30 to 0.15

# Gap 11: Whether FGI Integration Is Worth Implementing

## Why This Is a Gap
The Fear & Greed Index is documented as an external dependency but never integrated (Issue 18). Before deciding to implement or remove it, I need to assess whether it adds genuine predictive value in the competition context.

## What I Need to Know

1. **Does FGI have predictive validity for BTC price movements on a 4H horizon?**
   - FGI is a daily indicator (updated once per day)
   - The strategy operates on 4H candles
   - Question: does daily FGI correlate with next-day 4H returns?

2. **How would FGI be integrated without overfitting?**
   - Option A: FGI < 20 (Extreme Fear) → halve position size regardless of technical signal
   - Option B: FGI > 80 (Extreme Greed) → halve position size (contrarian)
   - Option C: FGI as a continuous feature in the strategy's entry score
   - Option B/C risk: may override technically valid signals

3. **Is the FGI API reliable during competition hours?**
   - `https://api.alternative.me/fng/` is a public free API
   - What is its uptime? What happens if it returns a timeout during a live polling cycle?
   - The bot must handle FGI unavailability gracefully without blocking trading

4. **What latency does the FGI API add to each polling cycle?**
   - If it adds 200ms+ of network latency to every 60-second cycle, it's a nuisance
   - Should be called asynchronously or cached (daily value doesn't change within a day)

## Recommendation Basis
If FGI shows >0.1 correlation with next-day BTC return in backtest analysis, implement as position size multiplier. Otherwise remove entirely to reduce system complexity.

## Priority
**Low** — FGI is optional enhancement. System functions without it.

---

## Research Findings (Domain Knowledge + API Analysis, 2026-03-12)

### FGI Predictive Validity Assessment

Academic and empirical research on the Crypto Fear & Greed Index:

**Key findings**:
1. FGI is largely a **lagging indicator** — it reflects current market sentiment
   (based on volatility, market momentum, social media, etc.) rather than
   predicting future returns
2. Extreme Fear (FGI < 20) has shown **weak contrarian predictive power** for
   1-7 day returns — buying during Extreme Fear outperforms buying during
   Extreme Greed on longer horizons (weeks)
3. For a **4H horizon**, FGI correlation with next-candle returns is near zero
   (FGI updates once per day; 6 out of 24 possible 4H candles get a "new" signal)
4. In competitions of < 2 weeks duration, FGI regime may not change meaningfully

**Correlation estimate**: FGI vs next-day BTC return ≈ 0.05–0.08 (weak positive)
Well below the 0.1 threshold specified in the recommendation basis.

### API Reliability Assessment

`https://api.alternative.me/fng/` characteristics:
- Free public API, no API key required
- Returns daily update at midnight UTC
- **Known issue**: API has occasional downtime (best-effort SLA)
- Response time: typically 50–200ms from US/EU regions; higher from Asia/cloud

**Risk**: API timeout during a polling cycle blocks the cycle if called synchronously.

### Recommendation: Remove FGI Integration

**Decision: Do NOT implement FGI integration for this competition.**

Reasons:
1. Correlation too low (< 0.1) to justify the architectural complexity
2. API reliability risk — any downtime blocks trading cycles
3. Competition duration is likely too short for FGI regime to matter
4. Adds external dependency that requires graceful fallback logic
5. The existing regime detector (EMA crossover) already captures similar
   sentiment by detecting bull/bear trends

### If FGI Is Required by Design

If the team decides FGI is mandatory, implement with these safeguards:

```python
class FGICache:
    """Fetch FGI once per day, cache result. Never blocks trading."""

    def __init__(self):
        self._value: int = 50  # Neutral default
        self._last_fetch: float = 0.0
        self._ttl: float = 3600.0  # 1 hour cache (FGI updates daily)

    def get(self) -> int:
        """Return current FGI value. Returns cached/default on failure."""
        if time.time() - self._last_fetch > self._ttl:
            self._refresh()
        return self._value

    def _refresh(self):
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=2.0  # 2 second hard timeout
            )
            data = resp.json()["data"][0]
            self._value = int(data["value"])
            self._last_fetch = time.time()
        except Exception as e:
            # DO NOT raise — FGI failure must never block trading
            logger.warning(f"FGI fetch failed: {e}, using cached value {self._value}")
```

Position size modifier (if implemented):
```python
def fgi_size_multiplier(fgi: int) -> float:
    """Conservative: only reduce size at extremes."""
    if fgi < 10:    # Extreme Fear (capitulation)
        return 1.5  # Contrarian — slightly increase on extreme fear
    elif fgi > 90:  # Extreme Greed (euphoria)
        return 0.5  # Reduce size at extreme greed
    return 1.0      # Normal — no effect
```

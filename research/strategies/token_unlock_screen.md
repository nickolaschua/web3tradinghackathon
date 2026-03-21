# Strategy 4: Token Unlock Negative Screen

## What it is

Token unlocks are scheduled vesting releases — pre-allocated coins given to team members,
investors, or ecosystem funds become transferable on a specific date. When a large unlock
approaches, insiders and market makers pre-position for the expected selling pressure:
- Price decline begins ~30 days before the unlock
- The impact peaks around the unlock date
- Team/investor unlocks average −25% impact; medium-sized unlocks (1-5% of supply) average −0.3%

This is a **supply-side mechanical event**, not an informational signal. The model cannot learn
it from OHLCV alone because the upcoming unlock is calendar data, not price data.

Implementation: before each rebalance, flag coins with large upcoming unlocks in the next 7 days.
Zero-weight or heavily underweight those coins in the portfolio regardless of model signal. This
is a pre-model filter, not an XGBoost feature (too sparse for XGBoost to learn reliably).

---

## How to implement in this codebase

### Data source

**Tokenomist.ai** (formerly TokenUnlocks.app) provides free unlock tracking for 500+ tokens.
The simplest approach requires no API — just a one-time manual check before the competition week
starts and a hardcoded exclusion list. For automation, they offer a free-tier API.

Endpoint (unofficial, may change — verify before competition):
```
GET https://api.tokenomist.ai/v1/unlocks?upcoming=7d
```

For hackathon purposes, the manual approach is more reliable:

1. Go to tokenomist.ai
2. Filter by: "next 7 days", sort by "% of circulating supply"
3. Cross-reference with the hackathon's 39-coin universe
4. Record any coin with > 0.5% supply unlock in the competition window

### Implementation: hardcoded exclusion list

The simplest correct implementation — hardcode the exclusion set before competition start:

```python
# bot/config/unlock_screen.py
"""
Token unlock exclusion screen.

Update UNLOCK_EXCLUSIONS before each competition week by checking tokenomist.ai.
Coins in this set receive zero weight regardless of model signal.

Last updated: [DATE]
Source: tokenomist.ai
"""
from datetime import date

# Coins to exclude due to large token unlocks during competition window.
# Key: pair symbol (Roostoo format)
# Value: dict with unlock details for audit trail
UNLOCK_EXCLUSIONS: dict[str, dict] = {
    # Example — replace with actual competition-week unlocks:
    # "AVAX/USD": {
    #     "unlock_date": date(2026, 3, 25),
    #     "pct_supply": 2.3,
    #     "unlock_type": "investor",    # team | investor | ecosystem
    #     "source": "tokenomist.ai",
    # },
}

# Unlock types ranked by severity (team = worst)
SEVERITY = {"team": 3, "investor": 2, "ecosystem": 1}

def should_exclude(pair: str, exclusions: dict = UNLOCK_EXCLUSIONS) -> bool:
    """Return True if this coin has a large upcoming unlock."""
    entry = exclusions.get(pair)
    if entry is None:
        return False
    # Only exclude team and investor unlocks > 0.5% supply
    pct = entry.get("pct_supply", 0)
    kind = entry.get("unlock_type", "ecosystem")
    return pct >= 0.5 and SEVERITY.get(kind, 0) >= 2
```

### Integration in the signal pipeline

Apply the screen after model scoring, before position sizing:

```python
# In bot/execution/portfolio.py or main.py, after generating model signals:

from bot.config.unlock_screen import should_exclude

def apply_unlock_screen(signals: dict[str, TradingSignal]) -> dict[str, TradingSignal]:
    """
    Zero out any BUY signals for coins with large upcoming unlocks.

    Does not affect SELL or HOLD signals — if you're already long a coin that
    just appeared on the unlock list, let the exit logic handle it normally.
    """
    filtered = {}
    for pair, signal in signals.items():
        if signal.direction == SignalDirection.BUY and should_exclude(pair):
            logger.info(
                "Unlock screen: suppressing BUY for %s (upcoming token unlock)", pair
            )
            filtered[pair] = TradingSignal(pair=pair)  # HOLD
        else:
            filtered[pair] = signal
    return filtered
```

### For backtesting

Build a lookup table of historical unlock dates per coin to test if the screen adds value:

```python
# Historical unlock data — requires one-time manual research from tokenomist.ai
# or their API's historical endpoint
HISTORICAL_UNLOCKS = pd.DataFrame([
    # {"pair": "AVAX/USD", "date": "2024-03-15", "pct_supply": 1.8, "type": "investor"},
    # ... add more
])

def get_unlock_flag(pair: str, timestamp: pd.Timestamp, lookahead_days: int = 7) -> int:
    """Return 1 if a large unlock is within lookahead_days of timestamp."""
    if HISTORICAL_UNLOCKS.empty:
        return 0
    mask = (
        (HISTORICAL_UNLOCKS["pair"] == pair) &
        (pd.to_datetime(HISTORICAL_UNLOCKS["date"]) > timestamp) &
        (pd.to_datetime(HISTORICAL_UNLOCKS["date"]) <= timestamp + pd.Timedelta(days=lookahead_days)) &
        (HISTORICAL_UNLOCKS["pct_supply"] >= 0.5) &
        (HISTORICAL_UNLOCKS["type"].isin(["team", "investor"]))
    )
    return int(mask.any())
```

### Automated fetch (optional, if Tokenomist API is available)

```python
import requests

def fetch_upcoming_unlocks(days_ahead: int = 7) -> list[dict]:
    """
    Fetch upcoming token unlocks from Tokenomist.ai.
    Returns list of dicts with keys: symbol, unlockDate, pctSupply, type.
    """
    try:
        resp = requests.get(
            "https://api.tokenomist.ai/v1/unlocks",
            params={"upcoming": f"{days_ahead}d", "minPct": 0.5},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        logger.warning("Tokenomist API fetch failed: %s — using hardcoded list", e)
        return []

def build_exclusion_set(universe_pairs: list[str], days_ahead: int = 7) -> set[str]:
    """
    Build the live exclusion set from Tokenomist API.
    Falls back to hardcoded UNLOCK_EXCLUSIONS if API fails.
    """
    # Map common token names to Roostoo pair format
    TOKEN_TO_PAIR = {
        "BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD",
        "AVAX": "AVAX/USD", "MATIC": "MATIC/USD", "LINK": "LINK/USD",
        # ... complete mapping
    }
    unlocks = fetch_upcoming_unlocks(days_ahead)
    exclusions = set()
    for unlock in unlocks:
        token = unlock.get("symbol", "")
        pair  = TOKEN_TO_PAIR.get(token)
        if pair and pair in universe_pairs:
            pct  = float(unlock.get("pctSupply", 0))
            kind = unlock.get("type", "ecosystem").lower()
            if pct >= 0.5 and kind in ("team", "investor"):
                exclusions.add(pair)
                logger.info(
                    "Unlock screen: flagging %s — %.1f%% supply unlock (%s) in %dd",
                    pair, pct, kind, days_ahead
                )
    return exclusions or {p for p in UNLOCK_EXCLUSIONS if should_exclude(p)}
```

---

## How to check for correctness

### Pre-competition verification checklist

Before competition week, manually verify:

```
1. Go to tokenomist.ai → Unlock Calendar → next 7 days
2. Sort by "% of Circulating Supply" descending
3. Cross-reference with the hackathon universe (39 coins)
4. For each flagged coin:
   - Record: pair, unlock date, % supply, type (team/investor/ecosystem)
   - Verify the data matches Coingecko or the project's official vesting schedule
5. Update UNLOCK_EXCLUSIONS in bot/config/unlock_screen.py
6. Run: python -c "from bot.config.unlock_screen import UNLOCK_EXCLUSIONS; print(UNLOCK_EXCLUSIONS)"
```

### Unit test for the screen

```python
# tests/test_unlock_screen.py
from bot.config.unlock_screen import should_exclude, UNLOCK_EXCLUSIONS

def test_exclude_large_investor_unlock():
    UNLOCK_EXCLUSIONS["TEST/USD"] = {
        "unlock_date": ...,
        "pct_supply": 2.0,
        "unlock_type": "investor",
    }
    assert should_exclude("TEST/USD") is True

def test_keep_small_ecosystem_unlock():
    UNLOCK_EXCLUSIONS["SMALL/USD"] = {
        "unlock_date": ...,
        "pct_supply": 0.3,
        "unlock_type": "ecosystem",
    }
    assert should_exclude("SMALL/USD") is False

def test_keep_ecosystem_large():
    UNLOCK_EXCLUSIONS["ECO/USD"] = {
        "unlock_date": ...,
        "pct_supply": 5.0,
        "unlock_type": "ecosystem",  # ecosystem unlocks are net positive on average
    }
    assert should_exclude("ECO/USD") is False
```

### Signal suppression log

Verify the screen is actually firing during the competition:

```python
# Add to apply_unlock_screen():
if any(should_exclude(p) for p in signals):
    excluded = [p for p in signals if should_exclude(p) and signals[p].direction == SignalDirection.BUY]
    logger.info("Unlock screen suppressed BUY signals for: %s", excluded)
```

---

## Maximizing value

### Grade unlocks by severity

Not all unlocks are equal. Use a weighting scheme:
- Team vesting: **full exclusion** (average -25% impact)
- Investor vesting: **full exclusion** (average -12% impact, but right tail risk is severe)
- Ecosystem/development: **reduce position size by 50%** (average +1.18% impact but high variance)
- Protocol rewards: **no action** (small, continuous, already priced in)

### 30-day pre-positioning is the real signal

The research shows price decline starts ~30 days before unlock, not on the unlock date itself.
If competition runs early in a month, check for unlocks in the next 30 days, not just the
next 7. Add an extended exclusion list with reduced-weight (not zero-weight) for coins with
unlocks in 7-30 days:

```python
UNLOCK_REDUCED_WEIGHT: dict[str, float] = {
    # pair → weight multiplier (e.g. 0.5 means half position size)
    # "SOL/USD": 0.5,   # large unlock in 3 weeks
}
```

### Track if the screen fires correctly post-competition

After the competition, look back at whether coins that were unlocked actually dropped. Build a
simple empirical validation:

```python
# For each historical unlock in your tracking list:
# compute return from D-7 to D+7 for excluded coins
# compare to universe average return over same period
# confirm negative alpha
```

This tells you whether the screen is adding value or just reducing exposure to random coins.

---

## Common pitfalls

### Pitfall 1: Tokenomist data is delayed or incorrect

Tokenomist crowdsources some unlock data from block explorers and project announcements. For
smaller coins, the data can be wrong (wrong date, wrong percentage, wrong category). Always
cross-verify large unlocks against the project's official tokenomics document or on-chain
vesting contract before adding to the exclusion list.

### Pitfall 2: Ecosystem unlocks are often positive

The research (Keyrock) shows ecosystem/development unlocks average +1.18% — slightly positive
because they signal active protocol development. Don't blindly exclude all unlocks. The harmful
categories are **team** and **investor** (insiders selling).

### Pitfall 3: The unlock is priced in already

If an unlock was announced months ago and the market has been pricing it in for 30 days, by
competition week there may be little incremental impact remaining. The screen is most valuable
when applied 0-7 days before the event, less valuable when the price has already fallen in
anticipation. Check the coin's recent price action — if it's already down 15-20% over the past
month, the unlock may already be priced in.

### Pitfall 4: Don't use as an XGBoost feature

Sparse binary features (1 on 2-3 days per year per coin, 0 on all other days) create extreme
class imbalance that XGBoost cannot learn from reliably. XGBoost needs many examples of each
condition. With perhaps 5-10 major unlocks across 39 coins per year, you have tens of examples
total — far too few for a tree-based model. Keep this as a rule-based filter outside the model.

### Pitfall 5: Unlock timing can slip

Projects occasionally delay unlock dates by days or weeks, especially if they need governance
approval. If you've hard-excluded a coin for a specific date and the project announces a delay,
the coin may have been unfairly excluded from a profitable signal. Build a mechanism to update
the exclusion list mid-competition if needed:

```python
# Quick update via env var override:
import os
EXCLUDED_OVERRIDE = set(os.environ.get("UNLOCK_EXCLUDE", "").split(","))
# In should_exclude: return (hardcoded check) or (pair in EXCLUDED_OVERRIDE)
```

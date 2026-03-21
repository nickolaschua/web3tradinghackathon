"""
Token unlock exclusion screen.

Update UNLOCK_EXCLUSIONS before each competition week by checking tokenomist.ai:
  1. Go to tokenomist.ai → Unlock Calendar → next 7 days
  2. Sort by "% of Circulating Supply" descending
  3. Cross-reference with the hackathon universe (39 coins)
  4. For each flagged coin (team/investor unlock >= 0.5% supply), add an entry below.

Last updated: 2026-03-21 (Round 1 start)
Source: tokenomist.ai — Vested Unlock preset, $100M market cap minimum filter

Reference: research/strategies/token_unlock_screen.md
"""
from __future__ import annotations

import logging
import os
from datetime import date

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coins to exclude due to large token unlocks during competition window.
# Key: pair symbol in Roostoo format (e.g. "AVAX/USD")
# Value: dict with unlock details for audit trail
# ---------------------------------------------------------------------------
UNLOCK_EXCLUSIONS: dict[str, dict] = {
    # Populated 2026-03-21 from tokenomist.ai Vested Unlock calendar (all entries are
    # team/investor vested unlocks; Vested Unlock filter was active during export).
    "SUI/USD": {
        "unlock_date": date(2026, 4, 1),
        "pct_supply": 1.10,
        "unlock_type": "investor",
        "source": "tokenomist.ai",
    },
    "ENA/USD": {
        "unlock_date": date(2026, 4, 2),
        "pct_supply": 0.52,
        "unlock_type": "investor",
        "source": "tokenomist.ai",
    },
}

# Coins with upcoming unlocks in the 7-30 day window (reduce size, don't zero out).
# Key: pair symbol; Value: weight multiplier (0.0-1.0)
UNLOCK_REDUCED_WEIGHT: dict[str, float] = {
    # "SOL/USD": 0.5,  # large unlock in 3 weeks — halve position size
}

# ---------------------------------------------------------------------------
# ROUND 2 PREP (if finalist — Round 2: Apr 4–Apr 14):
# Remove SUI/ENA (already unlocked). Add these to UNLOCK_EXCLUSIONS:
#   "APT/USD": unlock_date=date(2026,4,12), pct_supply=0.68, unlock_type="investor"
#     → Apr 12 is within Round 2; pre-unlock selling starts ~Apr 5
#   "SEI/USD": unlock_date=date(2026,4,15), pct_supply=0.97, unlock_type="investor"
#     → Unlocks day after Round 2 ends; pre-unlock pressure starts ~Apr 8 (within Round 2)
#   "ARB/USD": unlock_date=date(2026,4,16), pct_supply=1.75, unlock_type="investor"
#     → Unlocks 2 days after Round 2; pre-unlock pressure starts ~Apr 9 (within Round 2)
# ---------------------------------------------------------------------------

# Unlock type severity ranking (higher = worse for price)
_SEVERITY: dict[str, int] = {"team": 3, "investor": 2, "ecosystem": 1}

# Override via env var: UNLOCK_EXCLUDE="BTC/USD,ETH/USD" to add exclusions at runtime
_EXCLUDED_OVERRIDE: frozenset[str] = frozenset(
    p.strip() for p in os.environ.get("UNLOCK_EXCLUDE", "").split(",") if p.strip()
)


def should_exclude(pair: str, exclusions: dict | None = None) -> bool:
    """
    Return True if this coin has a large upcoming team or investor unlock.

    Ecosystem unlocks (avg +1.18% impact per research) are NOT excluded by default.
    Only team and investor unlocks >= 0.5% of circulating supply trigger exclusion.

    Args:
        pair:       Roostoo pair symbol, e.g. "AVAX/USD".
        exclusions: Override the UNLOCK_EXCLUSIONS dict (for testing).

    Returns:
        True if BUY signals for this pair should be suppressed.
    """
    if pair in _EXCLUDED_OVERRIDE:
        logger.info("Unlock screen: %s excluded via UNLOCK_EXCLUDE env var", pair)
        return True

    source = exclusions if exclusions is not None else UNLOCK_EXCLUSIONS
    entry = source.get(pair)
    if entry is None:
        return False

    pct  = entry.get("pct_supply", 0.0)
    kind = entry.get("unlock_type", "ecosystem").lower()
    if pct >= 0.5 and _SEVERITY.get(kind, 0) >= 2:
        logger.info(
            "Unlock screen: excluding %s (%.1f%% supply %s unlock)",
            pair, pct, kind,
        )
        return True
    return False


def get_size_multiplier(pair: str) -> float:
    """
    Return a position size multiplier for coins with near-term but non-exclusion unlocks.

    Returns 1.0 for coins with no upcoming unlock concern; a value < 1.0 if the coin
    appears in UNLOCK_REDUCED_WEIGHT (e.g. 0.5 = halve position size).
    """
    return UNLOCK_REDUCED_WEIGHT.get(pair, 1.0)


def apply_unlock_screen(signals: dict) -> dict:
    """
    Zero out BUY signals for coins with large upcoming team/investor unlocks.

    Does NOT affect SELL or HOLD signals — if already long a coin that appeared
    on the unlock list, let the normal exit logic handle the close.

    Args:
        signals: Dict mapping pair symbol → TradingSignal.

    Returns:
        Filtered dict with BUY signals suppressed for excluded coins.
    """
    from bot.strategy.base import TradingSignal, SignalDirection

    filtered: dict = {}
    for pair, signal in signals.items():
        if signal.direction == SignalDirection.BUY and should_exclude(pair):
            logger.info(
                "Unlock screen: suppressing BUY for %s (upcoming token unlock)", pair
            )
            filtered[pair] = TradingSignal(pair=pair)  # HOLD
        else:
            filtered[pair] = signal
    return filtered


__all__ = [
    "UNLOCK_EXCLUSIONS",
    "UNLOCK_REDUCED_WEIGHT",
    "should_exclude",
    "get_size_multiplier",
    "apply_unlock_screen",
]

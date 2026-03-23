# Codebase Audit — Issues & Potential Problems

Ongoing audit of the live trading bot. Issues fixed in severity order.

---

## CRITICAL — ALL FIXED (6/6)
1-6: Concentration check, free_usd, pyramid VWAP, close_position phantom, trailing stop reset, cold-start HWM.

## HIGH — ALL FIXED (5/5)
7-11: Per-pair regime detectors, robust traded_today flag, stop-loss updates last_trade, partial fill handling, reconciliation against wallet holdings. #12 reclassified as not a bug.

## MEDIUM — 2 FIXED, 7 DESIGN CHOICES
13: Stop-loss exits skip 65s cooldown. 20: Per-pair regime state persisted across restarts.
14-19, 21: Intentional design choices — documented and accepted.

## LOW — 2 FIXED, 9 ACCEPTED
26: Completed orders pruned after 1 hour. 27: Time re-syncs every 6 hours.
22-25, 28-32: Accepted — low risk, not worth the churn.

---

## Remaining Accepted Items (no action needed)

- **22** Private attribute access on LiveFetcher — cosmetic, competition bot
- **23** Monkey-patching `_source` on dataclass — works, logging-only use
- **24** Dead code paths (MR strategy, pairs ML) — may re-enable in Round 2
- **25** Regime not checked in reconciliation — mitigated by fix #20
- **28** No model feature validation at startup — XGBoost handles NaN natively
- **29** window=2880 vs buffer coupling — maxlen=4000, documented
- **30** Telegram HTML parse mode — rare edge case, messages still logged
- **31** Hardcoded pair mapping — covers full Roostoo universe
- **32** No watchdog — state persistence handles restarts, tmux adequate for 10 days

---

## Summary

| # | Severity | Status |
|---|----------|--------|
| 1-6 | CRITICAL | FIXED |
| 7-11 | HIGH | FIXED |
| 13, 20 | MEDIUM | FIXED |
| 14-19, 21 | MEDIUM | DESIGN CHOICE |
| 26, 27 | LOW | FIXED |
| 22-25, 28-32 | LOW | ACCEPTED |

**Total: 15 bugs fixed, 7 design choices documented, 9 accepted as low-risk.**

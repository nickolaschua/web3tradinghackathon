# Phase 6: Strategy Interface - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<vision>
## How This Should Work

This phase produces clean scaffolding that the user opens and fills in. When you open `momentum.py`, you should know immediately — without reading any other file — exactly what signals to implement. The interface contract is locked in by the build, and the alpha logic slots are clearly marked for the user.

The stubs exist so the user can focus 100% on their trading idea, not on figuring out what shape to return or how to wire into the rest of the system.

</vision>

<essential>
## What Must Be Nailed

- **Correct interface contract** — `TradingSignal.pair` is a required positional field (no default), `generate_signal` signature is exact. Every other layer depends on this being right from day one.
- **Stub usability** — Each stub has a detailed docstring (contract, available features, return expectations, concrete example) AND inline comments marking where to add logic conditions. User fills in alpha without needing to read any other file.

Both are non-negotiable.

</essential>

<boundaries>
## What's Out of Scope

- No real alpha/signal logic — stubs return HOLD by default; actual momentum and mean-reversion conditions are left entirely for the user to implement
- No backtesting hooks — the strategy interface is for live trading only; vectorbt/backtesting.py integration is not part of this phase

</boundaries>

<specifics>
## Specific Ideas

- Stubs should return a neutral HOLD signal out of the box so the bot starts safely without any alpha logic filled in
- Docstring should list every feature column the user can reference (from compute_features()) so they don't have to dig through data pipeline code
- Inline comments should mark the entry logic zone and exit logic zone clearly, e.g. `# --- ADD YOUR ENTRY CONDITIONS HERE ---`

</specifics>

<notes>
## Additional Context

This phase is part of the multi-agent build. Agents 1-3 own API/data/execution layers. Agent 4 owns strategy stubs and orchestration. Phase 6 (strategy stubs) can be built immediately without waiting for other agents. Phase 7 (main.py) must wait for Agents 1-3 to finish.

The critical bug to fix here: Issue 10 — `TradingSignal.pair` currently defaults to `""`, causing all order submissions to silently fail. Making `pair` required with no default is the single most important correctness fix in this phase.

</notes>

---

*Phase: 06-strategy-interface*
*Context gathered: 2026-03-16*

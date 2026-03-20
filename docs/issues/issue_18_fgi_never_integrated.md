# Issue 18: Fear & Greed Index Documented But Never Integrated

## Layer
Layer 1 — External Dependencies (`docs/01_layer0_external_dependencies.md`)
Layer 5 — Strategy Engine (design intent)

## Description
`docs/01_layer0_external_dependencies.md` documents the Alternative.me Fear & Greed Index (FGI) as an external dependency (`https://api.alternative.me/fng/`), including a complete `FearGreedClient` reference implementation and described use as a regime filter overlay.

However, FGI is:
- Not imported or called in `execution/regime.py`
- Not used in `strategy/momentum.py` or `strategy/mean_reversion.py`
- Not part of any feature engineering pipeline
- Not referenced in any state persistence
- Not present in `config.yaml` (which itself doesn't exist yet)

The FGI integration is fully documented but completely missing from all implementation layers.

## Impact
**Low** — FGI is optional. The system functions without it. However, the doc creates a false expectation that it's part of the system. If it's not going to be implemented, remove references to avoid confusion.

## Resolution Options
1. **Implement it**: Add FGI as a daily sentiment overlay that modifies the regime multiplier (e.g. Extreme Fear → halve position size regardless of EMA regime)
2. **Remove it**: Delete from Layer 0 docs and don't implement — reduces complexity, one fewer HTTP dependency
3. **Defer it**: Mark as "optional enhancement" in docs, clearly separated from core system

Recommendation: Implement as a simple multiplier overlay. The API call is trivial and daily FGI adds a genuine signal that is uncorrelated with technical indicators.

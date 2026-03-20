# Issue 23: HRP/CVaR Correlation Assumptions Break Down During Systemic Crashes

## Layer
Layer 6 — Risk Management / PortfolioAllocator

## Description
HRP (Hierarchical Risk Parity) clusters assets by their historical correlation structure and
weights them inversely proportional to cluster variance. CVaR minimization picks allocations
based on historical tail losses.

Both assumptions fail during systemic crypto crashes:

1. **Correlation collapse**: In normal markets BTC, ETH, SOL have moderate correlation (~0.6-0.7).
   During a crash, all three move together with correlation approaching 1.0. HRP clustering
   produces degenerate results — all assets land in one cluster, giving near-equal weights
   (same as the fallback). The diversification benefit disappears exactly when it's most needed.

2. **CVaR historical bias**: CVaR uses historical return simulation. If the historical window
   (the last 60+ bars) was mostly bullish, the model underestimates tail risk going into a
   crash. The optimizer may put maximum weight on the asset that *appeared* least risky
   historically, which could be the one about to crash hardest.

3. **Rebalancing lag**: `compute_weights()` runs once per 4H boundary. Weight adjustments
   lag behind the crash by up to 4 hours.

## Impact
**Medium** — The 50/50 HRP/CVaR blend with equal-weight fallback provides reasonable
protection, but the diversification benefit is illusory during correlated crashes.
Since Roostoo only supports BTC/USD as a tradeable pair (see Issue 20), the portfolio
weight for BTC will always be 1.0 in practice, making this a latent issue for
multi-asset deployments.

## Fix Required
No immediate code fix required given single-pair constraint (Issue 20).
Document that HRP/CVaR weights are only meaningful when multiple uncorrelated
tradeable pairs are available.

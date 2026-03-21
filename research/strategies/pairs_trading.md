# Strategy 6: Long-Only Pairs Trading (Cointegration-Based)

## What it is

Classical pairs trading shorts the outperformer and buys the underperformer when their price
spread deviates from its historical mean, profiting when they revert. Because the Roostoo API
is spot-only (BUY and SELL of individual coins — no short selling, no margin), this strategy
must be adapted to **long-only pairs trading**:

- When the spread between two cointegrated coins widens, buy the **laggard** (the underperformer)
- Exit when the spread reverts (the laggard has caught up)
- No position in the outperformer — only ever long one side of the pair

This is weaker than classical delta-neutral pairs trading (which hedges away market risk entirely),
but it still captures the mean-reversion in the relative performance of coins that move together
structurally. The edge is: when two historically-cointegrated coins diverge, the laggard tends
to close the gap rather than the leader reversing.

The pairs most likely to cointegrate in the hackathon's 39-coin universe:
- BTC/ETH (dominant reserve assets, near-perfect co-movement)
- ETH/BNB (large-cap Layer-1s with overlapping DeFi exposure)
- SOL/AVAX, SOL/NEAR (competing Layer-1 ecosystems)
- BNB/MATIC, ETH/MATIC (Ethereum ecosystem pairs)
- LINK/BAND (oracle pairs)

---

## How to implement in this codebase

### Step 1: Install statsmodels (if not already installed)

```bash
pip install statsmodels
```

The cointegration test uses `statsmodels.tsa.stattools.coint` (Engle-Granger test). The OLS
spread regression uses `statsmodels.api.OLS`. Both are in the standard package.

### Step 2: Create `bot/strategy/pairs_trading.py`

This is a **second independent strategy pipeline** — a new file, not integrated into the existing
`MomentumStrategy` or `MeanReversionStrategy` classes:

```python
"""
Long-only pairs trading strategy using Engle-Granger cointegration.

Because Roostoo is spot-only (no shorting), this strategy takes only the LONG leg:
- Buy the laggard when spread widens beyond +1.5 standard deviations
- Exit when spread reverts to within 0.5 standard deviations of mean

The "spread" is the residual of the OLS regression:
    log(price_A) = alpha + beta * log(price_B) + epsilon
When epsilon (the residual) is large and positive (A is expensive relative to B), buy B.
When epsilon is large and negative (A is cheap relative to B), buy A.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Engle-Granger test p-value threshold for cointegration
COINT_PVALUE_MAX = 0.10   # accept up to 10% — looser than the usual 5% to get more pairs

# Spread entry and exit thresholds (in standard deviations)
ENTRY_ZSCORE = 1.5    # enter long laggard when spread is >1.5 std from mean
EXIT_ZSCORE  = 0.5    # exit when spread reverts to within 0.5 std of mean
STOP_ZSCORE  = 3.0    # emergency exit if spread widens further to 3.0 std (spread breaking down)

# Minimum bars for cointegration test
MIN_BARS = 100         # ~16 days of 4H data

# How often to re-run the cointegration test (in bars)
RETEST_INTERVAL = 42   # re-test every 7 days


@dataclass
class PairState:
    """Tracks the live state of a single cointegrated pair."""
    pair_a: str          # e.g. "ETH/USD"
    pair_b: str          # e.g. "BNB/USD"
    beta: float = 1.0    # OLS coefficient: log(A) = alpha + beta * log(B) + eps
    alpha: float = 0.0   # OLS intercept
    spread_mean: float = 0.0
    spread_std: float = 1.0
    last_bar_tested: int = 0   # bar index of last cointegration test
    is_cointegrated: bool = False
    long_pair: Optional[str] = None    # which pair we're currently long (or None)


class PairsTradingStrategy:
    """
    Long-only cointegration-based pairs trading strategy.

    Usage:
        strategy = PairsTradingStrategy(config={})
        strategy.add_candidate_pair("ETH/USD", "BNB/USD")

        # In the main loop, once per 4H bar:
        for pair_state in strategy.active_pairs:
            signals = strategy.update(pair_state, coin_dfs, bar_index)
            # signals is a list of TradingSignal objects (0, 1, or 2 items)
    """

    def __init__(self, config: dict):
        self.config = config
        self.candidate_pairs: list[tuple[str, str]] = []
        self.pair_states: list[PairState] = []

    def add_candidate_pair(self, pair_a: str, pair_b: str) -> None:
        """Register a candidate pair for cointegration testing."""
        self.candidate_pairs.append((pair_a, pair_b))
        self.pair_states.append(PairState(pair_a=pair_a, pair_b=pair_b))

    def _test_cointegration(
        self, log_price_a: pd.Series, log_price_b: pd.Series
    ) -> tuple[bool, float, float]:
        """
        Run Engle-Granger cointegration test and compute OLS hedge ratio.

        Returns:
            (is_cointegrated, beta, alpha)
        """
        from statsmodels.tsa.stattools import coint
        from statsmodels.api import OLS, add_constant

        if len(log_price_a) < MIN_BARS or len(log_price_b) < MIN_BARS:
            return False, 1.0, 0.0

        # Align on common index
        aligned = pd.concat([log_price_a, log_price_b], axis=1).dropna()
        if len(aligned) < MIN_BARS:
            return False, 1.0, 0.0

        la = aligned.iloc[:, 0]
        lb = aligned.iloc[:, 1]

        # Engle-Granger test: is la - beta*lb stationary?
        try:
            _, pvalue, _ = coint(la, lb)
        except Exception as e:
            logger.warning("Cointegration test failed: %s", e)
            return False, 1.0, 0.0

        if pvalue > COINT_PVALUE_MAX:
            return False, 1.0, 0.0

        # OLS: la = alpha + beta * lb + residuals
        X = add_constant(lb)
        try:
            ols = OLS(la, X).fit()
            alpha, beta = ols.params
        except Exception as e:
            logger.warning("OLS regression failed: %s", e)
            return False, 1.0, 0.0

        return True, float(beta), float(alpha)

    def _compute_spread(
        self, log_price_a: pd.Series, log_price_b: pd.Series, beta: float, alpha: float
    ) -> pd.Series:
        """
        Compute the spread (OLS residual): log(A) - alpha - beta * log(B).

        A positive spread means A is overpriced relative to B → laggard is B (buy B).
        A negative spread means A is underpriced relative to B → laggard is A (buy A).
        """
        aligned = pd.concat([log_price_a, log_price_b], axis=1).dropna()
        la = aligned.iloc[:, 0]
        lb = aligned.iloc[:, 1]
        return la - alpha - beta * lb

    def update(
        self,
        state: PairState,
        coin_dfs: dict[str, pd.DataFrame],
        bar_index: int,
    ) -> list:
        """
        Update a pair's cointegration state and generate trading signals.

        Args:
            state:     PairState for this pair.
            coin_dfs:  Dict mapping pair symbol → OHLCV DataFrame (with DatetimeIndex).
            bar_index: Current bar counter (used to throttle cointegration re-testing).

        Returns:
            List of TradingSignal objects. Empty list = no action.
        """
        from bot.strategy.base import SignalDirection, TradingSignal

        df_a = coin_dfs.get(state.pair_a)
        df_b = coin_dfs.get(state.pair_b)
        if df_a is None or df_b is None or df_a.empty or df_b.empty:
            return []

        log_a = np.log(df_a["close"])
        log_b = np.log(df_b["close"])

        # Re-test cointegration periodically
        bars_since_test = bar_index - state.last_bar_tested
        if bars_since_test >= RETEST_INTERVAL or not state.is_cointegrated:
            is_coint, beta, alpha = self._test_cointegration(log_a, log_b)
            state.is_cointegrated = is_coint
            state.beta = beta
            state.alpha = alpha
            state.last_bar_tested = bar_index

            if is_coint:
                # Recompute spread statistics on the full history
                spread = self._compute_spread(log_a, log_b, beta, alpha)
                state.spread_mean = float(spread.mean())
                state.spread_std  = float(spread.std() + 1e-10)
                logger.info(
                    "Pair %s/%s: cointegrated (p<%.2f), beta=%.3f, "
                    "spread_mean=%.4f, spread_std=%.4f",
                    state.pair_a, state.pair_b, COINT_PVALUE_MAX,
                    beta, state.spread_mean, state.spread_std,
                )
            else:
                logger.debug(
                    "Pair %s/%s: NOT cointegrated at bar %d",
                    state.pair_a, state.pair_b, bar_index,
                )
                return []

        if not state.is_cointegrated:
            return []

        # Current spread z-score (use only the last bar — no look-ahead)
        current_spread = float(log_a.iloc[-2] - state.alpha - state.beta * log_b.iloc[-2])
        zscore = (current_spread - state.spread_mean) / state.spread_std

        signals = []

        # --- Exit conditions ---
        if state.long_pair is not None:
            should_exit = (
                abs(zscore) < EXIT_ZSCORE or    # spread has reverted
                abs(zscore) > STOP_ZSCORE        # spread breaking down (stop loss)
            )
            if should_exit:
                signals.append(TradingSignal(
                    pair=state.long_pair,
                    direction=SignalDirection.SELL,
                    size=1.0,
                    confidence=0.65,
                ))
                logger.info(
                    "Pairs exit: %s (zscore=%.2f)", state.long_pair, zscore
                )
                state.long_pair = None
            return signals  # don't generate new entries while managing an exit

        # --- Entry conditions ---
        if abs(zscore) > ENTRY_ZSCORE:
            # Positive spread: A is overpriced vs B → laggard is B
            # Negative spread: A is underpriced vs B → laggard is A
            laggard = state.pair_b if zscore > 0 else state.pair_a

            signals.append(TradingSignal(
                pair=laggard,
                direction=SignalDirection.BUY,
                size=0.25,    # modest size — pairs trade is a secondary strategy
                confidence=0.60,
            ))
            state.long_pair = laggard
            logger.info(
                "Pairs entry: long %s (zscore=%.2f)", laggard, zscore
            )

        return signals
```

### Step 3: Initialize in `main.py`

```python
from bot.strategy.pairs_trading import PairsTradingStrategy

pairs_strategy = PairsTradingStrategy(config=config)

# Register candidate pairs — these are the highest-probability cointegrated pairs
# in the hackathon's 39-coin universe. Run a pre-competition test to confirm:
pairs_strategy.add_candidate_pair("ETH/USD", "BNB/USD")
pairs_strategy.add_candidate_pair("SOL/USD", "AVAX/USD")
pairs_strategy.add_candidate_pair("ETH/USD", "SOL/USD")
# Add BTC/ETH only if performance improves — they're so correlated the spread is tiny
```

### Step 4: Main loop integration

```python
# After generating momentum/mean-reversion signals:
bar_index = 0  # increment each 4H bar

for state in pairs_strategy.pair_states:
    pair_signals = pairs_strategy.update(state, coin_dfs, bar_index)
    for sig in pair_signals:
        # Check for conflicts with existing momentum positions
        existing_signal = final_signals.get(sig.pair)
        if existing_signal and existing_signal.direction != SignalDirection.HOLD:
            # Momentum has a position — don't overlay pairs on top
            logger.debug("Pairs signal for %s skipped: momentum active", sig.pair)
        else:
            final_signals[sig.pair] = sig

bar_index += 1
```

---

## How to check for correctness

### Pre-competition cointegration test

Before competition start, run the test manually to confirm which pairs are actually cointegrated
in the recent 90-day window:

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

# Load historical 4H close prices for candidate pairs
def run_coint_scan(coin_dfs: dict[str, pd.DataFrame], candidates: list[tuple[str, str]]):
    results = []
    for pair_a, pair_b in candidates:
        log_a = np.log(coin_dfs[pair_a]["close"])
        log_b = np.log(coin_dfs[pair_b]["close"])
        aligned = pd.concat([log_a, log_b], axis=1).dropna()
        if len(aligned) < 100:
            continue
        _, pvalue, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
        results.append({"pair": f"{pair_a} / {pair_b}", "pvalue": round(pvalue, 4)})

    df = pd.DataFrame(results).sort_values("pvalue")
    print(df.to_string(index=False))
    return df

# Expected: ETH/BNB and SOL/AVAX typically cointegrate at p < 0.05
# BTC/ETH may have p < 0.01 but the beta ≈ 1 and spread is tiny in absolute terms
```

### Spread stationarity check

After fitting the OLS hedge ratio, verify the spread residuals are actually mean-reverting:

```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.api import OLS, add_constant

log_a = np.log(coin_dfs["ETH/USD"]["close"])
log_b = np.log(coin_dfs["BNB/USD"]["close"])
aligned = pd.concat([log_a, log_b], axis=1).dropna()

X = add_constant(aligned.iloc[:, 1])
ols = OLS(aligned.iloc[:, 0], X).fit()
residuals = ols.resid

# ADF test: if p < 0.05, residuals are stationary = spread is mean-reverting
adf_stat, adf_pvalue, *_ = adfuller(residuals)
print(f"ADF p-value: {adf_pvalue:.4f} (need < 0.05 for mean-reverting spread)")
print(f"Spread half-life: {-np.log(2) / np.log(abs(ols.params[1])):.1f} bars")
# Half-life < 20 bars (80h) means spread reverts within 3-4 days — tradeable
```

### Half-life sanity check

```python
# The half-life of mean reversion determines whether the spread reverts fast enough
# to be actionable in a 1-week competition window.
# Rule of thumb: half-life must be < 30 bars (~5 days) for a 7-day competition.
# Half-life estimation via AR(1):
spread = residuals
ar_coeff = np.corrcoef(spread[:-1], spread[1:])[0, 1]
half_life_bars = -np.log(2) / np.log(ar_coeff)
print(f"Half-life: {half_life_bars:.1f} 4H bars ({half_life_bars * 4 / 24:.1f} days)")
# If half-life > 30 bars, this pair is not suitable for a 7-day competition
```

### Z-score history plot

A quick visual sanity check before deploying:

```python
import matplotlib.pyplot as plt

spread_std   = float(residuals.std())
spread_mean  = float(residuals.mean())
zscore_series = (residuals - spread_mean) / spread_std

plt.figure(figsize=(12, 4))
plt.plot(zscore_series.values, label="Spread z-score")
plt.axhline( ENTRY_ZSCORE, color="green",  linestyle="--", label="Entry (+1.5)")
plt.axhline(-ENTRY_ZSCORE, color="green",  linestyle="--", label="Entry (-1.5)")
plt.axhline( EXIT_ZSCORE,  color="orange", linestyle=":",  label="Exit (+0.5)")
plt.axhline(-EXIT_ZSCORE,  color="orange", linestyle=":",  label="Exit (-0.5)")
plt.legend(); plt.title(f"ETH/BNB spread z-score"); plt.tight_layout()
plt.show()
# Visually confirm z-scores regularly cross ±1.5 and revert to 0 within the competition window
```

---

## Maximizing value

### Pre-screen pairs by half-life

Only deploy pairs whose half-life is ≤ 20 bars (~80 hours). Half-life > 30 bars means the
spread can stay wide for the entire competition week without reverting. A long position in the
laggard that never closes the gap is a directional trade dressed up as pairs trading.

### Hedge ratio stability check

Beta (the OLS coefficient) should be relatively stable over time. If rolling beta is swinging
from 0.5 to 2.0, the pair's structural relationship is breaking down. Compute rolling 30-day
beta and check for stability before entry:

```python
def rolling_beta(log_a: pd.Series, log_b: pd.Series, window: int = 84) -> pd.Series:
    cov = log_a.rolling(window).cov(log_b)
    var = log_b.rolling(window).var()
    return cov / (var + 1e-12)

beta_series = rolling_beta(log_a, log_b)
beta_stability = beta_series.std() / abs(beta_series.mean())  # coefficient of variation
if beta_stability > 0.3:
    logger.warning("Beta for %s/%s is unstable (CV=%.2f) — reduce confidence", pair_a, pair_b, beta_stability)
```

### Pair ranking: use half-life × spread_std as a combined score

Not all valid pairs are equally tradeable. Rank them:
- Low half-life = fast reversion = more opportunities in a short competition
- High spread_std = larger moves = bigger expected P&L per trade

```python
# Composite score: lower is better (fast reversion wins)
score = half_life / spread_std   # minimize this
```

Deploy pairs in order of score. If only running 2-3 pairs for simplicity, the highest-score
pairs are ETH/BNB and SOL/AVAX in most market conditions.

### Combine with regime filter

In BULL_TREND, long-only pairs trading aligns with momentum — buying the laggard is fine.
In BEAR_TREND, buying the laggard of a declining pair means buying into a falling market.
Apply the global regime filter:

```python
if regime_detector.get_regime() == RegimeState.BEAR_TREND:
    # Suppress all new pairs entries in bear market
    # (still exit existing positions normally)
    skip_new_entries = True
```

---

## Common pitfalls

### Pitfall 1: Long-only eliminates the hedge

Classical pairs trading is market-neutral because it shorts the outperformer at the same time
as buying the laggard. Long-only pairs trading is not market-neutral — if both coins drop
together (correlated sell-off), the laggard position loses money. This is the fundamental
limitation. Mitigate it by:
1. Applying the regime gate (no new entries in BEAR_TREND)
2. Keeping pairs position sizes small (≤ 25% of portfolio)
3. Using the STOP_ZSCORE emergency exit (if spread widens to 3.0 std, the pair is breaking down)

### Pitfall 2: Cointegration breaks during the competition

Pairs that were cointegrated for 90 days can decouple in days due to news events (one coin gets
an exchange listing, regulatory action, protocol hack). The periodic re-test catches this:
if a pair fails the cointegration test on re-test at bar 42, exit any open position and stop
generating signals for that pair.

### Pitfall 3: Using log prices vs. raw prices

The OLS regression must be run on **log prices**, not raw prices. If run on raw prices:
- Beta is in dollar terms (changes as prices change)
- The spread is not stationary even if the pair is cointegrated
- The z-score thresholds are meaningless

All spread calculations in the code above use `np.log(df["close"])`. Do not change this to raw prices.

### Pitfall 4: Stale spread statistics after refit

When the cointegration test re-runs at bar 42, the new beta and alpha will change the spread
definition. If there's an open position based on the old spread, the z-score instantly jumps
or drops. Close any open pairs position before refitting:

```python
# In update(), before re-testing:
if bars_since_test >= RETEST_INTERVAL and state.long_pair is not None:
    # Close open position before refitting — new beta changes the spread definition
    signals.append(TradingSignal(
        pair=state.long_pair,
        direction=SignalDirection.SELL,
        size=1.0,
        confidence=0.50,
    ))
    state.long_pair = None
    # Then proceed with refit
```

### Pitfall 5: Choosing pairs by recent correlation, not cointegration

Correlation and cointegration are different. Two coins can be highly correlated (both up 10%)
without being cointegrated (their spread has no tendency to mean-revert). The Engle-Granger test
checks for stationarity of the spread — correlation does not. Always use the coint() test, not
correlation as a proxy.

The intuition: BTC and ETH have ~0.95 correlation but when ETH breaks away from its BTC-implied
price for structural reasons (e.g., Ethereum upgrades), the spread may not revert for weeks.
Cointegration tests whether the spread has historically reverted; correlation tests whether they
move together in the same direction at the same time.

### Pitfall 6: Adding too many pairs

More pairs = more positions = more exposure = higher correlation to the overall market. With
39 coins, you could test 39×38/2 = 741 candidate pairs. Running more than 3-4 active pairs
simultaneously concentrates the portfolio in correlated altcoin longs, defeating the purpose
of diversification. Cap at 3 active pairs; run the rest as monitors only.

### Pitfall 7: The competition window may be too short for half-life

In a 7-day competition, even a pair with a 20-bar (80h) half-life may only generate 2-3 entry
opportunities. A pair with a 30-bar half-life is essentially untradeable in this window. Run
the pre-competition cointegration scan and half-life check before deploying this strategy —
if the best available pair has a half-life > 25 bars, skip pairs trading entirely and allocate
that capital to the momentum strategy.

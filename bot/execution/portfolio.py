"""
PortfolioAllocator: HRP + CVaR blended portfolio weights.

Two complementary approaches:
- HRP (Hierarchical Risk Parity): clusters assets by correlation, weights inversely
  proportional to cluster variance. No expected returns needed. Prevents BTC+ETH
  concentration when they're highly correlated.
- CVaR (Conditional VaR / Expected Shortfall): historical simulation at beta=0.95.
  No normality assumption — correct for fat-tailed crypto distributions.

Blends both at hrp_blend ratio (default 50/50). Falls back to equal 1/N weights
when PyPortfolioOpt is not installed or insufficient history (<60 bars).
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_HISTORY_BARS = 60


def _build_returns_df(price_history: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a simple-returns DataFrame from per-pair OHLCV DataFrames.

    Inner-joins on shared timestamps, forward-fills up to 1 bar to handle
    minor gaps, then drops any remaining NaN rows.

    Returns empty DataFrame if fewer than 2 pairs or fewer than _MIN_HISTORY_BARS rows.
    """
    close_series: Dict[str, pd.Series] = {}
    for pair, df in price_history.items():
        if df.empty or "close" not in df.columns:
            continue
        close_series[pair] = df["close"]

    if len(close_series) < 2:
        return pd.DataFrame()

    closes = pd.DataFrame(close_series).sort_index()
    closes = closes.ffill(limit=1).dropna()

    if len(closes) < _MIN_HISTORY_BARS:
        return pd.DataFrame()

    returns = closes.pct_change().dropna()
    return returns


def _equal_weights(pairs: list[str]) -> Dict[str, float]:
    n = len(pairs)
    return {p: 1.0 / n for p in pairs} if n > 0 else {}


def get_hrp_weights(price_history: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Compute HRP (Hierarchical Risk Parity) weights.

    Uses Ledoit-Wolf shrinkage covariance for more robust clustering.
    Falls back to equal weights on ImportError or any optimization failure.
    """
    pairs = list(price_history.keys())

    try:
        from pypfopt import HRPOpt, risk_models
    except ImportError:
        logger.warning("PyPortfolioOpt not installed — HRP falling back to equal weights")
        return _equal_weights(pairs)

    returns = _build_returns_df(price_history)
    if returns.empty:
        logger.info("HRP: insufficient history (%d pairs, need 60+ bars) — equal weights", len(pairs))
        return _equal_weights(pairs)

    try:
        cov = risk_models.CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()
        hrp = HRPOpt(returns=returns, cov_matrix=cov)
        raw = hrp.optimize()
        weights = {k: max(float(v), 0.0) for k, v in raw.items()}
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        logger.info("HRP weights: %s", {k: f"{v:.3f}" for k, v in weights.items()})
        return weights
    except Exception as exc:
        logger.warning("HRP optimization failed (%s) — equal weights", exc)
        return _equal_weights(pairs)


def get_cvar_weights(
    price_history: Dict[str, pd.DataFrame],
    beta: float = 0.95,
) -> Dict[str, float]:
    """
    Compute CVaR-minimizing (Expected Shortfall) weights.

    Historical simulation at `beta` confidence level — no normality assumption,
    correct for fat-tailed crypto return distributions.
    Falls back to equal weights on ImportError or any optimization failure.
    """
    pairs = list(price_history.keys())

    try:
        from pypfopt import EfficientCVaR
    except ImportError:
        logger.warning("PyPortfolioOpt not installed — CVaR falling back to equal weights")
        return _equal_weights(pairs)

    returns = _build_returns_df(price_history)
    if returns.empty:
        logger.info("CVaR: insufficient history — equal weights")
        return _equal_weights(pairs)

    try:
        mu = returns.mean()  # mean return per bar — Series indexed by pair
        ecvar = EfficientCVaR(mu, returns, beta=beta)
        ecvar.min_cvar()
        raw = ecvar.clean_weights()
        weights = {k: max(float(v), 0.0) for k, v in raw.items()}
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        logger.info("CVaR weights (beta=%.2f): %s", beta, {k: f"{v:.3f}" for k, v in weights.items()})
        return weights
    except Exception as exc:
        logger.warning("CVaR optimization failed (%s) — equal weights", exc)
        return _equal_weights(pairs)


def _blend_weights(
    hrp: Dict[str, float],
    cvar: Dict[str, float],
    hrp_blend: float = 0.5,
) -> Dict[str, float]:
    """Blend HRP and CVaR weights at hrp_blend ratio and renormalize."""
    all_keys = set(hrp) | set(cvar)
    blended = {
        k: hrp_blend * hrp.get(k, 0.0) + (1.0 - hrp_blend) * cvar.get(k, 0.0)
        for k in all_keys
    }
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended


class PortfolioAllocator:
    """
    Compute multi-asset portfolio weights using HRP + CVaR blend.

    Only runs on feature_pairs (BTC/ETH/SOL) — the pairs with full 4H history
    in LiveFetcher._buffers. All other tradeable pairs inherit equal fallback weights.

    Call compute_weights() once per 4H boundary (at the same time signals are
    generated), then call get_pair_weight() per trade to read the cached weight.

    Falls back gracefully to equal 1/N weights when:
    - PyPortfolioOpt is not installed
    - Fewer than 60 bars of shared history
    - Any optimization error
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._weights: Dict[str, float] = {}

    def compute_weights(self, price_history: Dict[str, pd.DataFrame]) -> None:
        """
        Compute and cache blended HRP + CVaR weights from price history.

        Args:
            price_history: Dict of pair → OHLCV DataFrame (from live_fetcher._to_dataframe()).
                           Typically feature_pairs only (BTC/USD, ETH/USD, SOL/USD).
        """
        pairs = list(price_history.keys())
        if not pairs:
            return

        beta = self._config.get("cvar_beta", 0.95)
        hrp_blend = self._config.get("hrp_blend", 0.5)

        hrp = get_hrp_weights(price_history)
        cvar = get_cvar_weights(price_history, beta=beta)
        self._weights = _blend_weights(hrp, cvar, hrp_blend=hrp_blend)

        logger.info(
            "PortfolioAllocator: final blended weights = %s",
            {k: f"{v:.3f}" for k, v in self._weights.items()},
        )

    def get_pair_weight(self, pair: str, n_active_pairs: int = 1) -> float:
        """
        Return portfolio weight for `pair`.

        Falls back to 1/n_active_pairs when weights have not been computed yet
        (e.g., insufficient history on first 4H boundary).

        Args:
            pair: Roostoo pair symbol (e.g. "BTC/USD").
            n_active_pairs: Number of actively traded pairs (for equal-weight fallback).

        Returns:
            Weight in [0.0, 1.0].
        """
        if self._weights:
            return self._weights.get(pair, 1.0 / max(n_active_pairs, 1))
        return 1.0 / max(n_active_pairs, 1)

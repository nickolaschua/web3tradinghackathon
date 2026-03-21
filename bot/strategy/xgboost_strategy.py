"""
XGBoost-based BTC/USD momentum strategy for live trading.

Wraps xgb_btc_15m_iter5.pkl (19 features, trained on 15M bars) and generates
BUY signals when P(BUY) >= threshold. Exits are handled entirely by the
RiskManager ATR trailing-stop (no SELL signal from the model — consistent with
how the backtest uses the model).

Only produces signals for BTC/USD. Returns HOLD for all other pairs so that
the MeanReversionStrategy fallback can trade ETH/SOL.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from bot.strategy.base import BaseStrategy, SignalDirection, TradingSignal

logger = logging.getLogger(__name__)


class XGBoostStrategy(BaseStrategy):
    """
    XGBoost classifier strategy using predict_proba().

    Generates BUY signals when P(BUY) >= threshold. Returns HOLD otherwise.
    No SELL signal — exits are handled by the ATR trailing-stop in RiskManager,
    matching the backtest's exit logic.

    Args:
        model_path: Path to the pickled XGBoostClassifier (.pkl).
        threshold:  Minimum P(BUY) to generate a BUY signal (default 0.65).
        pair:       The single pair this strategy trades (default "BTC/USD").
                    All other pairs receive HOLD.
    """

    def __init__(
        self,
        model_path: str = "models/xgb_btc_15m_iter5.pkl",
        threshold: float = 0.65,
        pair: str = "BTC/USD",
        exit_threshold: float = 0.10,
    ) -> None:
        self._threshold = threshold
        self._exit_threshold = exit_threshold
        self._pair = pair

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"XGBoostStrategy: model not found at {model_path}. "
                f"Run: python scripts/train_model_15m.py --output {model_path}"
            )

        with open(path, "rb") as f:
            self._model = pickle.load(f)

        self._feature_cols: list[str] = list(self._model.feature_names_in_)
        logger.info(
            "XGBoostStrategy: loaded %s | threshold=%.2f | features=%d (%s...)",
            model_path,
            threshold,
            len(self._feature_cols),
            self._feature_cols[:3],
        )

    def generate_signal(self, pair: str, features: pd.DataFrame) -> TradingSignal:
        """
        Generate a BUY or HOLD signal from the latest feature row.

        Returns HOLD immediately for any pair that is not self._pair.
        Uses only features.iloc[-1] — no look-back within this call.

        Args:
            pair:     Tradeable pair, e.g. "BTC/USD".
            features: Full feature history DataFrame (already shifted 1 bar).

        Returns:
            TradingSignal with direction BUY (confidence=P(BUY)) or HOLD.
        """
        if pair != self._pair:
            return TradingSignal(pair=pair)

        if features.empty:
            return TradingSignal(pair=pair)

        # Build a single-row DataFrame with exactly the columns the model expects.
        # XGBoost handles NaN natively (uses learned missing-value direction per split),
        # so missing feature columns are filled with NaN rather than raising.
        last = features.iloc[[-1]].copy()
        missing = [c for c in self._feature_cols if c not in last.columns]
        if missing:
            logger.warning(
                "XGBoostStrategy: %d features missing from feature matrix for %s: %s",
                len(missing), pair, missing,
            )
            for col in missing:
                last[col] = float("nan")

        row = last[self._feature_cols]

        try:
            proba = float(self._model.predict_proba(row)[0, 1])
        except Exception as exc:
            logger.error("XGBoostStrategy: predict_proba failed for %s: %s", pair, exc)
            return TradingSignal(pair=pair)

        logger.debug(
            "XGBoostStrategy: %s P(BUY)=%.4f threshold=%.2f exit_threshold=%.2f",
            pair, proba, self._threshold, self._exit_threshold,
        )

        if proba >= self._threshold:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.BUY,
                size=1.0,
                confidence=round(min(proba, 1.0), 4),
            )

        if proba <= self._exit_threshold:
            return TradingSignal(
                pair=pair,
                direction=SignalDirection.SELL,
                confidence=round(min(proba, 1.0), 4),
            )

        return TradingSignal(pair=pair)  # HOLD


__all__ = ["XGBoostStrategy"]

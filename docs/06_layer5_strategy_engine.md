# Layer 5 — Strategy Engine

## What This Layer Does

The Strategy Engine is where validated signals get turned into trading decisions. It takes the feature matrix from Layer 3, applies your strategy logic, and outputs a structured action: BUY, SELL, or HOLD, with the target pair and suggested position size.

This layer contains two strategies (momentum and mean reversion) behind a shared abstract interface, and a regime detection module that selects which strategy is active at any given time. The regime module also adjusts the position sizing multiplier so that exposure automatically scales with market conditions.

**This layer is deployed on EC2.** It runs on every candle boundary in the main loop.

---

## What This Layer Is Trying to Achieve

1. Translate the feature matrix into a single, unambiguous trading action per cycle
2. Ensure the active strategy always matches the current market regime — momentum in trending markets, mean reversion in sideways markets, cash in bear markets
3. Keep strategy logic simple enough to debug and modify quickly during the competition
4. Allow hot-swapping of strategy parameters via `config.yaml` without redeploying code

---

## How It Contributes to the Bigger Picture

This layer is the intellectual core of the system. Every other layer exists to support this one: clean data so features are accurate, validated parameters so the logic is trustworthy, risk management to enforce exits, and execution to submit the orders.

The critical constraint: strategy logic must be deterministic and stateless across calls. Given the same feature matrix and config, it should always produce the same output. This makes it testable, debuggable, and crash-safe (the state layer handles position tracking, not this layer).

---

## Files in This Layer

```
strategy/
├── base.py             Abstract strategy interface
├── momentum.py         Primary: multi-timeframe trend following
└── mean_reversion.py   Backup: Bollinger Band mean reversion

execution/
└── regime.py           Market regime detection and strategy selection
```

---

## `execution/regime.py`

Regime detection answers the question: what kind of market are we in right now, and what is the appropriate posture?

The three regimes and their implications:

| Regime | Signal | Position Multiplier | Active Strategy |
|---|---|---|---|
| BULL_TREND | BTC EMA20 > EMA50 AND ADX > 25 | 1.0× (full size) | momentum.py |
| SIDEWAYS | ADX < 20 | 0.5× (half size) | mean_reversion.py |
| BEAR_TREND | BTC EMA20 < EMA50 AND ADX > 20 | 0.0× (no new positions) | none (cash) |

```python
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL_TREND = "BULL_TREND"
    SIDEWAYS = "SIDEWAYS"
    BEAR_TREND = "BEAR_TREND"

@dataclass
class RegimeState:
    regime: MarketRegime
    size_multiplier: float
    ema20: float
    ema50: float
    adx: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"

SIZE_MULTIPLIERS = {
    MarketRegime.BULL_TREND: 1.0,
    MarketRegime.SIDEWAYS:   0.5,
    MarketRegime.BEAR_TREND: 0.0,
}

class RegimeDetector:
    """
    Detects market regime from BTC daily data.
    Always use BTC as the regime reference — it leads the entire market.
    """

    ADX_TREND_THRESHOLD = 25.0
    ADX_RANGE_THRESHOLD = 20.0
    # Hysteresis: require N consecutive signals before switching regime
    # Prevents thrashing at regime boundaries
    CONFIRMATION_BARS = 2

    def __init__(self):
        self._pending_regime = None
        self._pending_count = 0
        self._current_regime = MarketRegime.SIDEWAYS  # Conservative default

    def detect(self, btc_daily_features: pd.DataFrame) -> RegimeState:
        """
        Detect regime from the last row of BTC daily features.
        Uses hysteresis to prevent rapid regime switching.
        """
        latest = btc_daily_features.iloc[-1]

        ema20 = latest.get("ema_20", np.nan)
        ema50 = latest.get("ema_50", np.nan)
        adx = latest.get("adx_14", np.nan)

        if any(pd.isna(x) for x in [ema20, ema50, adx]):
            logger.warning("Regime detection: NaN in features, holding current regime")
            return self._build_state(self._current_regime, ema20, ema50, adx, "LOW")

        # Classify raw signal
        if ema20 > ema50 and adx > self.ADX_TREND_THRESHOLD:
            raw_regime = MarketRegime.BULL_TREND
        elif ema20 < ema50 and adx > self.ADX_RANGE_THRESHOLD:
            raw_regime = MarketRegime.BEAR_TREND
        else:
            raw_regime = MarketRegime.SIDEWAYS

        # Hysteresis: only switch after N consecutive confirmations
        if raw_regime == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = raw_regime
            self._pending_count = 1

        if self._pending_count >= self.CONFIRMATION_BARS:
            if raw_regime != self._current_regime:
                logger.warning(f"REGIME CHANGE: {self._current_regime.value} → {raw_regime.value} "
                               f"| EMA20={ema20:.0f} EMA50={ema50:.0f} ADX={adx:.1f}")
                self._current_regime = raw_regime

        confidence = "HIGH" if self._pending_count >= self.CONFIRMATION_BARS else "MEDIUM"
        return self._build_state(self._current_regime, ema20, ema50, adx, confidence)

    def _build_state(self, regime, ema20, ema50, adx, confidence) -> RegimeState:
        return RegimeState(
            regime=regime,
            size_multiplier=SIZE_MULTIPLIERS[regime],
            ema20=float(ema20) if not pd.isna(ema20) else 0,
            ema50=float(ema50) if not pd.isna(ema50) else 0,
            adx=float(adx) if not pd.isna(adx) else 0,
            confidence=confidence,
        )
```

---

## `strategy/base.py`

The abstract interface every strategy must implement. This allows the main loop to call `strategy.generate_signal()` without knowing which strategy is active.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd

class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    action: SignalAction
    pair: str
    confidence: float        # 0.0 to 1.0
    suggested_size_pct: float  # Fraction of available capital (before regime multiplier)
    reason: str              # Human-readable description for logging
    stop_price: Optional[float] = None  # Suggested initial stop level

class BaseStrategy(ABC):
    """
    Abstract base strategy. All concrete strategies must implement generate_signal().
    
    IMPORTANT: Strategies must be stateless with respect to positions.
    Position tracking lives in the OMS (Layer 7), not here.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def generate_signal(self, features: pd.DataFrame,
                        current_price: float,
                        current_position: float) -> TradingSignal:
        """
        Given the current feature matrix, current price, and current position size,
        return a trading signal.
        
        Args:
            features: Output of features.py for this pair, latest row contains current state
            current_price: Latest price from ticker
            current_position: Current position size in USD (0 if not in position)
        
        Returns:
            TradingSignal with action, confidence, and sizing guidance
        """
        pass

    @abstractmethod
    def get_regime_compatibility(self) -> list[str]:
        """
        Returns list of regime names this strategy is designed for.
        Used by regime detection to select the appropriate strategy.
        """
        pass
```

---

## `strategy/momentum.py`

The primary strategy. Uses a composite momentum score derived from the validated IC analysis to generate directional signals on 4H candles.

```python
import pandas as pd
import numpy as np
import logging
from .base import BaseStrategy, TradingSignal, SignalAction

logger = logging.getLogger(__name__)

class MultiTimeframeMomentumStrategy(BaseStrategy):
    """
    Multi-timeframe momentum strategy.
    
    Signal logic:
    - MACD histogram direction (validated as primary IC feature)
    - RSI threshold filter (prevent buying overbought)
    - EMA trend confirmation (direction alignment)
    - Volume confirmation (volume z-score > 0 adds conviction)
    
    Entry: composite score above threshold
    Exit: MACD reversal OR RSI overbought OR trailing stop (in risk layer)
    """

    def get_regime_compatibility(self) -> list[str]:
        return ["BULL_TREND"]

    def generate_signal(self, features: pd.DataFrame,
                        current_price: float,
                        current_position: float) -> TradingSignal:

        if len(features) < 2:
            return TradingSignal(SignalAction.HOLD, "", 0, 0,
                                  "Insufficient feature history")

        latest = features.iloc[-1]
        prev = features.iloc[-2]

        # ── Extract validated features ─────────────────────────────────────
        macd_hist = latest.get("macd_hist", np.nan)
        macd_hist_prev = prev.get("macd_hist", np.nan)
        rsi = latest.get("rsi_14", np.nan)
        ema_spread = latest.get("ema_20_50_spread", np.nan)
        adx = latest.get("adx_14", np.nan)
        vol_z = latest.get("volume_zscore", np.nan)
        atr = latest.get("atr_14", np.nan)

        if any(pd.isna(x) for x in [macd_hist, rsi, ema_spread, adx]):
            return TradingSignal(SignalAction.HOLD, "", 0, 0, "NaN in features")

        # ── Entry signal ───────────────────────────────────────────────────
        macd_turning_up = macd_hist > 0 and macd_hist_prev <= 0  # MACD crossing zero up
        macd_positive = macd_hist > 0
        rsi_not_overbought = rsi < self.config.get("rsi_entry_max", 65)
        rsi_not_oversold_crash = rsi > self.config.get("rsi_entry_min", 25)
        trend_aligned = ema_spread > 0  # Fast EMA above slow EMA
        adx_trending = adx > self.config.get("adx_min", 20)
        volume_confirming = vol_z > -0.5  # Not on unusually low volume

        # Composite score: each condition adds to conviction
        entry_score = sum([
            macd_positive * 2,          # Core signal, weighted higher
            macd_turning_up * 1,        # Bonus for fresh crossover
            rsi_not_overbought * 1,
            trend_aligned * 1,
            adx_trending * 1,
            volume_confirming * 0.5,
        ])
        entry_threshold = self.config.get("entry_score_threshold", 4.0)

        # ── Exit signal ────────────────────────────────────────────────────
        macd_turning_down = macd_hist < 0 and macd_hist_prev >= 0
        rsi_overbought = rsi > self.config.get("rsi_exit", 72)
        trend_broken = ema_spread < -0.005  # Fast EMA clearly below slow

        # ── Generate action ────────────────────────────────────────────────
        if current_position > 0:
            # We're in a position — check for exit
            if macd_turning_down or rsi_overbought or trend_broken:
                reason = (f"EXIT: macd_down={macd_turning_down}, "
                          f"rsi_ob={rsi_overbought}, trend_broken={trend_broken}")
                logger.info(reason)
                return TradingSignal(
                    action=SignalAction.SELL,
                    pair="",  # Caller fills in the pair
                    confidence=0.9,
                    suggested_size_pct=1.0,  # Exit full position
                    reason=reason,
                )
        else:
            # Not in position — check for entry
            if entry_score >= entry_threshold and rsi_not_oversold_crash:
                confidence = min(entry_score / (entry_threshold * 1.5), 1.0)
                # Suggest initial stop: 2× ATR below entry
                stop = current_price - 2.0 * atr if not pd.isna(atr) else None
                reason = f"ENTRY: score={entry_score:.1f}/{entry_threshold}"
                logger.info(reason)
                return TradingSignal(
                    action=SignalAction.BUY,
                    pair="",
                    confidence=confidence,
                    suggested_size_pct=self.config.get("base_position_size", 0.25),
                    reason=reason,
                    stop_price=stop,
                )

        return TradingSignal(SignalAction.HOLD, "", 0, 0,
                             f"HOLD: score={entry_score:.1f}/{entry_threshold}")
```

---

## `strategy/mean_reversion.py`

The backup strategy, activated when ADX < 20 indicates a range-bound market. Simpler logic, smaller position sizes, and tighter stops — mean reversion goes wrong fast if the range breaks.

```python
import pandas as pd
import numpy as np
import logging
from .base import BaseStrategy, TradingSignal, SignalAction

logger = logging.getLogger(__name__)

class BollingerMeanReversionStrategy(BaseStrategy):
    """
    Mean reversion on Bollinger Bands.
    Only active in SIDEWAYS regime (ADX < 20).
    
    Entry: price at lower band + RSI oversold confirmation
    Exit:  price returns to middle band OR RSI overbought
    """

    def get_regime_compatibility(self) -> list[str]:
        return ["SIDEWAYS"]

    def generate_signal(self, features: pd.DataFrame,
                        current_price: float,
                        current_position: float) -> TradingSignal:

        if len(features) < 2:
            return TradingSignal(SignalAction.HOLD, "", 0, 0, "Insufficient history")

        latest = features.iloc[-1]

        bb_pct_b = latest.get("bb_pct_b", np.nan)
        rsi = latest.get("rsi_14", np.nan)
        adx = latest.get("adx_14", np.nan)

        if any(pd.isna(x) for x in [bb_pct_b, rsi]):
            return TradingSignal(SignalAction.HOLD, "", 0, 0, "NaN in features")

        # Safety check: if ADX crept above threshold, suppress entries
        if not pd.isna(adx) and adx > self.config.get("adx_max", 22):
            return TradingSignal(SignalAction.HOLD, "", 0, 0,
                                  f"ADX too high for mean reversion: {adx:.1f}")

        bb_lower_threshold = self.config.get("bb_lower_threshold", 0.1)  # Near lower band
        bb_middle_threshold = self.config.get("bb_middle_threshold", 0.5)  # Near middle
        rsi_oversold = self.config.get("rsi_oversold", 35)
        rsi_exit = self.config.get("rsi_mr_exit", 60)

        if current_position > 0:
            # Exit when price returns to middle band or RSI normalises
            if bb_pct_b > bb_middle_threshold or rsi > rsi_exit:
                reason = f"MR EXIT: bb_pct_b={bb_pct_b:.2f}, rsi={rsi:.1f}"
                return TradingSignal(SignalAction.SELL, "", 0.85, 1.0, reason)
        else:
            # Entry at lower band with RSI confirmation
            if bb_pct_b < bb_lower_threshold and rsi < rsi_oversold:
                reason = f"MR ENTRY: bb_pct_b={bb_pct_b:.2f}, rsi={rsi:.1f}"
                return TradingSignal(
                    action=SignalAction.BUY,
                    pair="",
                    confidence=0.7,
                    # Smaller size for mean reversion — lower conviction
                    suggested_size_pct=self.config.get("mr_position_size", 0.15),
                    reason=reason,
                )

        return TradingSignal(SignalAction.HOLD, "", 0, 0,
                             f"MR HOLD: bb={bb_pct_b:.2f}, rsi={rsi:.1f}")
```

---

## Strategy Selection in the Main Loop

```python
# In main.py, simplified

# Load strategies once on startup
momentum_strategy = MultiTimeframeMomentumStrategy(config["momentum"])
mr_strategy = BollingerMeanReversionStrategy(config["mean_reversion"])
regime_detector = RegimeDetector()

def select_strategy(regime_state: RegimeState) -> BaseStrategy | None:
    if regime_state.regime == MarketRegime.BULL_TREND:
        return momentum_strategy
    elif regime_state.regime == MarketRegime.SIDEWAYS:
        return mr_strategy
    else:  # BEAR_TREND
        return None  # No strategy = cash position

# Per-cycle call
regime_state = regime_detector.detect(btc_daily_features)
active_strategy = select_strategy(regime_state)

if active_strategy is None:
    # Regime is BEAR — close any open positions and hold cash
    pass
else:
    for pair in active_pairs:
        signal = active_strategy.generate_signal(
            features=feature_matrices[pair],
            current_price=current_prices[pair],
            current_position=oms.get_position_value(pair),
        )
        # Signal passes to risk layer next
```

---

## `config.yaml` — Hot-Swappable Strategy Parameters

All strategy thresholds live in config.yaml. During the competition you can modify them and restart the bot in under 30 seconds without touching code.

```yaml
momentum:
  rsi_entry_max: 65
  rsi_entry_min: 25
  rsi_exit: 72
  adx_min: 20
  entry_score_threshold: 4.0
  base_position_size: 0.25   # 25% of available capital per entry

mean_reversion:
  bb_lower_threshold: 0.10
  bb_middle_threshold: 0.50
  rsi_oversold: 35
  rsi_mr_exit: 60
  adx_max: 22
  mr_position_size: 0.15

regime:
  adx_trend_threshold: 25.0
  adx_range_threshold: 20.0
  confirmation_bars: 2

risk:
  atr_stop_multiplier: 2.0
  hard_stop_pct: 0.08
  max_positions: 3
  max_single_position_pct: 0.40
  circuit_breaker_drawdown: 0.30

pairs:
  primary: ["BTC/USD", "ETH/USD", "SOL/USD"]
```

---

## Failure Modes This Layer Prevents

| Failure | Prevention |
|---|---|
| Long-only in bear regime | BEAR_TREND → 0.0× multiplier → no new positions, existing positions still managed by risk layer |
| Strategy not matching market regime | Regime detector gates which strategy is active |
| Strategy thrashing at regime boundaries | Hysteresis (CONFIRMATION_BARS) prevents rapid switching |
| Overconfident entries in ranging market | ADX minimum threshold gates momentum entries |
| Strategies diverging between environments | Stateless design; same inputs always produce same output |
| Hard-coded thresholds that can't be tuned | All thresholds in config.yaml, hot-swappable without redeploy |

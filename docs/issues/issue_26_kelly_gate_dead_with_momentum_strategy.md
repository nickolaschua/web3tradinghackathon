# Issue 26: Kelly Gate Never Fires with MomentumStrategy — Dead Code Path

## Layer
Layer 6 — Risk Management

## Description
`RiskManager.size_new_position()` implements a Half-Kelly gate:

```python
kelly = (p * b - (1.0 - p)) / b
if kelly <= 0:
    return SizingResult(BLOCKED_NEGATIVE_KELLY, ...)
```

With `b = expected_win_loss_ratio = 1.5` (from config) and `MomentumStrategy` returning
`confidence ∈ [0.5, 0.9]`:

- Minimum Kelly: `(0.5 × 1.5 - 0.5) / 1.5 = 0.167 > 0`

The gate **never fires**. Any trade with confidence ≥ 0.5 passes. The gate only becomes
meaningful when XGBoost is wired and can return low-probability predictions (confidence < 0.40).

With MomentumStrategy, `signal.confidence` is computed as:
```python
confidence = 0.5 + 0.4 * (raw_score / max_possible)  # always >= 0.5
```

## Impact
**Low** — Gate is correctly implemented and will become active once XGBoost predictions
replace MomentumStrategy signals. No incorrect behavior today; just dead protection.

## Fix Required
No immediate fix needed. When XGBoost is wired:
- XGBoost `predict_proba()` returns probabilities in [0, 1]
- A prediction of 0.35 → Kelly = (0.35 × 1.5 - 0.65) / 1.5 = -0.083 → BLOCKED
- This is the intended behavior — low-confidence ML signals are gated out

Document in code that this gate is intentionally dormant until XGBoost is wired.

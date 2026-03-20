# Issue 28: PyPortfolioOpt Not Installed Silently Falls Back to Equal Weights

## Layer
Layer 6 — Risk Management / PortfolioAllocator

## Description
`PortfolioAllocator.compute_weights()` catches `ImportError` from `pypfopt` and silently
falls back to equal 1/N weights:

```python
try:
    from pypfopt import HRPOpt, risk_models
except ImportError:
    logger.warning("PyPortfolioOpt not installed — HRP falling back to equal weights")
    return _equal_weights(pairs)
```

The warning is logged at `WARNING` level, but:
1. In a production deployment where logging is set to `INFO`, this warning appears but
   may be buried in startup noise and never noticed.
2. There is no startup check that asserts `pypfopt` is importable.
3. `requirements.txt` includes `PyPortfolioOpt>=1.5`, but if the deployment environment
   installs packages from a cache or constraints file that omits it, the bot silently
   degrades without any runtime error.

Test output confirmed this: the test suite ran with `PyPortfolioOpt not installed` messages
but all portfolio tests still passed because the fallback is correctly implemented.

## Impact
**Low** — Equal weight fallback is safe and correct behavior. The risk is silent
capability loss: you think you have HRP/CVaR optimization but you have equal weights.

## Fix Required
Add an explicit startup check in `main.py` (or a `__init__` health check):

```python
try:
    import pypfopt
    logger.info(f"PyPortfolioOpt {pypfopt.__version__} loaded — HRP/CVaR active")
except ImportError:
    logger.warning("PyPortfolioOpt not installed — portfolio optimization DISABLED, using equal weights")
```

This makes the degradation visible at bot startup rather than buried in cycle logs.

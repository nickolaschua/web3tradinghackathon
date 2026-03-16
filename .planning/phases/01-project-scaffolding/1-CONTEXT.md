# Phase 1: Project Scaffolding - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<vision>
## How This Should Work

Phase 1 is pure structure — no logic, no API calls, no data fetching. Create every directory and `__init__.py` file so the `bot.*` namespace is importable from day one. Drop in `requirements.txt`, `config.yaml`, `.env.example`, and `.gitignore` so every other agent can start immediately without waiting on anything.

Think of it as laying the foundation before any walls go up: when this phase is done, you can run `from bot.api.client import RoostooClient` and get an ImportError about the missing class — not a ModuleNotFoundError about the package. That's the signal it's done.

</vision>

<essential>
## What Must Be Nailed

All three are equally important — none can be skipped:

- **Correct `bot.*` namespace** — all subdirs have `__init__.py`; imports like `from bot.api.client import RoostooClient` resolve to the right path without any sys.path hacks
- **Dependency pinning** — `requirements.txt` uses `pandas-ta-classic` (NOT `pandas-ta`), which is the actively-maintained fork that works on Python 3.11 + pandas 2.x; original `pandas-ta` is broken on this stack
- **Three API key sets in `.env.example`** — testing keys (`ROOSTOO_API_KEY_TEST`, `ROOSTOO_SECRET_TEST`), Round 1 competition keys (`ROOSTOO_API_KEY`, `ROOSTOO_SECRET`), and a commented placeholder for finalist keys; mislabelling these during competition = disqualification

</essential>

<boundaries>
## What's Out of Scope

- **No logic** — all `__init__.py` files are empty or bare minimum; no classes, no functions, no imports between packages
- **No tests** — `tests/__init__.py` is an empty skeleton; no actual test code until later phases
- **No API calls** — no signing, no HTTP, no data fetching
- **No Dockerfile** — that belongs to Phase 8 (EC2 Deployment)

</boundaries>

<specifics>
## Specific Ideas

- `config.yaml` should be **fully populated** with all tunable parameters from PROJECT.md spec: `tradeable_pairs: ["BTC/USD"]`, `feature_pairs: ["BTC/USD", "ETH/USD", "SOL/USD"]`, `candle_interval: "4h"`, `max_positions: 1`, `hard_stop_pct`, `atr_stop_multiplier`, `circuit_breaker_drawdown` thresholds (30%/20%/10%), `trade_cooldown_seconds: 65`, `regime.*` parameters — not placeholders, real values
- `.gitignore` must cover: `.env`, `__pycache__/`, `*.pyc`, `state.json`, `state.json.bak`, `state.json.tmp`, `logs/`, `data/parquet/` — secrets and large data must not land in git

</specifics>

<notes>
## Additional Context

- EC2 is Python 3.11 on ap-southeast-2 (Sydney) — `pandas-ta-classic` requirement is non-negotiable for this Python version
- The `bot.*` namespace mandate comes from FAQ Q37 best practices + orchestration.md; bare `from api.client import ...` imports will break on EC2 where the working directory may differ
- Round 1 starts Mar 21 8PM; first trade must execute before Mar 22 8PM — no time for import debugging during competition
- The three-key-set `.env.example` is critical: test keys ≠ competition keys ≠ finalist keys; using wrong keys = wasted trades or disqualification

</notes>

---

*Phase: 01-project-scaffolding*
*Context gathered: 2026-03-16*

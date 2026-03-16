# Phase 1: Project Scaffolding - Research

**Researched:** 2026-03-16
**Domain:** Python project packaging — `bot.*` namespace, `pandas-ta-classic`, systemd-compatible layout
**Confidence:** HIGH

<research_summary>
## Summary

Phase 1 is standard Python packaging with one non-commodity item: `pandas-ta-classic`. The rest (directory layout, `__init__.py`, `config.yaml`, `.env.example`, `.gitignore`) is commodity work fully specified in PROJECT.md and CONTEXT.md — no research gaps.

The only thing worth verifying was `pandas-ta-classic`: correct PyPI name, import name, API compatibility, and Python 3.11 + pandas 2.x support. Research confirms it is a healthy, actively-maintained fork (v0.3.78, released Feb 2026) with an identical DataFrame accessor API (`df.ta.rsi()`, `df.ta.macd()`, etc.).

The `bot.*` namespace pattern (empty `__init__.py` files + `WorkingDirectory` in systemd unit) works without `setup.py`, `pyproject.toml`, or `sys.path` manipulation.

**Primary recommendation:** Use `pandas-ta-classic` (PyPI) / `pandas_ta_classic` (import), pin version `>=0.3.78`, confirm `.ta` accessor registers correctly. No `pyproject.toml` needed for this deployment pattern.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas-ta-classic | >=0.3.78 | Technical indicators via `df.ta.*` | Actively-maintained fork of pandas-ta; works on Python 3.11 + pandas 2.x |
| pandas | >=2.0 | DataFrames | Required by pandas-ta-classic |
| numpy | >=2.0 | Numerical ops | Transitively required |
| python-dotenv | latest | Load `.env` into `os.environ` | Standard pattern for config from env files |
| pyyaml | latest | Parse `config.yaml` | Standard YAML loader |
| requests | latest | HTTP for Roostoo API | Simple, stable REST client |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cryptography / hmac | stdlib | HMAC SHA256 signing | Built-in, no extra dep needed |
| logging | stdlib | Bot and trade logs | Rotating file handler built-in |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pandas-ta-classic | pandas-ta | Original is unmaintained, broken on Python 3.11 + pandas 2.x — do NOT use |
| pyyaml | tomllib / json | YAML is more readable for config; already in PROJECT.md spec |
| python-dotenv | os.environ directly | dotenv gives .env file support; makes EC2 deployment cleaner |

**Installation:**
```bash
pip install pandas-ta-classic pandas python-dotenv pyyaml requests
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```
web3tradinghackathon/        ← project root (WorkingDirectory for systemd)
├── main.py                  ← entry point; uses `from bot.api.client import ...`
├── requirements.txt
├── .env                     ← NOT committed; loaded by python-dotenv
├── .env.example             ← committed; documents key sets
├── state.json               ← runtime artifact; gitignored
├── logs/                    ← runtime artifact; gitignored
├── data/parquet/            ← seed data; gitignored
├── bot/
│   ├── __init__.py          ← empty
│   ├── api/
│   │   ├── __init__.py      ← empty
│   │   ├── client.py
│   │   └── rate_limiter.py
│   ├── config/
│   │   ├── __init__.py      ← empty
│   │   └── config.yaml
│   ├── data/
│   │   ├── __init__.py      ← empty
│   │   └── fetcher.py
│   ├── execution/
│   │   ├── __init__.py      ← empty
│   │   ├── regime.py
│   │   ├── risk.py
│   │   └── order_manager.py
│   ├── monitoring/
│   │   ├── __init__.py      ← empty
│   │   └── telegram.py
│   ├── persistence/
│   │   ├── __init__.py      ← empty
│   │   └── state_manager.py
│   └── strategy/
│       ├── __init__.py      ← empty
│       ├── base.py
│       ├── momentum.py
│       └── mean_reversion.py
└── tests/
    └── __init__.py          ← empty skeleton
```

### Pattern 1: `bot.*` Namespace Without `sys.path` Hacks
**What:** Empty `__init__.py` in every `bot/` subdirectory. Run from project root.
**When to use:** Always — this is the only supported pattern per PROJECT.md + orchestration spec.
**Example:**
```python
# main.py — works when WorkingDirectory = project root
from bot.api.client import RoostooClient
from bot.execution.risk import RiskManager
from bot.monitoring.telegram import TelegramAlerter
```
**Why it works:** Python adds the script's directory (`main.py` location = project root) to `sys.path[0]` automatically. With `WorkingDirectory=` set in the systemd unit, this is guaranteed on EC2 too. No `PYTHONPATH` or `sys.path.insert()` needed.

### Pattern 2: Systemd `WorkingDirectory` for Path Resolution
**What:** Set `WorkingDirectory=/home/ubuntu/web3tradinghackathon/web3tradinghackathon` in the service unit.
**When to use:** EC2 systemd deployment.
**Example:**
```ini
[Service]
WorkingDirectory=/home/ubuntu/web3tradinghackathon/web3tradinghackathon
ExecStart=/usr/bin/python3.11 main.py
Environment="PYTHONUNBUFFERED=1"
```

### Pattern 3: `pandas-ta-classic` Usage
**What:** Import the package; it registers `df.ta` accessor automatically — same as original `pandas-ta`.
**Example:**
```python
import pandas_ta_classic  # registers df.ta accessor

df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.ema(length=20, append=True)
```
**Key point:** The PyPI name uses hyphens (`pandas-ta-classic`), the import uses underscores (`pandas_ta_classic`), but the accessor is still `.ta` — identical usage to original.

### Anti-Patterns to Avoid
- **`import pandas_ta`** — installs original broken package; breaks on Python 3.11 + pandas 2.x
- **`sys.path.insert(0, ...)`** in `main.py` — unnecessary and fragile; use `WorkingDirectory` instead
- **Relative imports** (`from .api.client import ...`) in `main.py` — `main.py` is a script, not a package; relative imports will fail
- **Non-empty `__init__.py`** in Phase 1 — no logic, no imports between packages until later phases
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Technical indicators (RSI, MACD, EMA) | Custom rolling calculations | `pandas-ta-classic` `df.ta.*` | Battle-tested, pandas 2.x compatible, identical API to original |
| YAML config loading | Custom parser | `pyyaml` `yaml.safe_load()` | stdlib-adjacent, universal |
| `.env` file loading | Manual `open()` parsing | `python-dotenv` `load_dotenv()` | Handles edge cases, comments, quoting |
| HMAC signing | Third-party library | `hmac` + `hashlib` (stdlib) | No extra dep; standard for HMAC-SHA256 |

**Key insight:** Phase 1 has no complex problems — it's pure structure. The only "don't hand-roll" item is indicators: use `pandas-ta-classic`, not manual rolling computations.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Wrong `pandas-ta` Package
**What goes wrong:** `pip install pandas-ta` installs the original unmaintained package, which raises `AttributeError` or import errors on Python 3.11 + pandas 2.x.
**Why it happens:** The PyPI name is similar; easy to type the wrong one.
**How to avoid:** `requirements.txt` must say `pandas-ta-classic`, not `pandas-ta`. Pin with `pandas-ta-classic>=0.3.78`.
**Warning signs:** `ImportError`, `AttributeError: 'DataFrame' object has no attribute 'ta'`, or `TypeError` during indicator calls on first import.

### Pitfall 2: Missing `__init__.py` in a Subdirectory
**What goes wrong:** `from bot.execution.risk import RiskManager` raises `ModuleNotFoundError` even though the file exists.
**Why it happens:** Python 3 has "namespace packages" (directories without `__init__.py`) but they behave differently and can cause confusing import errors with some tools.
**How to avoid:** Every `bot/` subdirectory must have an `__init__.py` (even empty). Verify with `find bot/ -type d` — each must have the file.
**Warning signs:** Intermittent `ModuleNotFoundError` that works in some environments but fails on EC2.

### Pitfall 3: `WorkingDirectory` Not Set in systemd Unit
**What goes wrong:** `ModuleNotFoundError: No module named 'bot'` when the systemd service starts, even though it works when run manually.
**Why it happens:** systemd doesn't run from the project root by default; `sys.path[0]` becomes `/` or the user home dir.
**How to avoid:** Explicitly set `WorkingDirectory=` in the `[Service]` section to the full path of the project root.
**Warning signs:** Script works with `python main.py` from project root, fails under systemd.

### Pitfall 4: `.env` Committed to Git
**What goes wrong:** API keys in git history → competition disqualification risk.
**Why it happens:** Forgetting `.env` in `.gitignore` before first commit.
**How to avoid:** `.gitignore` must include `.env` BEFORE first `git add`. Also gitignore `state.json`, `state.json.tmp`, `state.json.bak`, `logs/`, `data/parquet/`.
**Warning signs:** `git status` shows `.env` as untracked (add before any commit) or tracked (use `git rm --cached .env`).

### Pitfall 5: Hardcoded Values in `config.yaml` That Contradict PROJECT.md
**What goes wrong:** Later phases use wrong thresholds (e.g., wrong circuit breaker %, wrong cooldown).
**Why it happens:** Approximating values from memory instead of reading PROJECT.md spec.
**How to avoid:** All values in `config.yaml` must come directly from PROJECT.md spec section. Key values: `trade_cooldown_seconds: 65`, `max_positions: 1`, circuit breaker at 30%/20%/10%, `candle_interval: "4h"`.
</common_pitfalls>

<code_examples>
## Code Examples

### requirements.txt (correct package names)
```
# Source: PyPI verified — pandas-ta-classic is the correct fork
pandas-ta-classic>=0.3.78
pandas>=2.0
numpy>=2.0
requests>=2.31
python-dotenv>=1.0
pyyaml>=6.0
```

### pandas-ta-classic import and usage
```python
# Source: xgboosted/pandas-ta-classic PyPI + GitHub
# PyPI: pandas-ta-classic (hyphens)
# Import: pandas_ta_classic (underscores)
# Accessor: df.ta (same as original pandas-ta)

import pandas_ta_classic  # registers df.ta accessor

import pandas as pd

df = pd.DataFrame({"close": [100, 101, 99, 102, 103]})
df.ta.rsi(length=14, append=True)    # adds RSI_14 column
df.ta.macd(fast=12, slow=26, signal=9, append=True)  # adds MACD columns
df.ta.ema(length=20, append=True)    # adds EMA_20 column
```

### .env.example (three key sets)
```bash
# Source: PROJECT.md spec — three key sets required
# Testing keys (use before competition starts)
ROOSTOO_API_KEY_TEST=your_testing_key_here
ROOSTOO_SECRET_TEST=your_testing_secret_here

# Round 1 competition keys (active from Mar 21 8PM)
ROOSTOO_API_KEY=your_round1_key_here
ROOSTOO_SECRET=your_round1_secret_here

# Round 2 finalist keys (placeholder — issued if finalist)
# ROOSTOO_API_KEY_R2=
# ROOSTOO_SECRET_R2=

# Telegram alerting (optional but recommended)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Verify bot namespace after scaffolding
```bash
# Run from project root — should get ImportError (class not found), NOT ModuleNotFoundError (package not found)
python -c "from bot.api.client import RoostooClient"
# Expected: ModuleNotFoundError: cannot import name 'RoostooClient' from 'bot.api.client'
# NOT: ModuleNotFoundError: No module named 'bot'
```
</code_examples>

<sota_updates>
## State of the Art (2025-2026)

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pandas-ta | pandas-ta-classic | 2023 (fork) | Original unmaintained; classic fork is the community standard |
| setup.py | pyproject.toml | 2021+ | pyproject.toml is PEP 517/518 standard; but not needed for this deployment pattern |
| requirements.txt only | requirements.txt + pyproject.toml | 2022+ | For simple script deployments, requirements.txt alone is sufficient |

**New tools/patterns to consider:**
- **`uv`**: Fast Python package installer; `pandas-ta-classic` documentation mentions it. Not necessary for EC2 deployment but faster than `pip` for CI.

**Deprecated/outdated:**
- **`pandas-ta`** (original by twopirllc): Unmaintained since 2022; ownership change raised supply chain concerns; broken on pandas 2.x.
- **`setup.py`**: Still works but superseded by `pyproject.toml`; not needed here.
</sota_updates>

<open_questions>
## Open Questions

1. **`pandas_ta_classic` accessor name**
   - What we know: The library registers a `.ta` accessor (same as original `pandas-ta`)
   - What's unclear: If both `pandas-ta` and `pandas-ta-classic` are installed simultaneously, there could be a namespace conflict on `.ta`
   - Recommendation: Only install `pandas-ta-classic`; never have both installed. The requirements.txt should NOT include `pandas-ta`.

2. **`config.yaml` location loading**
   - What we know: File lives at `bot/config/config.yaml`
   - What's unclear: Phase 1 only creates the file; the loader (`bot/config/__init__.py` or a `loader.py`) comes later
   - Recommendation: Phase 1 creates the YAML file only; loading logic is out of scope for this phase.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- PyPI: https://pypi.org/project/pandas-ta-classic/ — package name, version (0.3.78), install command
- GitHub: https://github.com/xgboosted/pandas-ta-classic — fork origin, API compatibility, Python 3.11 + pandas 2.x support
- Python docs: https://docs.python.org/3/reference/import.html — `sys.path[0]` behavior when running scripts

### Secondary (MEDIUM confidence)
- pandas-ta-classic documentation: https://xgboosted.github.io/pandas-ta-classic/ — df.ta accessor usage
- Python packaging guide: https://packaging.python.org — `__init__.py` vs namespace packages

### Tertiary (LOW confidence - needs validation)
- None — all findings verified against primary sources
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: Python 3.11 packaging, `bot.*` namespace
- Ecosystem: `pandas-ta-classic` fork verification
- Patterns: systemd WorkingDirectory, empty `__init__.py` namespace
- Pitfalls: wrong pandas-ta package, missing __init__.py, systemd path, .env in git

**Confidence breakdown:**
- Standard stack: HIGH — verified on PyPI, GitHub
- Architecture: HIGH — standard Python behavior, verified with docs
- Pitfalls: HIGH — common, well-documented issues
- Code examples: HIGH — from PyPI + official Python docs

**Research date:** 2026-03-16
**Valid until:** 2026-06-16 (90 days — stable packaging ecosystem)
</metadata>

---

*Phase: 01-project-scaffolding*
*Research completed: 2026-03-16*
*Ready for planning: yes*

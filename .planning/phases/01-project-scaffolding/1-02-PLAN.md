---
phase: 01-project-scaffolding
type: execute
---

<objective>
Create all config and dependency files: `requirements.txt`, `bot/config/config.yaml`, `.env.example`, and `.gitignore`.

Purpose: Every agent needs the correct dependency list, a fully-populated config, the right key-set template, and secrets excluded from git — all before any code is written.
Output: Four files with production-ready content sourced directly from PROJECT.md specs.
</objective>

<execution_context>
~/.claude/get-shit-done/workflows/execute-phase.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/01-project-scaffolding/1-CONTEXT.md
@.planning/phases/01-project-scaffolding/1-RESEARCH.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create requirements.txt with correct package names</name>
  <files>requirements.txt</files>
  <action>
    Create `requirements.txt` at the project root with these exact packages:

    ```
    pandas-ta-classic>=0.3.78
    pandas>=2.0
    numpy>=2.0
    requests>=2.31
    python-dotenv>=1.0
    pyyaml>=6.0
    ```

    CRITICAL: Use `pandas-ta-classic` (hyphenated, with `-classic` suffix) — NOT `pandas-ta`.
    The original `pandas-ta` package is unmaintained and broken on Python 3.11 + pandas 2.x.
    The import name is `pandas_ta_classic` (underscores) but the PyPI name uses hyphens.
    `hmac` and `hashlib` are stdlib — do NOT add them as pip dependencies.
  </action>
  <verify>
    grep "pandas-ta-classic" requirements.txt returns a match.
    grep "pandas-ta$" requirements.txt returns no match (the non-classic version must not be present).
  </verify>
  <done>
    `requirements.txt` exists at project root. Contains `pandas-ta-classic>=0.3.78`. Does NOT contain a bare `pandas-ta` line.
  </done>
</task>

<task type="auto">
  <name>Task 2: Create bot/config/config.yaml with all tunable parameters</name>
  <files>bot/config/config.yaml</files>
  <action>
    Create `bot/config/config.yaml` with all tunable parameters sourced from PROJECT.md. Use real values — no placeholders.

    Required sections and values:

    **Trading universe:**
    - `tradeable_pairs: ["BTC/USD"]`  ← BTC/USD only; Roostoo API constraint
    - `feature_pairs: ["BTC/USD", "ETH/USD", "SOL/USD"]`  ← cross-asset features
    - `candle_interval: "4h"`
    - `max_positions: 1`

    **Cooldown:**
    - `trade_cooldown_seconds: 65`  ← 65s minimum between place_order calls (PROJECT.md + FAQ Q19)

    **Risk management:**
    - `hard_stop_pct: 0.05`  ← 5% hard stop below entry price
    - `atr_stop_multiplier: 2.0`  ← multiplier applied to ATR proxy for stop placement
    - `trailing_stop_multiplier: 1.5`  ← multiplier for trailing stop distance

    **Circuit breaker (tiered — from PROJECT.md):**
    ```yaml
    circuit_breaker:
      halt_threshold: 0.30      # 30%+ drawdown: full halt (0% size)
      reduce_heavy_threshold: 0.20  # 20-30%: 25% normal size
      reduce_light_threshold: 0.10  # 10-20%: 50% normal size
      # <10% drawdown: 100% normal size
    ```

    **Data pipeline:**
    - `warmup_bars: 35`  ← MACD(12,26,9) minimum warmup
    - `regime_warmup_bars: 300`  ← 4H bars needed before regime detection is reliable (300 4H bars ≈ 50 daily bars for EMA50)

    **Regime detection:**
    ```yaml
    regime:
      ema_fast: 20
      ema_slow: 50
      confirmation_bars: 2    ← hysteresis to prevent thrashing at crossover boundary
    ```

    **Logging:**
    - `log_level: "INFO"`
    - `log_max_bytes: 10485760`  ← 10 MB
    - `log_backup_count: 10`

    Format: standard YAML. No Python-specific syntax. Group related params under their section keys.
  </action>
  <verify>
    python -c "import yaml; cfg = yaml.safe_load(open('bot/config/config.yaml')); assert cfg['trade_cooldown_seconds'] == 65; assert cfg['tradeable_pairs'] == ['BTC/USD']; assert cfg['circuit_breaker']['halt_threshold'] == 0.30; print('config.yaml OK')"
    Expected output: config.yaml OK
  </verify>
  <done>
    `bot/config/config.yaml` exists. `yaml.safe_load()` parses it without error. Key values present: `trade_cooldown_seconds=65`, `tradeable_pairs=["BTC/USD"]`, `circuit_breaker.halt_threshold=0.30`, `regime.ema_fast=20`, `regime.ema_slow=50`.
  </done>
</task>

<task type="auto">
  <name>Task 3: Create .env.example with three key sets</name>
  <files>.env.example</files>
  <action>
    Create `.env.example` at the project root. This file IS committed to git (it is the template, not the secrets).
    The actual `.env` is gitignored.

    Required content — three key sets exactly as specified in PROJECT.md:

    ```bash
    # Roostoo API Keys — DO NOT commit .env, only .env.example
    # Copy this file to .env and fill in your actual keys

    # Testing keys (use for smoke tests before competition starts)
    ROOSTOO_API_KEY_TEST=your_testing_key_here
    ROOSTOO_SECRET_TEST=your_testing_secret_here

    # Round 1 competition keys (active from Mar 21 8PM SGT)
    ROOSTOO_API_KEY=your_round1_key_here
    ROOSTOO_SECRET=your_round1_secret_here

    # Round 2 finalist keys (placeholder — keys issued only if you reach finals)
    # ROOSTOO_API_KEY_R2=
    # ROOSTOO_SECRET_R2=

    # Telegram alerting (optional but strongly recommended for monitoring)
    TELEGRAM_BOT_TOKEN=your_bot_token_here
    TELEGRAM_CHAT_ID=your_chat_id_here

    # API base URL
    ROOSTOO_BASE_URL=https://mock-api.roostoo.com
    ```

    Key points:
    - TEST key pair uses `_TEST` suffix; Round 1 pair has NO suffix (these are the default active keys)
    - Round 2 keys are COMMENTED OUT — they are issued only if the team reaches finals
    - Using wrong keys during competition = wasted trades or disqualification
  </action>
  <verify>
    grep "ROOSTOO_API_KEY_TEST" .env.example returns a match.
    grep "ROOSTOO_API_KEY=" .env.example returns a match (the Round 1 key without _TEST suffix).
    grep "ROOSTOO_API_KEY_R2" .env.example returns a commented-out line.
  </verify>
  <done>
    `.env.example` exists at project root. Contains all three key sets: TEST, Round 1 (no suffix), and commented R2 placeholder. Also includes TELEGRAM_ vars and ROOSTOO_BASE_URL.
  </done>
</task>

<task type="auto">
  <name>Task 4: Create .gitignore covering secrets and runtime artifacts</name>
  <files>.gitignore</files>
  <action>
    Create `.gitignore` at the project root. Must cover ALL of these — secrets and large/ephemeral runtime files must never land in git:

    **Secrets (CRITICAL):**
    - `.env`

    **Python cache:**
    - `__pycache__/`
    - `*.pyc`
    - `*.pyo`
    - `*.pyd`
    - `.pytest_cache/`
    - `*.egg-info/`
    - `dist/`
    - `build/`
    - `.venv/`
    - `venv/`

    **Runtime state (atomic write produces .tmp and .bak intermediates):**
    - `state.json`
    - `state.json.tmp`
    - `state.json.bak`

    **Logs:**
    - `logs/`

    **Historical data (large Parquet files):**
    - `data/parquet/`
    - `*.parquet`

    **OS/editor noise:**
    - `.DS_Store`
    - `Thumbs.db`
    - `*.swp`
    - `*.swo`

    If a `.gitignore` already exists at the root, ADD the missing entries; do not overwrite existing content.
  </action>
  <verify>
    git check-ignore -v .env  ← should output the .gitignore rule that matches .env
    git check-ignore -v state.json  ← should output the matching rule
    git check-ignore -v logs/  ← should output the matching rule
  </verify>
  <done>
    `.gitignore` exists. `git check-ignore .env`, `git check-ignore state.json`, and `git check-ignore logs/` all return matches (files are ignored). `.env.example` is NOT ignored (it should be committed).
  </done>
</task>

</tasks>

<verification>
Before declaring this plan complete:
- [ ] `grep "pandas-ta-classic" requirements.txt` matches; `grep "^pandas-ta$" requirements.txt` does NOT match
- [ ] `python -c "import yaml; yaml.safe_load(open('bot/config/config.yaml'))"` exits 0
- [ ] `config.yaml` contains `trade_cooldown_seconds: 65` and `halt_threshold: 0.30` under `circuit_breaker`
- [ ] `.env.example` contains `ROOSTOO_API_KEY_TEST`, `ROOSTOO_API_KEY` (no suffix), and a commented `ROOSTOO_API_KEY_R2`
- [ ] `git check-ignore .env` returns a match; `git check-ignore .env.example` returns NO match
- [ ] `git check-ignore state.json` returns a match
- [ ] `git check-ignore logs/` returns a match
</verification>

<success_criteria>

- All tasks completed
- All verification checks pass
- `requirements.txt` uses `pandas-ta-classic` exclusively (never `pandas-ta`)
- `config.yaml` has real values for all PROJECT.md-specified parameters (no placeholders)
- `.env.example` has all three key sets in the correct format
- `.gitignore` protects `.env`, `state.json*`, `logs/`, and `data/parquet/`
  </success_criteria>

<output>
After completion, create `.planning/phases/01-project-scaffolding/1-02-SUMMARY.md`:

# Phase 1 Plan 02: Config and Dependency Files Summary

**[Substantive one-liner — e.g. "requirements.txt, config.yaml, .env.example, and .gitignore created with production-ready values"]**

## Accomplishments

- [Key outcome 1]
- [Key outcome 2]

## Files Created/Modified

- `requirements.txt` - Dependency list with pandas-ta-classic>=0.3.78
- `bot/config/config.yaml` - Full parameter set from PROJECT.md spec
- `.env.example` - Three API key set template
- `.gitignore` - Secrets and runtime artifacts excluded

## Decisions Made

[Key decisions and rationale, or "None"]

## Issues Encountered

[Problems and resolutions, or "None"]

## Next Step

Phase 1 complete. Both plans (01-01 package skeleton, 01-02 config files) done. Ready for Phase 2: API Client & Rate Limiter.
</output>

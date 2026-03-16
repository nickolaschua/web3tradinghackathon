---
phase: 01-project-scaffolding
type: execute
---

<objective>
Create all missing `bot/*` subdirectory `__init__.py` files and the `tests/__init__.py` skeleton so the `bot.*` namespace is fully importable.

Purpose: Every subsequent agent imports from `bot.api`, `bot.data`, `bot.config`, `bot.strategy`, `bot.monitoring`, `bot.persistence`, and `bot.execution`. All must resolve without ModuleNotFoundError before any logic is written.
Output: Empty `__init__.py` in every `bot/` subdirectory that is currently missing one; `tests/__init__.py` skeleton.
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
  <name>Task 1: Create missing subdirectory __init__.py files</name>
  <files>
    bot/api/__init__.py,
    bot/data/__init__.py,
    bot/config/__init__.py,
    bot/strategy/__init__.py,
    bot/monitoring/__init__.py,
    bot/persistence/__init__.py
  </files>
  <action>
    Create each file as completely empty (zero bytes / no content).
    DO NOT add any imports, classes, or comments — Phase 1 scope is pure structure, no logic.

    SKIP: `bot/__init__.py` and `bot/execution/__init__.py` — these already exist from earlier Phase 5 scaffolding work. Do not overwrite them.

    Directories to create if they do not exist: `bot/api/`, `bot/data/`, `bot/config/`, `bot/strategy/`, `bot/monitoring/`, `bot/persistence/`.
  </action>
  <verify>
    From the project root run:
    python -c "from bot.api.client import RoostooClient"
    Expected output: ModuleNotFoundError: cannot import name 'RoostooClient' from 'bot.api.client' (or: No module named 'bot.api.client')
    NOT acceptable: ModuleNotFoundError: No module named 'bot' OR No module named 'bot.api'
    Any error mentioning the package itself (not the class) means an __init__.py is still missing.
  </verify>
  <done>
    `python -c "import bot.api; import bot.data; import bot.config; import bot.strategy; import bot.monitoring; import bot.persistence"` exits with code 0.
    All six subdirectory packages are importable.
  </done>
</task>

<task type="auto">
  <name>Task 2: Create tests/__init__.py skeleton</name>
  <files>tests/__init__.py</files>
  <action>
    Create `tests/` directory if it does not exist.
    Create `tests/__init__.py` as an empty file (zero bytes).
    No test logic — the skeleton is just to establish the `tests` package for future phases.
  </action>
  <verify>
    python -c "import tests" exits with code 0.
  </verify>
  <done>
    `tests/__init__.py` exists and `import tests` succeeds from project root.
  </done>
</task>

</tasks>

<verification>
Before declaring this plan complete:
- [ ] `python -c "import bot.api; import bot.data; import bot.config; import bot.strategy; import bot.monitoring; import bot.persistence"` exits 0
- [ ] `python -c "import tests"` exits 0
- [ ] `python -c "from bot.api.client import RoostooClient"` returns an ImportError about the class name, NOT about the package
- [ ] `bot/__init__.py` and `bot/execution/__init__.py` are unchanged (not accidentally overwritten)
- [ ] All new `__init__.py` files are empty (no content added)
</verification>

<success_criteria>

- All tasks completed
- All verification checks pass
- Six missing subdirectory `__init__.py` files created (api, data, config, strategy, monitoring, persistence)
- `tests/__init__.py` skeleton created
- `bot.*` namespace fully importable from project root with no sys.path manipulation
  </success_criteria>

<output>
After completion, create `.planning/phases/01-project-scaffolding/1-01-SUMMARY.md`:

# Phase 1 Plan 01: Package Skeleton Summary

**[Substantive one-liner — e.g. "bot.* namespace complete: all six missing subdirectory __init__.py files created"]**

## Accomplishments

- [Key outcome 1]
- [Key outcome 2]

## Files Created/Modified

- `bot/api/__init__.py` - Empty package marker
- `bot/data/__init__.py` - Empty package marker
- `bot/config/__init__.py` - Empty package marker
- `bot/strategy/__init__.py` - Empty package marker
- `bot/monitoring/__init__.py` - Empty package marker
- `bot/persistence/__init__.py` - Empty package marker
- `tests/__init__.py` - Empty test package skeleton

## Decisions Made

None — structure is fully specified in PROJECT.md.

## Issues Encountered

[Problems and resolutions, or "None"]

## Next Step

Ready for 1-02-PLAN.md (config and dependency files)
</output>

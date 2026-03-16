---
phase: 08-ec2-deployment
plan: "08-01"
subsystem: deployment
tags: [systemd, ec2, al2023, python3.11, venv]
requires: []
provides:
  - deploy/roostoo-bot.service (production-ready systemd unit)
  - deploy/bootstrap.sh (AL2023 one-shot setup)
  - deploy/deploy.sh (update deploy script)
affects: ["08-02-deploy-verify"]
tech-stack:
  added: []
  patterns: ["venv-absolute-path-ExecStart", "EnvironmentFile-bare-kv", "ReadWritePaths-whole-dir"]
key-files:
  created:
    - deploy/roostoo-bot.service
    - deploy/bootstrap.sh
    - deploy/deploy.sh
key-decisions:
  - "ReadWritePaths=/home/ec2-user/bot (whole dir, not individual files) — state.json doesn't exist until first write; narrow paths cause systemd start failure"
  - "MemoryDenyWriteExecute omitted — breaks numpy/pandas JIT at import"
  - "network-online.target ensures IP assigned before bot's first API call"
---

# Phase 8 Plan 01: Deployment Artifacts Summary

Three production-ready deployment artifacts committed: systemd unit (venv ExecStart, hardening, EnvironmentFile), AL2023 bootstrap script (python3.11 explicit), code update deploy script.

## Accomplishments

- **Created `/deploy/roostoo-bot.service`** — Production-ready systemd unit with:
  - Venv Python ExecStart (`/home/ec2-user/bot/venv/bin/python main.py`)
  - Network dependency (`After=network-online.target`, `Wants=network-online.target`)
  - Environment file injection (`EnvironmentFile=/home/ec2-user/bot/.env`)
  - Hardening directives (NoNewPrivileges, PrivateTmp, ProtectSystem=strict, RestrictAddressFamilies, RestrictRealtime, LockPersonality)
  - Safe for Python/pandas/numpy (no MemoryDenyWriteExecute, no ProtectHome)
  - ReadWritePaths covering whole bot directory (handles state.json not existing on first start)

- **Created `/deploy/bootstrap.sh`** — One-shot AL2023 setup script with:
  - System package update (dnf)
  - Python 3.11 explicit installation (not python3 which is 3.9 on AL2023)
  - Git clone with repo URL placeholder (user edits)
  - Venv creation using python3.11 binary
  - Requirements installation via venv pip
  - `.env.example` → `.env` copy with 600 permissions
  - Systemd service installation and enablement

- **Created `/deploy/deploy.sh`** — Update deployment script with:
  - SSH to EC2 instance with IP argument
  - `git pull origin main` for latest code
  - `pip install -r requirements.txt` via venv
  - `systemctl restart roostoo-bot.service`
  - Service status verification

## Files Created/Modified

- `deploy/roostoo-bot.service` - Production-ready systemd unit with venv ExecStart, hardening, EnvironmentFile
- `deploy/bootstrap.sh` - AL2023 first-time setup: dnf python3.11, git clone, venv, pip install, service install
- `deploy/deploy.sh` - Code update script: git pull + pip install + systemctl restart

## Decisions Made

1. **ReadWritePaths scope**: Entire `/home/ec2-user/bot` directory (not narrow file paths). Rationale: `state.json` doesn't exist until first bot run; systemd enforces ReadWritePaths at startup, so narrow paths cause "Permission denied" before any bot code runs.

2. **MemoryDenyWriteExecute omitted**: Pandas and numpy use W+X memory for JIT compilation; this security hardening breaks them silently at import. Safe hardening subset chosen instead.

3. **ProtectHome omitted**: WorkingDirectory is `/home/ec2-user/bot`, which is under `/home/ec2-user`; ProtectHome creates conflict. Removed to avoid startup failure.

4. **network-online.target**: Ensures IP is assigned before bot's first API call (mock-api.roostoo.com). Prevents race condition on startup.

5. **python3.11 explicit in bootstrap.sh**: AL2023 ships with python3=3.9; venv activation also reads 3.9. Script hardcodes `python3.11` binary to guarantee correct version throughout.

6. **ExecStart venv path**: Absolute path to venv Python (`/home/ec2-user/bot/venv/bin/python main.py`), not `source activate` or `python3` symlink. Systemd runs in clean environment; explicit path is only reliable method.

## Issues Encountered

None. All requirements met; all verification checks pass.

## Next Step

Ready for 08-02-PLAN.md — EC2 provisioning and deployment execution (launch instance, populate `.env`, smoke test, confirm first trade executes).

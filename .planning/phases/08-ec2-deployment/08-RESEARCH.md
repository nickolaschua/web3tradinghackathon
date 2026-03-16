# Phase 8: EC2 Deployment - Research

**Researched:** 2026-03-17
**Domain:** EC2 t3.micro (ap-southeast-2) + Amazon Linux 2023 + Python 3.11 + systemd + chrony
**Confidence:** HIGH

<research_summary>
## Summary

Phase 8 deploys the Roostoo bot to a t3.micro EC2 instance in ap-southeast-2. The stack is Amazon Linux 2023 (AL2023) + Python 3.11 + virtualenv + systemd service + chrony time sync. All of these are well-established patterns with authoritative documentation.

**Key finding 1 (AL2023):** AL2023 ships with Python 3.9 as the system default (`/usr/bin/python3`). Python 3.11 is available natively via `dnf install python3.11` — no source compilation needed. Do NOT reassign the `python3` symlink; it breaks AL2023 internals. Use `python3.11 -m venv venv` explicitly.

**Key finding 2 (chrony):** Amazon Time Sync (169.254.169.123) is **already configured by default** on AL2023 via `/etc/chrony.d/link-local.sources`. No manual chrony setup is required — just verify with `chronyc sources -v` and confirm `^*` next to `169.254.169.123`.

**Key finding 3 (systemd .env):** systemd's `EnvironmentFile=` injects `.env` variables directly into the process environment. The Python code therefore does NOT need `load_dotenv()` — `os.environ['ROOSTOO_API_KEY']` works as-is. This also means the `.env` file must use bare `KEY=VALUE` format (no quotes, no `export` prefix).

**Primary recommendation:** Use AL2023 + dnf Python 3.11 + venv ExecStart (no activation) + EnvironmentFile for secrets + `network-online.target` dependency + `RestartSec=10` + `ReadWritePaths=` for logs/state under `ProtectSystem=strict`.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| Amazon Linux 2023 | Latest (AL2023) | EC2 OS | AWS-maintained, dnf-based, Python 3.11 available natively |
| Python 3.11 | 3.11.x (dnf) | Runtime | Required by project; available as `python3.11` package in AL2023 |
| virtualenv (venv) | stdlib | Dependency isolation | No global pip pollution; `python3.11 -m venv venv` |
| systemd | System default | Process management | Handles restart, boot, logging; no supervisor/pm2 needed |
| chrony | AL2023 default | NTP time sync | Already configured for Amazon Time Sync on AL2023 |

### Supporting
| Component | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| journalctl | System default | Log viewing | `journalctl -u roostoo-bot -f` for live tailing |
| systemd-analyze | System default | Security scoring | `systemd-analyze security roostoo-bot.service` |
| git | dnf | Code deployment | `git pull origin main` deploy pattern |
| scp / ssh | System | File transfer | Copy .env to instance (never commit secrets) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| systemd | supervisor | systemd is native, no extra install; supervisor adds dependency |
| systemd | screen/nohup | screen/nohup has no auto-restart on crash; not suitable for production |
| AL2023 | Ubuntu 22.04 | Both work; AL2023 has Time Sync preconfigured + AWS support |
| git pull deploy | scp deploy | git pull is simpler for one-person team; scp copies whole project |

**Installation (on EC2 after SSH):**
```bash
# System packages
sudo dnf update -y
sudo dnf install python3.11 python3.11-pip git -y

# Project setup
cd /home/ec2-user
git clone <repo-url> bot
cd bot
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure on EC2
```
/home/ec2-user/
├── bot/                        # git repo clone
│   ├── main.py
│   ├── bot/                    # Python package
│   ├── requirements.txt
│   ├── .env                    # NOT in git; copied via scp
│   ├── logs/                   # needs ReadWritePaths= in service
│   │   ├── bot.log
│   │   └── trades.log
│   └── state.json              # atomic write target; needs ReadWritePaths=
└── venv/                       # OUTSIDE repo dir (or inside, both work)
    └── bin/python3.11
```

### Pattern 1: systemd Service Unit (Correct ExecStart + .env + venv)
**What:** Point ExecStart directly to venv Python binary — no activation needed. Load secrets via EnvironmentFile. Depend on `network-online.target` (not just `network.target`) so the bot starts after a real IP is assigned.
**When to use:** Always — this is the standard pattern.

```ini
# /etc/systemd/system/roostoo-bot.service

[Unit]
Description=Roostoo Quant Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/bot
EnvironmentFile=/home/ec2-user/bot/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/ec2-user/venv/bin/python main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Hardening
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/home/ec2-user/bot/logs /home/ec2-user/bot/state.json
RestrictAddressFamilies=AF_INET AF_INET6
RestrictRealtime=yes
LockPersonality=yes

[Install]
WantedBy=multi-user.target
```

**Critical notes:**
- `ExecStart` uses absolute path to venv python — no `source activate` in systemd
- `EnvironmentFile=` injects `.env` vars into process env — Python reads via `os.environ`, no `load_dotenv()` call needed
- `ReadWritePaths=` is REQUIRED when `ProtectSystem=strict` — without it, the bot cannot write `state.json` or logs
- Do NOT add `MemoryDenyWriteExecute=yes` — Python + numpy/pandas uses JIT; this will crash the process

### Pattern 2: .env File Format for EnvironmentFile
**What:** systemd `EnvironmentFile=` requires bare `KEY=VALUE` format. No quotes, no `export`, no spaces around `=`.
**When to use:** The .env file loaded by systemd (not the one for local dev if using python-dotenv).

```bash
# /home/ec2-user/bot/.env  — systemd EnvironmentFile format
ROOSTOO_API_KEY_TEST=your_test_key_here
ROOSTOO_SECRET_TEST=your_test_secret_here
ROOSTOO_API_KEY=your_round1_key_here
ROOSTOO_SECRET=your_round1_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

File permissions must be `600` (only ec2-user can read):
```bash
chmod 600 /home/ec2-user/bot/.env
```

### Pattern 3: chrony Verification on AL2023
**What:** AL2023 pre-configures Amazon Time Sync via `/etc/chrony.d/link-local.sources`. No manual config needed. Just verify.
**When to use:** After instance launch — verification only.

```bash
# Verify Amazon Time Sync is active
chronyc sources -v
# Expected: line with ^* next to 169.254.169.123

# Detailed metrics
chronyc tracking
# Expected: Reference ID = 169.254.169.123, Stratum 4, Leap status = Normal

# If not synced (rare), add manually to /etc/chrony.conf:
# server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4
# sudo systemctl restart chronyd
```

### Pattern 4: Deploy Script (post-launch updates)
**What:** Simple shell script for pushing code updates after initial deployment.
**When to use:** Any code change that needs to go to EC2.

```bash
#!/bin/bash
# deploy.sh — run from repo root on local machine
EC2_HOST="ec2-user@<instance-ip>"
EC2_DIR="/home/ec2-user/bot"

ssh $EC2_HOST "cd $EC2_DIR && \
  git pull origin main && \
  source /home/ec2-user/venv/bin/activate && \
  pip install -r requirements.txt && \
  sudo systemctl restart roostoo-bot && \
  sudo systemctl status roostoo-bot --no-pager"
```

### Pattern 5: Systemd Enable + Start Sequence
**What:** One-time setup after writing the service file.

```bash
sudo systemctl daemon-reload
sudo systemctl enable roostoo-bot.service   # auto-start on reboot
sudo systemctl start roostoo-bot.service    # start now
sudo systemctl status roostoo-bot.service --no-pager
journalctl -u roostoo-bot.service -f        # follow logs
```

### Anti-Patterns to Avoid
- **`network.target` instead of `network-online.target`:** Only ensures network _service_ started, not that IP is assigned — bot fails to reach `mock-api.roostoo.com` on startup
- **`MemoryDenyWriteExecute=yes`:** Breaks Python JIT in numpy/pandas; service crashes silently at import
- **Changing `/usr/bin/python3` symlink:** Breaks AL2023 core tooling (dnf, cloud-init, etc.)
- **`Restart=always`:** Masks repeated crashes in rapid loop; use `Restart=on-failure` + `RestartSec=10`
- **Relative paths in ExecStart:** systemd does not expand `~` — must use `/home/ec2-user/...`
- **`.env` with quoted values for EnvironmentFile:** `KEY="value"` passes literal quotes to Python; use bare `KEY=value`
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Time sync | Manual NTP config | AL2023 default chrony | Already pre-configured on AL2023; just verify |
| Process restart on crash | Custom watchdog script | systemd `Restart=on-failure` | Built-in, handles SIGTERM/SIGKILL, logs to journal |
| Log rotation | Custom log rotation code | logrotate or journald | journald handles rotation automatically for systemd services |
| Secret injection | Custom env-loading wrapper | systemd `EnvironmentFile=` | Injects before process start; no code changes needed |
| Boot autostart | rc.local / cron @reboot | systemd `enable` | Handles ordering, dependencies, and failure recovery |
| SSH key management | Password auth | EC2 key pair → `~/.ssh/authorized_keys` | Password auth is insecure; key pairs are EC2 standard |

**Key insight:** EC2 + AL2023 + systemd is a well-oiled stack. Every problem listed above has been solved at the OS level — don't layer application code over OS primitives.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: `network.target` Causes Startup Race
**What goes wrong:** Bot starts before DHCP assigns the instance IP; first API call to `mock-api.roostoo.com` fails; systemd restarts after `RestartSec=10`; this loop repeats until the race is won — or worse, the first restart happens mid-`startup_reconciliation()`.
**Why it happens:** `network.target` only means systemd's network _service unit_ started, not that an IP is assigned or DNS is reachable.
**How to avoid:** Use `After=network-online.target` + `Wants=network-online.target` in `[Unit]`.
**Warning signs:** Bot logs show connection refused / DNS resolution failure in the first 5-10 seconds after launch; `systemctl status` shows repeated start cycles.

### Pitfall 2: `ProtectSystem=strict` Without `ReadWritePaths`
**What goes wrong:** Bot launches, imports fine, tries to write `state.json` or `logs/bot.log` — gets `PermissionError: [Errno 13]` and crashes. systemd restarts; same crash; tight restart loop.
**Why it happens:** `ProtectSystem=strict` makes the entire filesystem read-only for the service, including the project directory.
**How to avoid:** Add `ReadWritePaths=/home/ec2-user/bot/logs /home/ec2-user/bot/state.json` to the service unit.
**Warning signs:** `journalctl -u roostoo-bot` shows `PermissionError` on first write; repeated crash/restart cycle visible in `systemctl status`.

### Pitfall 3: `.env` Quoted Values Break API Auth
**What goes wrong:** `ROOSTOO_API_KEY="abc123"` in `.env` → Python gets `'"abc123"'` (with literal quotes) → HMAC signing uses wrong key → all API calls return 401 Unauthorized.
**Why it happens:** systemd `EnvironmentFile=` is NOT the same as bash — it does not strip quotes.
**How to avoid:** Use bare `KEY=value` format in the `.env` file. No quotes, no `export` prefix.
**Warning signs:** API calls return 401; if you `print(os.environ['ROOSTOO_API_KEY'])` the value shows surrounding quotes.

### Pitfall 4: `python-dotenv` Double-Load Conflict
**What goes wrong:** If both systemd `EnvironmentFile=` and `load_dotenv()` in the Python code are active, `load_dotenv()` by default does NOT override existing env vars — so the systemd-injected values take precedence. This is usually fine, but can cause confusion during debugging (changing `.env` content doesn't change runtime behavior until systemd restarts).
**Why it happens:** Two paths injecting the same variable.
**How to avoid:** When using systemd `EnvironmentFile=`, remove `load_dotenv()` from the bot code (or keep it only with `override=False` which is the default). Best practice: use `EnvironmentFile=` in systemd only; drop `load_dotenv()` from `main.py`.
**Warning signs:** Changing `.env` on server has no effect until `sudo systemctl restart roostoo-bot`.

### Pitfall 5: Python 3.11 venv with Wrong Python Binary
**What goes wrong:** `python3 -m venv venv` creates a 3.9 venv on AL2023 (3.9 is the `python3` default). Bot imports fail if any library requires 3.11+ features, or if there's a subtle difference in f-strings / match-case syntax.
**Why it happens:** AL2023 sets `python3` → Python 3.9; must explicitly use `python3.11`.
**How to avoid:** Always use `python3.11 -m venv venv` to create the venv.
**Warning signs:** `venv/bin/python --version` shows 3.9.x; potential import issues with newer language features.

### Pitfall 6: systemd Doesn't See `state.json` Path Before First Run
**What goes wrong:** `ReadWritePaths=/home/ec2-user/bot/state.json` requires the path to exist; if `state.json` doesn't exist yet, systemd may refuse to start.
**Why it happens:** systemd validates `ReadWritePaths` entries at startup time.
**How to avoid:** Create the file before starting the service, or whitelist the parent directory: `ReadWritePaths=/home/ec2-user/bot` (then narrow down after first run confirms it works).
**Warning signs:** `systemctl start` fails immediately with a path-related error before the Python process even launches.
</common_pitfalls>

<code_examples>
## Code Examples

### Complete systemd Service File (Production-Ready)
```ini
# /etc/systemd/system/roostoo-bot.service
# Source: Verified from official systemd docs + AWS EC2 patterns

[Unit]
Description=Roostoo Quant Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/bot
EnvironmentFile=/home/ec2-user/bot/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/ec2-user/venv/bin/python main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Hardening (safe subset for Python + network service)
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/home/ec2-user/bot
RestrictAddressFamilies=AF_INET AF_INET6
RestrictRealtime=yes
LockPersonality=yes
# NOTE: Do NOT add MemoryDenyWriteExecute=yes — breaks Python JIT (numpy/pandas)

[Install]
WantedBy=multi-user.target
```

### AL2023 Bootstrap Script (run once after SSH)
```bash
#!/bin/bash
# Source: AWS AL2023 docs + community verification

# 1. Update system
sudo dnf update -y

# 2. Install Python 3.11 (available natively in AL2023)
sudo dnf install python3.11 python3.11-pip git -y

# 3. Clone repo
cd /home/ec2-user
git clone https://github.com/<org>/<repo>.git bot
cd bot

# 4. Create venv with Python 3.11 explicitly (NOT python3 which is 3.9)
python3.11 -m venv /home/ec2-user/venv
source /home/ec2-user/venv/bin/activate
pip install -r requirements.txt

# 5. Create .env from .env.example (then edit with actual keys)
cp .env.example .env
chmod 600 .env
nano .env  # fill in ROOSTOO_API_KEY_TEST, ROOSTOO_SECRET_TEST, TELEGRAM creds

# 6. Install and enable systemd service
sudo cp deploy/roostoo-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable roostoo-bot.service
sudo systemctl start roostoo-bot.service
```

### chrony Verification (AL2023 — verify only, no config needed)
```bash
# Source: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configure-ec2-ntp.html

# Check that 169.254.169.123 is the preferred source (^* marker)
chronyc sources -v

# Expected output includes:
# ^* 169.254.169.123    3   6   17   43   -30us[ -226us] +/- 287us

# Detailed sync info
chronyc tracking
# Reference ID should be 169.254.169.123, Leap status: Normal

# If NOT synced (add manually):
# echo "server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4" | sudo tee /etc/chrony.d/amazon-time-sync.sources
# sudo systemctl restart chronyd
```

### Smoke Test Sequence
```bash
# 1. Direct run first (before enabling systemd) — catches import errors
cd /home/ec2-user/bot
source /home/ec2-user/venv/bin/activate
python main.py  # Should see: startup_reconciliation() logs, first ticker poll

# 2. Check service starts cleanly
sudo systemctl status roostoo-bot.service --no-pager

# 3. Follow live logs
journalctl -u roostoo-bot.service -f

# 4. Verify key milestones in logs:
#    - "startup_reconciliation complete" (or similar)
#    - First ticker poll (Roostoo API responds)
#    - Telegram STARTED alert received
#    - state.json written on first cycle
#    - No exceptions in first 120 seconds

# 5. Check state.json written
ls -la /home/ec2-user/bot/state.json

# 6. Verify restart works
sudo systemctl restart roostoo-bot.service
journalctl -u roostoo-bot.service --since "1 minute ago"
```

### Security Group (minimum required)
```
Inbound:
  - Port 22 (SSH) — your IP only (not 0.0.0.0/0)

Outbound:
  - Port 443 (HTTPS) — 0.0.0.0/0  (for mock-api.roostoo.com + Telegram)
  - Port 80 (HTTP) — 0.0.0.0/0   (optional, for initial pip installs)
  - Port 123 (UDP NTP) — 0.0.0.0/0 (chrony, though link-local doesn't need this)
```

### Switching from Testing Keys to Round 1 Keys
```bash
# Edit .env on EC2 — swap TEST keys for live keys before Mar 21 8PM
nano /home/ec2-user/bot/.env
# Change: ROOSTOO_API_KEY and ROOSTOO_SECRET to Round 1 values

# Restart to pick up new env
sudo systemctl restart roostoo-bot.service

# Verify startup reconciliation runs clean with new keys
journalctl -u roostoo-bot.service -f
```
</code_examples>

<sota_updates>
## State of the Art (2025-2026)

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `yum` package manager | `dnf` on AL2023 | AL2023 launch (2022) | `yum` still works as alias but `dnf` is canonical |
| NTP daemon (`ntpd`) | chrony | ~2020 on most distros | chrony faster to sync, better accuracy; ntpd deprecated on AL2023 |
| supervisor/pm2 for Python | systemd native | 2018+ | systemd handles everything; no extra process manager needed |
| nohup + screen for background | systemd service | 2018+ | No crash recovery with nohup; systemd restarts automatically |
| Python 3.9 default on AL2023 | Python 3.11 via dnf | AL2023 recent update | 3.11 available as first-class package; no source compile |
| `network.target` in service | `network-online.target` | Best practice clarification | Prevents boot-time race conditions for internet-dependent services |

**New patterns to consider:**
- **AL2023 PTP hardware clock (`/dev/ptp_ena`):** For t3 instances on Nitro, chrony can use the hardware PTP clock for microsecond accuracy. Not needed for this bot (millisecond precision is fine), but available.
- **AL2023 `amazon-chrony-config` package:** Manages Amazon Time Sync config automatically; updates via `dnf update` — don't manually override `/etc/chrony.d/link-local.sources`.

**Deprecated/outdated on AL2023:**
- **`yum erase ntp*` + manual chrony install:** Not needed on AL2023; chrony is default
- **Building Python 3.11 from source:** Not needed on AL2023; use `dnf install python3.11`
- **`/etc/rc.local` for boot startup:** Use systemd `enable` instead
</sota_updates>

<open_questions>
## Open Questions

1. **Does `main.py` call `load_dotenv()`?**
   - What we know: PROJECT.md doesn't specify; earlier phases didn't add it
   - What's unclear: If `load_dotenv()` IS in main.py, it coexists harmlessly with `EnvironmentFile=` (systemd vars take precedence since they're already in the environment when Python starts)
   - Recommendation: During plan-phase, check if `load_dotenv()` exists in main.py. If so, leave it (it's harmless as a fallback for local dev). The systemd `EnvironmentFile=` will always win for the production service.

2. **Should the venv live inside or outside the git repo dir?**
   - What we know: Both `/home/ec2-user/venv` (outside) and `/home/ec2-user/bot/venv` (inside) work
   - What's unclear: Personal preference; inside is simpler; outside is cleaner for gitignore
   - Recommendation: Put venv inside the repo dir (`/home/ec2-user/bot/venv`) — simpler paths, already in `.gitignore`. Update `ReadWritePaths=` accordingly.

3. **Does the bot need a Telegram webhook or polling?**
   - What we know: `TelegramAlerter` uses outbound HTTP POST to Telegram API (send-only); no webhook needed
   - What's unclear: Nothing — outbound only, no inbound port needed
   - Recommendation: Security group needs port 443 outbound only.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- [AWS EC2 NTP Config (Official)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configure-ec2-ntp.html) — chrony Amazon Time Sync config, exact line, verification commands
- [AL2023 Python docs (Official)](https://docs.aws.amazon.com/linux/al2023/ug/python.html) — Python 3.11 installation via dnf, symlink warning
- [Amazon Time Sync Blog (AWS)](https://aws.amazon.com/blogs/aws/keeping-time-with-amazon-time-sync-service/) — confirmed default behavior on AL2023

### Secondary (MEDIUM confidence)
- systemd freedesktop docs — `EnvironmentFile=`, `ProtectSystem=strict`, `ReadWritePaths=` semantics; verified against multiple sources
- [AL2023 GitHub issue #345](https://github.com/amazonlinux/amazon-linux-2023/issues/345) — confirms AL2023 uses `/etc/chrony.d/` modular config, not single `/etc/chrony.conf`
- [Medium: systemd Python venv service](https://medium.com/@mailmeonriju/how-to-write-a-custom-systemctl-linux-service-for-python-scripts-with-virtualenv-and-env-file-c63c4625cbd7) — EnvironmentFile .env pattern; cross-verified with systemd docs
- [systemd service hardening (GitHub gist)](https://gist.github.com/ageis/f5595e59b1cddb1513d1b425a323db04) — NoNewPrivileges, ProtectSystem, RestrictAddressFamilies; cross-verified against ArchWiki

### Tertiary (LOW confidence — validate during execution)
- `MemoryDenyWriteExecute=yes` incompatibility with Python JIT libraries — sourced from systemd hardening guide and community reports; should test before enabling
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: EC2 + AL2023 + Python 3.11 + systemd
- Ecosystem: chrony, venv, EnvironmentFile, journalctl, git deploy
- Patterns: systemd service unit, chrony verification, bootstrap script, deploy script
- Pitfalls: network-online.target, ProtectSystem+ReadWritePaths, .env quoting, python3 symlink, MemoryDenyWriteExecute

**Confidence breakdown:**
- AL2023 Python 3.11 install: HIGH — official AWS docs, confirmed via dnf search
- chrony Amazon Time Sync: HIGH — official AWS docs, confirmed default on AL2023
- systemd service unit pattern: HIGH — well-documented, multiple cross-referenced sources
- Hardening directives: HIGH — ArchWiki + systemd docs; MemoryDenyWriteExecute caveat noted
- Deploy script pattern: HIGH — standard git pull + pip + systemctl restart, widely used

**Research date:** 2026-03-17
**Valid until:** 2026-06-17 (90 days — AL2023 + systemd + chrony are stable; unlikely to change)
</metadata>

---

*Phase: 08-ec2-deployment*
*Research completed: 2026-03-17*
*Ready for planning: yes*

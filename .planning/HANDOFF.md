# Handoff: EC2 Deployment (08-02)

**Paused:** 2026-03-17 (night)
**Resume:** Tomorrow morning — pick up at Step 6 inside the EC2 Session Manager terminal

---

## For the Human

### Where you are right now

You have:
- ✅ Signed into the hackathon AWS portal (`https://d-906625dad1.awsapps.com/start`)
- ✅ Launched EC2 instance (t3.medium, ap-southeast-2, from `HackathonBotTemplate`)
- ✅ Connected via Session Manager (browser terminal)
- ✅ Ran `cd ~` — you are in your home directory on the instance

You have NOT yet done:
- ❌ Cloned repo / set up venv / installed dependencies (Step 6)
- ❌ Filled in `.env` with testing keys (Step 7)
- ❌ Installed systemd service (Step 8)
- ❌ Verified chrony (Step 9)
- ❌ Started the bot and smoke tested it (Tasks 2 & 3)
- ❌ Switched to Round 1 competition keys (Task 4 — deadline: Mar 21 8PM SGT)

### Steps to complete tomorrow

**Reconnect to EC2:**
1. Go to `https://d-906625dad1.awsapps.com/start` → sign in
2. AWS Account → your team → Management Console
3. Confirm region is `ap-southeast-2`
4. EC2 → Instances → select your instance → Connect → Session Manager → Connect
5. In the terminal: `cd ~`

**Step 6 — Clone repo and set up venv:**
```bash
git clone https://github.com/<your-org>/<your-repo>.git bot
cd bot
python3.11 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
```

**Step 7 — Fill in .env** (bare KEY=VALUE — NO quotes, no `export`, no spaces around `=`):
```bash
nano .env
```
Set:
```
ROOSTOO_API_KEY=your_testing_api_key_here
ROOSTOO_SECRET=your_testing_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

**Step 8 — Install systemd service:**
```bash
sudo cp deploy/roostoo-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable roostoo-bot.service
```

**Step 9 — Verify chrony:**
```bash
chronyc sources -v
```
Expect `^*` next to `169.254.169.123`. If not:
```bash
echo "server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4" | sudo tee /etc/chrony.d/amazon-time-sync.sources
sudo systemctl restart chronyd && chronyc sources -v
```

Tell Claude: **"bootstrapped"** — then Claude will guide you through the remaining tasks (start service, smoke test, switch to Round 1 keys).

### ⚠️ Hard deadline

**Round 1 keys must be active by: Mar 21 8PM SGT (Mar 21 12:00 UTC)**

---

## For the Claude Agent

### Session context

- **Project:** Roostoo quant trading bot (web3 hackathon)
- **Working directory:** `C:\Users\nicko\Desktop\web3tradinghackathon\web3tradinghackathon`
- **Planning config:** yolo mode
- **Current plan executing:** `.planning/phases/08-ec2-deployment/08-02-PLAN.md`

### Where execution paused

The plan has 4 tasks, all `checkpoint:human-action` or `checkpoint:human-verify`:

| Task | Type | Status |
|------|------|--------|
| 1: Launch EC2, bootstrap, fill .env | checkpoint:human-action | **IN PROGRESS** — user has launched EC2 and connected; paused mid-Step 5 |
| 2: Start service, verify logs | checkpoint:human-action | Not started |
| 3: Smoke test verification | checkpoint:human-verify | Not started |
| 4: Switch to Round 1 keys | checkpoint:human-action | Not started |

### What to do when user resumes

The user will say something like "I'm back" or "continuing deployment".

Resume at **Task 1, Step 6** (clone repo). The user has already:
- Launched EC2 (t3.medium, ap-southeast-2, via HackathonBotTemplate — NOT t3.micro)
- Connected via Session Manager (browser terminal — SSH is blocked in this hackathon environment)
- Is inside the EC2 terminal at `~/`

### Key environment facts (different from what PLAN.md assumed)

| Assumption in plan | Reality (from aws_ec2_guide.md) |
|---|---|
| t3.micro | t3.medium (enforced by template) |
| Create own key pair + SSH | No key pair; Session Manager only |
| Manual AMI/networking config | Pre-configured via `HackathonBotTemplate` |
| `sudo dnf install python3.11` | Python already pre-installed via template |
| Sign in at aws.amazon.com | SSO portal: `https://d-906625dad1.awsapps.com/start` |

### When Task 1 is confirmed ("bootstrapped")

Present **Task 2 checkpoint:**
```
════════════════════════════════════════
CHECKPOINT: Human Action Required
════════════════════════════════════════

Task 2 of 4: Start roostoo-bot.service and follow live logs

Step 1 — Direct run first (catches import errors before systemd):
  cd /home/ec2-user/bot
  source venv/bin/activate
  python main.py
Watch 30 seconds. Ctrl+C to stop.

Step 2 — Start via systemd:
  sudo systemctl start roostoo-bot.service

Step 3 — Check status:
  sudo systemctl status roostoo-bot.service --no-pager
Expected: Active: active (running)

Step 4 — Follow logs for 120 seconds:
  journalctl -u roostoo-bot.service -f

Type "service-running" when systemctl shows active and no exceptions in first 30 seconds.
════════════════════════════════════════
```

### After all tasks complete

Create `.planning/phases/08-ec2-deployment/08-02-SUMMARY.md` using the template in the plan's `<output>` section.

Then commit: `docs(08-02): complete ec2-deploy-verify plan`

Update STATE.md and ROADMAP.md (Phase 8 complete → milestone complete).

### Hard deadline

Round 1 keys (Task 4) must be active by **Mar 21 8PM SGT = Mar 21 12:00 UTC**.

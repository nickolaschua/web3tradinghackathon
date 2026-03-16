#!/bin/bash
# Roostoo Trading Bot - AL2023 Bootstrap Script
#
# One-shot setup script for fresh EC2 instances running Amazon Linux 2023.
# Run once after first SSH to a new instance.
#
# IMPORTANT: Edit REPO_URL before running.
# On AL2023, `python3` is 3.9 — this script explicitly uses `python3.11` everywhere.
#

set -e  # exit on first error

REPO_URL="REPLACE_WITH_YOUR_REPO_URL"  # placeholder; user edits before running
BOT_DIR="/home/ec2-user/bot"
VENV_DIR="${BOT_DIR}/venv"

echo "=== [1/6] Updating system packages ==="
sudo dnf update -y

echo "=== [2/6] Installing Python 3.11 and git ==="
sudo dnf install python3.11 python3.11-pip git -y

echo "=== [3/6] Cloning repo ==="
cd /home/ec2-user
git clone "$REPO_URL" bot
cd bot

echo "=== [4/6] Creating Python 3.11 venv (explicitly python3.11, NOT python3 which is 3.9) ==="
python3.11 -m venv "$VENV_DIR"
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r requirements.txt

echo "=== [5/6] Creating .env from template ==="
cp .env.example .env
chmod 600 .env
echo ""
echo "ACTION REQUIRED: Edit .env with your actual keys:"
echo "  nano ${BOT_DIR}/.env"
echo "  Set ROOSTOO_API_KEY and ROOSTOO_SECRET to your TESTING keys for smoke test"
echo "  Format: bare KEY=VALUE (no quotes, no export prefix)"
echo ""

echo "=== [6/6] Installing systemd service ==="
sudo cp deploy/roostoo-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable roostoo-bot.service
echo "Service installed and enabled. Start with: sudo systemctl start roostoo-bot.service"
echo ""
echo "=== Bootstrap complete ==="

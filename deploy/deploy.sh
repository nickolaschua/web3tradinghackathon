#!/bin/bash
# Roostoo Trading Bot - Update Deploy Script
#
# Run from repo root on LOCAL machine to push updates to EC2 instance.
# Usage: ./deploy/deploy.sh <ec2-instance-ip>
#
# Steps:
# 1. SSH to EC2 instance
# 2. git pull latest code
# 3. pip install updated dependencies
# 4. systemctl restart bot service
# 5. verify service is running

set -e

EC2_IP="${1:-}"
if [ -z "$EC2_IP" ]; then
  echo "Usage: ./deploy/deploy.sh <ec2-instance-ip>"
  exit 1
fi

EC2_HOST="ec2-user@${EC2_IP}"
BOT_DIR="/home/ec2-user/bot"
VENV_DIR="${BOT_DIR}/venv"

echo "Deploying to ${EC2_HOST}..."
ssh "$EC2_HOST" "
  set -e
  cd ${BOT_DIR}
  git pull origin main
  ${VENV_DIR}/bin/pip install -r requirements.txt
  sudo systemctl restart roostoo-bot.service
  echo 'Waiting 3s for service to start...'
  sleep 3
  sudo systemctl status roostoo-bot.service --no-pager
"
echo "Deploy complete."

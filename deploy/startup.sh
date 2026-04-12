#!/bin/bash
set -e

LOG_TAG="kraken-bot"
logger -t $LOG_TAG "=== VM startup beginning ==="

# ── System packages ──
apt-get update -qq
apt-get install -y -qq git python3 python3-pip python3-venv > /dev/null 2>&1
logger -t $LOG_TAG "System packages installed"

# ── Python venv (BEFORE cloning repo, to avoid name conflicts) ──
if [ ! -d "/opt/kraken-env" ]; then
    python3 -m venv /opt/kraken-env
    logger -t $LOG_TAG "Python venv created"
fi

# ── Pull GitHub token from Secret Manager ──
GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token)
REPO_URL="https://${GITHUB_TOKEN}@github.com/data2dev/project_kraken.git"

# ── Clone or update repo ──
REPO_DIR="/opt/kraken"
if [ -d "$REPO_DIR/.git" ]; then
    cd $REPO_DIR
    git remote set-url origin $REPO_URL
    git pull --ff-only
    logger -t $LOG_TAG "Repo updated"
else
    git clone $REPO_URL $REPO_DIR
    logger -t $LOG_TAG "Repo cloned"
fi
cd $REPO_DIR

# ── Install Python dependencies ──
source /opt/kraken-env/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
logger -t $LOG_TAG "Python dependencies installed"

# ── Pull Kraken secrets from Secret Manager ──
export KRAKEN_API_KEY=$(gcloud secrets versions access latest --secret=kraken-api-key)
export KRAKEN_API_SECRET=$(gcloud secrets versions access latest --secret=kraken-api-secret)

if [ -z "$KRAKEN_API_KEY" ] || [ -z "$KRAKEN_API_SECRET" ]; then
    logger -t $LOG_TAG "ERROR: Failed to retrieve Kraken secrets from Secret Manager"
    exit 1
fi
logger -t $LOG_TAG "Secrets loaded from Secret Manager"

# ── Create directories ──
mkdir -p $REPO_DIR/data_store $REPO_DIR/models $REPO_DIR/logs

# ── Write systemd service ──
cat > /etc/systemd/system/kraken-bot.service << SVCEOF
[Unit]
Description=MSIF-OEM Kraken Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/kraken
Environment="PATH=/opt/kraken-env/bin:/usr/bin:/bin"
Environment="KRAKEN_API_KEY=${KRAKEN_API_KEY}"
Environment="KRAKEN_API_SECRET=${KRAKEN_API_SECRET}"
ExecStart=/opt/kraken-env/bin/python main.py
Restart=always
RestartSec=30
StandardOutput=append:/opt/kraken/logs/bot.log
StandardError=append:/opt/kraken/logs/bot.log

[Install]
WantedBy=multi-user.target
SVCEOF

# ── Start the service ──
systemctl daemon-reload
systemctl enable kraken-bot
systemctl restart kraken-bot
logger -t $LOG_TAG "=== kraken-bot service started ==="

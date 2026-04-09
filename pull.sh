#!/bin/bash
# Deploy latest code from GitHub and restart the bot.
#
# Run from your local machine:
#   gcloud compute ssh kraken-bot --zone=us-central1-a --command="bash /opt/kraken/deploy/pull.sh"

set -e
cd /opt/kraken

echo "Fetching GitHub token..."
GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token)
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/data2dev/project_kraken.git"

echo "Pulling latest from GitHub..."
git pull --ff-only

echo "Restarting bot..."
systemctl restart kraken-bot

sleep 2
systemctl status kraken-bot --no-pager

echo ""
echo "Done. Tail logs with: journalctl -u kraken-bot -f"

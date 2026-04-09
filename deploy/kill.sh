#!/bin/bash
# Kill switch for the Kraken bot.
#
# Usage from your LOCAL machine:
#
#   Stop the bot (VM stays running):
#     gcloud compute ssh kraken-bot --zone=us-central1-a --command="bash /opt/kraken/deploy/kill.sh"
#
#   Stop the bot AND the VM (stops billing for compute):
#     gcloud compute ssh kraken-bot --zone=us-central1-a --command="bash /opt/kraken/deploy/kill.sh --vm"
#
#   Or just stop the VM directly (kills everything):
#     gcloud compute instances stop kraken-bot --zone=us-central1-a

echo "=== KILL SWITCH ==="

echo "Stopping kraken-bot service..."
systemctl stop kraken-bot
systemctl disable kraken-bot

echo "Killing any remaining python processes..."
pkill -f "python main.py" 2>/dev/null
pkill -f "python backfill.py" 2>/dev/null
pkill -f "python train.py" 2>/dev/null

echo "Status:"
systemctl status kraken-bot --no-pager 2>/dev/null || echo "  Service stopped"

RUNNING=$(pgrep -f "python main.py" 2>/dev/null)
if [ -z "$RUNNING" ]; then
    echo "  No bot processes running"
else
    echo "  WARNING: processes still alive: $RUNNING"
    echo "  Force killing..."
    kill -9 $RUNNING 2>/dev/null
fi

echo ""
echo "Bot is dead."

if [ "$1" = "--vm" ]; then
    echo "Shutting down VM in 5 seconds..."
    sleep 5
    shutdown -h now
fi

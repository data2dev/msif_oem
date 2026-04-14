# Live tail (stays open, shows new lines as they appear)
gcloud compute ssh kraken-bot --zone=us-central1-b --command="tail -f /opt/kraken/logs/bot.log"

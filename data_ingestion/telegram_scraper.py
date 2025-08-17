"""
Telegram Scraper with config-file authentication.
Now scrapes only 5 channels for focused data collection.
"""

from telethon.sync import TelegramClient
from telethon.errors import SessionPasswordNeededError
import yaml
import json
from pathlib import Path
import random

# Original channel list (kept for reference)
ALL_CHANNELS = [
    "@ZemenExpress", "@nevacomputer", "@meneshayeofficial", "@ethio_brand_collection",
    "@Leyueqa", "@sinayelj", "@Shewabrand", "@helloomarketethiopia", "@modernshoppingcenter",
    "@qnashcom", "@Fashiontera", "@kuruwear", "@gebeyaadama", "@MerttEka", "@forfreemarket",
    "@classybrands", "@marakibrand", "@aradabrand2", "@marakisat2", "@belaclassic", "@AwasMart"
]

# Select 5 random channels (or choose specific ones)
SELECTED_CHANNELS = random.sample(ALL_CHANNELS, 5)  
# Alternatively, manually pick 5:
# SELECTED_CHANNELS = ["@ZemenExpress", "@helloomarketethiopia", "@qnashcom", "@Fashiontera", "@gebeyaadama"]

def scrape_channel(client, channel, limit=100):
    """Scrape messages from a single channel"""
    data = []
    for msg in client.iter_messages(channel, limit=limit):
        data.append({
            "channel": channel,
            "message_id": msg.id,
            "date": str(msg.date),
            "text": msg.text,
            "views": msg.views
        })
    return data

def main():
    # Read from config file
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)["telegram"]

    api_id = cfg["api_id"]
    api_hash = cfg["api_hash"]
    output = cfg["output"]

    client = TelegramClient("session", api_id, api_hash)
    client.start()

    all_data = []
    print(f"Scraping these 5 channels: {SELECTED_CHANNELS}")
    
    for channel in SELECTED_CHANNELS:
        print(f"Scraping {channel}...")
        try:
            data = scrape_channel(client, channel, limit=100)
            all_data.extend(data)
            print(f"  → Got {len(data)} messages")
        except Exception as e:
            print(f"Failed on {channel}: {e}")

    # Save results
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\nFinished! Saved {len(all_data)} messages → {output}")
    client.disconnect()

if __name__ == "__main__":
    main()
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')

def send_to_telegram(message, to):
    if to not in ['owner', 'group']:
        print("Hiba történt: a 'to' paraméter nem megfelelő (owner vagy group).")
    else:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        CHAT_ID = os.getenv('GROUP_CHAT_ID') if to == 'group' else os.getenv('OWNER_CHAT_ID')
        payload = {
            "chat_id": CHAT_ID,
            "text": message
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("Sikeresen elküldve Telegramra!")
        else:
            print(f"Hiba történt: {response.status_code} - {response.text}")

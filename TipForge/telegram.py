import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')

def send_to_telegram(message, to, topic_id=None):
    """
    Üzenet küldése Telegramra
    
    Args:
        message: Küldendő üzenet
        to: 'owner', 'group', vagy 'channel'
        topic_id: Topic ID (opcionális, csak group-oknál)
    """
    if to not in ['owner', 'group', 'channel']:
        print("Hiba történt: a 'to' paraméter nem megfelelő (owner, group vagy channel).")
        return
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    # Chat ID meghatározása
    if to == 'group':
        CHAT_ID = os.getenv('GROUP_CHAT_ID')
    elif to == 'channel':
        CHAT_ID = os.getenv('CHANNEL_CHAT_ID')
    else:
        CHAT_ID = os.getenv('OWNER_CHAT_ID')
    
    # Payload összeállítása
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    
    # Ha topic ID van megadva és group a cél
    if topic_id is not None and to == 'group':
        payload["message_thread_id"] = topic_id
    
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("Sikeresen elküldve Telegramra!")
    else:
        print(f"Hiba történt: {response.status_code} - {response.text}")

# Példa használatra:
# send_to_telegram("Üzenet", "group", topic_id=12)  # Konkrét topic-ba
# send_to_telegram("Üzenet", "group")                # General topic-ba
# send_to_telegram("Üzenet", "owner")                # Owner-nek
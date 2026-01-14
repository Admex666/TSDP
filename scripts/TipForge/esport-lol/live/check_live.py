
from riot_api import RiotEsportsAPI
import json

def check_live():
    api = RiotEsportsAPI()
    live_events = api.get_live()
    print(f"Live events found: {len(live_events)}")
    for event in live_events:
        print(f"Event ID: {event.get('id')}, Match: {event.get('match', {}).get('id')}")
        match = event.get('match', {})
        games = match.get('games', [])
        for game in games:
            print(f"  Game ID: {game.get('id')}, State: {game.get('state')}")

if __name__ == "__main__":
    check_live()

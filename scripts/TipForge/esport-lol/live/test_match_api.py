
from riot_api import RiotEsportsAPI
import json

def test_match(event_id):
    api = RiotEsportsAPI()
    print(f"Fetching details for event: {event_id}")
    event_details = api.get_event_details(event_id)
    
    if not event_details:
        print("Could not fetch event details. Trying as game_id directly...")
        stats = api.get_latest_match_state(event_id)
        if stats:
             print("Successfully fetched stats using ID as game_id:")
             print(json.dumps(stats, indent=2))
        else:
             print("Could not fetch stats as game_id either.")
        return

    print("Event Details successfully fetched.")
    
    match = event_details.get("match", {})
    games = match.get("games", [])
    
    if not games:
        print("No games found in this event.")
        return
        
    for i, game in enumerate(games):
        game_id = game.get("id")
        state = game.get("state")
        print(f"Game {i+1}: ID={game_id}, State={state}")
        
        if state == "inProgress" or state == "completed":
            print(f"Fetching stats for game {game_id}...")
            stats = api.get_latest_match_state(game_id)
            if stats:
                print(json.dumps(stats, indent=2))
            else:
                print(f"No stats returned for game {game_id}")

if __name__ == "__main__":
    test_match("115746408286951071")

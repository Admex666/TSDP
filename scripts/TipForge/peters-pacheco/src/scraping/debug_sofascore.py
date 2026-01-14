from src.scraping.sofascore import get_events_for_round, scrape_sofascore
import json

def debug_season_fetch(tournament_id, season_id):
    print(f"DEBUG: Testing fetch for Tournament {tournament_id}, Season {season_id}")
    
    # Test Round 1
    print("Fetching Round 1...")
    events = get_events_for_round(tournament_id, season_id, 1)
    
    if not events:
        print("FAIL: No events returned for Round 1.")
        # Check raw URL manually to see response code?
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/1"
        print(f"Test URL: {url}")
        
        # Try raw scrape with verbose error
        res = scrape_sofascore(url)
        print(f"Raw response: {res}")
    else:
        print(f"SUCCESS: Found {len(events)} events in Round 1.")
        print("Sample Event:")
        print(json.dumps(events[0], indent=2))
        
    
    print("\n--- User Suggested Test (Simple Chrome) ---")
    try:
        from curl_cffi import requests as cureq
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/1"
        print(f"Testing URL: {url}")
        response = cureq.get(url=url, impersonate="chrome", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            # print(response.json())
        else:
            print("Failed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test 41886 (22/23 Season)
    debug_season_fetch(17, 41886)

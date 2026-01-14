
from riot_api import RiotEsportsAPI
import json

import logging
logging.basicConfig(level=logging.DEBUG)

def debug_frames(game_id):
    api = RiotEsportsAPI()
    last_time = None
    
    # Try fetching a few pages
    for page in range(5):
        print(f"--- Page {page+1} ---")
        window = api.get_live_stats_window(game_id, starting_time=last_time)
        if not window:
            print(f"Window response is None or empty. Status: {api.get_live_stats_window(game_id, starting_time=last_time)}") 
            break
            
        if not window.get("frames"):
            print(f"Window has no frames. Keys: {window.keys()}")
            # Break only if we really can't proceed
            break
            
        frames = window["frames"]
        print(f"Frames received: {len(frames)}")
        
        # Print info from the last frame of the batch
        last_frame = frames[-1]
        ts = last_frame.get("rfc460Timestamp")
        blue = last_frame.get("blueTeam", {})
        red = last_frame.get("redTeam", {})
        print(f"Last Frame TS: {ts}")
        print(f"Blue Gold: {blue.get('totalGold')}, Red Gold: {red.get('totalGold')}")
        
        last_time = ts

    print("\n--- Testing forcing a recent timestamp ---")
    # Current time roughly - 5 mins? No, let's try a bit after the last seen frame.
    # Last seen was 16:20:16Z. Let's try 16:21:00Z
    forced_time = "2026-01-14T16:21:00.000Z"
    print(f"Requesting from {forced_time}")
    window = api.get_live_stats_window(game_id, starting_time=forced_time)
    if window and window.get("frames"):
        frames = window["frames"]
        print(f"Success! Got {len(frames)} frames.")
        last_frame = frames[-1]
        ts = last_frame.get("rfc460Timestamp")
        blue = last_frame.get("blueTeam", {})
        red = last_frame.get("redTeam", {})
        print(f"Forced Frame Last TS: {ts}")
        print(f"Blue Gold: {blue.get('totalGold')}, Red Gold: {red.get('totalGold')}")
    else:
        print("Still no frames with forced timestamp.")
        
    print("\n--- Testing get_live_stats_details ---")
    details = api.get_live_stats_details(game_id)
    if details and details.get("frames"):
        frames = details["frames"]
        print(f"Details frames: {len(frames)}")
        last_frame = frames[-1]
    print("\n--- Testing JS-like timestamp logic ---")
    from datetime import datetime, timedelta, timezone
    
    # 1. Get current UTC time
    now = datetime.now(timezone.utc)
    
    # 2. Round down to nearest 10 seconds
    seconds = now.second
    rounded_seconds = seconds - (seconds % 10)
    
    # 3. Create new time with rounded seconds
    # Note: we also set microseconds to 0
    dt_rounded = now.replace(second=rounded_seconds, microsecond=0)
    
    # 4. Subtract delay (60s minimum)
    # The JS code starts with u=60. 
    delay_seconds = 60
    dt_final = dt_rounded - timedelta(seconds=delay_seconds)
    
    # 5. Format to ISO
    # JS toISOString() format: 2026-01-14T16:21:00.000Z
    formatted_time = dt_final.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    
    print(f"Calculated startingTime: {formatted_time} (Now: {now})")
    
    window = api.get_live_stats_window(game_id, starting_time=formatted_time)
    if window:
        if window.get("frames"):
            frames = window["frames"]
            print(f"Success! Got {len(frames)} frames with JS logic.")
            last_frame = frames[-1]
            ts = last_frame.get("rfc460Timestamp")
            blue = last_frame.get("blueTeam", {})
            red = last_frame.get("redTeam", {})
            print(f"JS Logic Frame Last TS: {ts}")
            print(f"Blue Gold: {blue.get('totalGold')}, Red Gold: {red.get('totalGold')}")
        else:
            print(f"Window retrieved but no frames. Keys: {window.keys()}")
    else:
        print("Failed to retrieve window with JS logic.")

if __name__ == "__main__":
    debug_frames("115746408286951071")

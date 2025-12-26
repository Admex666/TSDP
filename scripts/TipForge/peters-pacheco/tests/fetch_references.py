import os
import sys
import time

# Add src to path
sys.path.append(os.getcwd())

from src.scraping.fbref_loader import FBrefDataLoader

def fetch_refs():
    loader = FBrefDataLoader(data_dir="data")
    # URLs
    urls = {
        "schedule.html": "https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures",
        "match.html": "https://fbref.com/en/matches/3a6836b4/Burnley-Manchester-City-August-11-2023-Premier-League",
        "player_summary.html": "https://fbref.com/en/players/1f44ac21/matchlogs/2023-2024/summary/Erling-Haaland-Match-Logs",
        "player_keepers.html": "https://fbref.com/en/players/3bb7b8b4/matchlogs/2023-2024/keeper/Ederson-Match-Logs" # Changed to singular 'keeper'
    }
    
    ref_dir = os.path.join("data", "reference")
    os.makedirs(ref_dir, exist_ok=True)
    
    for filename, url in urls.items():
        print(f"Fetching {filename}...")
        try:
            # We use _fetch_url_selenium directly to get HTML text
            html = loader._fetch_url_selenium(url)
            
            path = os.path.join(ref_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Saved {path}")
            
            # Be polite
            time.sleep(5) 
        except Exception as e:
            print(f"Error fetching {filename}: {e}")

if __name__ == "__main__":
    fetch_refs()

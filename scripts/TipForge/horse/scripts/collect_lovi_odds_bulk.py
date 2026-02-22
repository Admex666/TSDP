import sys
import os
import json
import time
from datetime import datetime
# Add current directory to path so research module is found
sys.path.append(os.getcwd())

from research.scrape_lovi_historical import LoviScraper

def collect_all_odds():
    # 1. Lekérjük a meglévő eredmények dátumait
    results_path = 'data/historical_results_combined.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Kigyűjtjük az egyedi dátumokat
    # A format: YYYY-MM-DD
    target_dates = sorted(list(set(r.get('race_date') for r in data.get('races', []) if r.get('race_date'))))
    
    if not target_dates:
        print("No dates found in historical results.")
        return

    output_path = 'data/historical_odds_lovi.json'
    
    # 2. Betöltjük a már meglévő oddsokat (ha van), hogy ne töltsük le újra
    collected_odds = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # Dátum alapú tárolás a keresés megkönnyítésére
                for entry in existing_data.get('days', []):
                    collected_odds[entry['date']] = entry
        except Exception as e:
            print(f"Warning: Could not load existing odds: {e}")

    scraper = LoviScraper()
    
    new_days_count = 0
    
    # 3. Végigmegyünk a dátumokon
    for date_str in target_dates:
        if date_str in collected_odds:
            # print(f"Skipping {date_str}, already collected.")
            continue
            
        print(f"\n>>> Scraping odds for: {date_str}")
        day_result = scraper.scrape_date(date_str)
        
        if day_result and day_result.get('races'):
            collected_odds[date_str] = day_result
            new_days_count += 1
            
            # Mentés minden sikeres nap után (inkrementális mentés biztonságosabb)
            final_data = {
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "total_days": len(collected_odds)
                },
                "days": list(collected_odds.values())
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
                
            print(f"Progress: {len(collected_odds)}/{len(target_dates)} days collected.")
        else:
            print(f"No Hungarian races found for {date_str} or scrape failed.")
            # Bejelöljük üresnek, hogy legközelebb ne próbálkozzon feleslegesen
            collected_odds[date_str] = {"date": date_str, "races": [], "note": "No races found"}

        time.sleep(2) # Kicsit több szünet a tömeges letöltésnél

    print(f"\nFinished! Collected {new_days_count} new days.")
    print(f"Total days in file: {len(collected_odds)}")

if __name__ == "__main__":
    collect_all_odds()

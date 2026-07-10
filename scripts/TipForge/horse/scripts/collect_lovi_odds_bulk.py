import sys
import os
import json
import time
from datetime import datetime
from tqdm import tqdm

# Add current directory to path so research module is found
sys.path.append(os.getcwd())

from research.scrape_lovi_historical import LoviScraper

def collect_all_odds():
    results_path = 'data/historical_results_combined.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect unique dates in YYYY-MM-DD
    target_dates = sorted(list(set(r.get('race_date') for r in data.get('races', []) if r.get('race_date'))))
    
    if not target_dates:
        print("No dates found in historical results.")
        return

    output_path = 'data/historical_odds_lovi.json'
    
    # Load already collected odds to skip them
    collected_odds = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for entry in existing_data.get('days', []):
                    collected_odds[entry['date']] = entry
        except Exception as e:
            print(f"Warning: Could not load existing odds: {e}")

    # Filter dates that still need scraping
    remaining_dates = [d for d in target_dates if d not in collected_odds]
    
    print(f"Total dates in results: {len(target_dates)}")
    print(f"Already scraped odds days: {len(target_dates) - len(remaining_dates)}")
    print(f"Remaining days to scrape odds: {len(remaining_dates)}")

    if not remaining_dates:
        print("Odds for all dates are already collected.")
        return

    scraper = LoviScraper()
    new_days_count = 0
    
    # Iterate with tqdm progress bar
    for date_str in tqdm(remaining_dates, desc="Scraping BetLovi odds"):
        day_result = scraper.scrape_date(date_str)
        
        if day_result and day_result.get('races'):
            collected_odds[date_str] = day_result
            new_days_count += 1
            
            # Save progress incrementally after each day
            final_data = {
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "total_days": len(collected_odds)
                },
                "days": list(collected_odds.values())
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
        else:
            # Mark date as empty to avoid scraping again in next runs
            collected_odds[date_str] = {"date": date_str, "races": [], "note": "No races found or scrape failed"}
            final_data = {
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "total_days": len(collected_odds)
                },
                "days": list(collected_odds.values())
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)

        time.sleep(2) # rate limit

    print(f"\nFinished! Collected {new_days_count} new days.")
    print(f"Total days in file: {len(collected_odds)}")

if __name__ == "__main__":
    collect_all_odds()

import re
import json
import os
import requests
import time
from datetime import datetime
from tqdm import tqdm

def get_racing_days(year, discipline="trotting"):
    """Fetches the list of racing days for a specific year and discipline."""
    url = f"https://mla.kincsempark.hu/racing-days/{discipline}/{year}/"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Extract the racing_days JSON from the script tag
        pattern = r"var racing_days = (\[.*?\]);"
        match = re.search(pattern, response.text)
        if match:
            return json.loads(match.group(1))
        else:
            print(f"No racing days found in JS for {year}")
            return []
    except Exception as e:
        print(f"Error fetching racing days for {year}: {e}")
        return []

def parse_results_page(date, discipline="trotting"):
    """Parses a single result page and extracts all race data."""
    url = f"https://mla.kincsempark.hu/results/{discipline}/{date}/"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        pattern = r'races_table_divs\[".*?"\]\s*=\s*(\{.*?\});'
        matches = re.findall(pattern, response.text, re.DOTALL)
        
        races = []
        for match in matches:
            try:
                races.append(json.loads(match))
            except:
                continue
        return races
    except Exception as e:
        print(f"\nError fetching results for {date}: {e}")
        return []

def crawl_historical(years, output_file):
    """Crawls multiple years incrementally and saves results to a consolidated file."""
    all_results = {
        "metadata": {
            "last_updated": datetime.now().isoformat(),
            "years_crawled": years
        },
        "races": []
    }
    
    # 1. Load existing results to skip them and prevent duplicate requests
    existing_dates = set()
    existing_race_ids = set()
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                all_results["races"] = existing_data.get("races", [])
                
                # Update metadata years list
                saved_years = existing_data.get("metadata", {}).get("years_crawled", [])
                all_results["metadata"]["years_crawled"] = sorted(list(set(saved_years + years)))
                
                for r in all_results["races"]:
                    if r.get("race_date"):
                        existing_dates.add(r["race_date"])
                    if r.get("id"):
                        existing_race_ids.add(r["id"])
            print(f"Loaded {len(all_results['races'])} existing races from {output_file}.")
            print(f"Already crawled dates count: {len(existing_dates)}")
        except Exception as e:
            print(f"Warning: Could not load existing file {output_file}: {e}. Starting fresh.")

    # 2. Get finished days for all target years
    all_days = []
    for year in years:
        days = get_racing_days(year)
        finished_days = [d for d in days if d.get("results")]
        all_days.extend(finished_days)
        
    # Filter out already crawled days
    remaining_days = [d for d in all_days if d.get("date") not in existing_dates]
    
    print(f"Total finished days found: {len(all_days)}")
    print(f"Already crawled days: {len(all_days) - len(remaining_days)}")
    print(f"Remaining days to crawl: {len(remaining_days)}")
    
    if not remaining_days:
        print("All target dates are already crawled.")
        return

    # 3. Crawl remaining days with progress bar
    for day in tqdm(remaining_days, desc="Crawling Kincsem Park days"):
        date = day["date"]
        races = parse_results_page(date)
        
        if races:
            new_races = []
            for r in races:
                r["race_date"] = date
                # Skip duplicate races by ID
                if r.get("id") and r["id"] in existing_race_ids:
                    continue
                new_races.append(r)
                if r.get("id"):
                    existing_race_ids.add(r["id"])
            
            all_results["races"].extend(new_races)
            
            # Save progress incrementally on hard drive after each crawled day
            all_results["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
                
        time.sleep(0.5)
        
    print(f"\nCrawling complete. Total races in consolidated database: {len(all_results['races'])}")

if __name__ == "__main__":
    # Crawl 2020-2025 historical data
    crawl_historical(["2020", "2021", "2022", "2023", "2024", "2025"], "data/historical_results_combined.json")

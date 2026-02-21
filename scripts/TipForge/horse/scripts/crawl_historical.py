import re
import json
import os
import requests
import time
from datetime import datetime

def get_racing_days(year, discipline="trotting"):
    """
    Fetches the list of racing days for a specific year and discipline.
    """
    url = f"https://mla.kincsempark.hu/racing-days/{discipline}/{year}/"
    print(f"Fetching racing days for {year} from: {url}")
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
        print(f"Error fetching racing days: {e}")
        return []

def parse_results_page(date, discipline="trotting"):
    """
    Parses a single result page and extracts all race data.
    """
    url = f"https://mla.kincsempark.hu/results/{discipline}/{date}/"
    print(f"  Fetching results for {date}...")
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
        print(f"    Error fetching results for {date}: {e}")
        return []

def crawl_historical(years, output_file):
    """
    Crawls multiple years and saves all results to a consolidated file.
    """
    all_results = {
        "metadata": {
            "last_updated": datetime.now().isoformat(),
            "years_crawled": years
        },
        "races": []
    }
    
    for year in years:
        days = get_racing_days(year)
        # Filter for days that are already finished (status 2 usually means finished)
        # or simply look for results field
        finished_days = [d for d in days if d.get("results")]
        
        print(f"Found {len(finished_days)} finished racing days in {year}")
        
        for day in finished_days:
            date = day["date"]
            races = parse_results_page(date)
            if races:
                # Add date to each race object for easier filtering later
                for r in races:
                    r["race_date"] = date
                all_results["races"].extend(races)
            
            # Rate limiting
            time.sleep(0.5)
            
            # Save progress incrementally
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
                
    print(f"\nCrawling complete. Total races collected: {len(all_results['races'])}")
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    # Crawl 2025 and 2024 for training data
    crawl_historical(["2025", "2024"], "data/historical_results_combined.json")

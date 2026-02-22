import json
from datetime import datetime

def get_dates():
    with open('data/historical_results_combined.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dates = sorted(list(set(r.get('race_date') for r in data.get('races', []) if r.get('race_date'))))
    print(f"Total Unique Dates: {len(dates)}")
    if dates:
        print(f"First Date: {dates[0]}")
        print(f"Last Date: {dates[-1]}")
    return dates

if __name__ == "__main__":
    get_dates()

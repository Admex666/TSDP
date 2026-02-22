import json
import os

def check_mismatches():
    results_path = 'data/historical_results_combined.json'
    odds_path = 'data/historical_odds_lovi.json'

    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    with open(odds_path, 'r', encoding='utf-8') as f:
        odds_data = json.load(f)

    res_dates = set(r.get('race_date') for r in results_data.get('races', []))
    odds_dates = set(d.get('date') for d in odds_data.get('days', []))
    
    common_dates = res_dates.intersection(odds_dates)
    print(f"Common dates: {len(common_dates)}")
    
    if common_dates:
        sample_date = sorted(list(common_dates))[-1]
        print(f"Analyzing sample date: {sample_date}")
        
        res_races = [r for r in results_data.get('races', []) if r.get('race_date') == sample_date]
        odds_day = next(d for d in odds_data.get('days', []) if d.get('date') == sample_date)
        odds_races = odds_day.get('races', [])
        
        print(f"\nResults races on {sample_date}:")
        for r in res_races:
            print(f"  Time: {r.get('start')} | Name: {r.get('race_name')}")
            
        print(f"\nOdds races on {sample_date}:")
        for r in odds_races:
            print(f"  Time: {r.get('meta', {}).get('start')} | Name: {r.get('title')}")

if __name__ == "__main__":
    check_mismatches()

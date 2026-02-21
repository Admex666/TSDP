import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    """Loads all consolidated JSON files and pre-normalizes dates."""
    results = json.load(open("data/historical_results_combined.json", encoding="utf-8"))
    horses = json.load(open("data/all_horses.json", encoding="utf-8"))["data"]
    drivers = json.load(open("data/all_drivers.json", encoding="utf-8"))["data"]
    
    # Pre-normalize career dates
    for h_id in horses:
        for r in horses[h_id].get("results", []):
            if "date" in r:
                r["date"] = r["date"].replace(".", "-")
    for d_id in drivers:
        for r in drivers[d_id].get("results", []):
            if "date" in r:
                r["date"] = r["date"].replace(".", "-")
                
    return results, horses, drivers
def parse_km_time(time_str):
    """Converts '1:18.4' format to float (seconds)."""
    if not time_str or time_str in ["0:00.0", "gal.", "tü.", "rnyak."]:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except:
        return None
def calculate_point_in_time_stats(career_history, before_date):
    """
    Calculates horse/driver stats based ONLY on races occurring before the target date.
    """
    if not career_history:
        return {"win_rate": 0.0, "avg_km_time": None, "runs": 0}
    
    target_date = before_date.replace(".", "-")
    past_races = [r for r in career_history if r.get("date") and r["date"] < target_date]
    
    if not past_races:
        return {"win_rate": 0.05, "avg_km_time": None, "runs": 0}

    runs = len(past_races)
    wins = sum(1 for r in past_races if r.get("rank") == "I.")
    
    km_times = [parse_km_time(r.get("km_time")) for r in past_races[-5:]]
    km_times = [t for t in km_times if t is not None]
    avg_km = sum(km_times) / len(km_times) if km_times else None
    
    return {
        "win_rate": wins / runs,
        "avg_km_time": avg_km,
        "runs": runs
    }

def engineer_features():
    results, horse_data, driver_data = load_data()
    
    rows = []
    print(f"Processing {len(results['races'])} races with point-in-time logic...")

    for race in results["races"]:
        race_id = race.get("race_id")
        date_str = race.get("race_date")
        try:
            race_dist = float(race.get("distance", "1900").replace("A", "").replace("G", ""))
        except:
            race_dist = 1900.0
        
        for p in race.get("participants", []):
            h_id = str(p.get("id"))
            d_id = str(p.get("driver_jockey_id"))
            
            # Target
            rank = p.get("rank")
            target_win = 1 if rank == "I." else 0
            
            # Horse PiT stats
            h_pit = calculate_point_in_time_stats(horse_data.get(h_id, {}).get("results", []), date_str)
            
            # Driver PiT stats
            d_pit = calculate_point_in_time_stats(driver_data.get(d_id, {}).get("results", []), date_str)
            
            row = {
                "race_id": race_id,
                "date": date_str,
                "distance": race_dist,
                "horse_id": h_id,
                "driver_id": d_id,
                "horse_win_rate": h_pit["win_rate"],
                "horse_avg_km": h_pit["avg_km_time"],
                "horse_runs": h_pit["runs"],
                "driver_win_rate": d_pit["win_rate"],
                "driver_runs": d_pit["runs"],
                "win": target_win
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df["horse_avg_km"] = df["horse_avg_km"].fillna(df["horse_avg_km"].mean())
    
    output_path = "data/training_set.csv"
    df.to_csv(output_path, index=False)
    print(f"Success. dataset with {len(df)} entries saved to {output_path}")

if __name__ == "__main__":
    if os.path.exists("data/historical_results_combined.json") and \
       os.path.exists("data/all_horses.json") and \
       os.path.exists("data/all_drivers.json"):
        engineer_features()
    else:
        print("Required JSON files for feature engineering not found yet.")

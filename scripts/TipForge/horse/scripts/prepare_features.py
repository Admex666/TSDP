import json
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime

def load_data():
    """Loads all consolidated JSON files and pre-normalizes dates."""
    base_path = r"E:\Data\TSDP\scripts\TipForge\horse\data"
    results = json.load(open(os.path.join(base_path, "historical_results_combined.json"), encoding="utf-8"))
    horses = json.load(open(os.path.join(base_path, "all_horses.json"), encoding="utf-8"))["data"]
    drivers = json.load(open(os.path.join(base_path, "all_drivers.json"), encoding="utf-8"))["data"]
    
    # Build race_id -> field_size map
    race_field_map = {}
    for race in results["races"]:
        race_field_map[str(race.get("race_id"))] = len(race.get("participants", []))

    # Pre-normalize career dates
    for h_id in horses:
        for r in horses[h_id].get("results", []):
            if "date" in r:
                r["date"] = r["date"].replace(".", "-")
    for d_id in drivers:
        for r in drivers[d_id].get("results", []):
            if "date" in r:
                r["date"] = r["date"].replace(".", "-")
                
    return results, horses, drivers, race_field_map

def parse_km_time(time_str):
    """Converts '1:18.4' format to float (seconds)."""
    if not time_str or time_str in ["0:00.0", "gal.", "tü.", "rnyak.", "diszk."]:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except:
        return None

def calculate_point_in_time_stats(career_history, before_date, race_field_map):
    """
    Calculates advanced horse/driver stats based ONLY on races occurring before the target date.
    """
    default_stats = {
        "win_rate": 0.05, "top_3_rate": 0.15, "avg_percentile": 0.5, 
        "avg_speed": None, "total_prize": 0, "days_since_last": 100,
        "win_rate_l5": 0.05, "top_3_rate_l5": 0.15, "avg_percentile_l5": 0.5, "avg_speed_l5": None,
        "best_speed_life": None, "speed_ratio": 1.0, "points_l5": 0, "top3_l3": 0
    }
    
    if not career_history:
        return default_stats
    
    target_date = before_date.replace(".", "-")
    try:
        dt_target = datetime.strptime(target_date, "%Y-%m-%d")
    except:
        dt_target = datetime.now()

    past_races = [r for r in career_history if r.get("date") and r["date"] < target_date]
    
    if not past_races:
        return default_stats

    def get_rank_num(r):
        p = r.get("placement") or r.get("rank") or "10."
        if p.startswith("I.") or p.startswith("1."): return 1
        if p.startswith("II.") or p.startswith("2."): return 2
        if p.startswith("III.") or p.startswith("3."): return 3
        if p.startswith("IV.") or p.startswith("4."): return 4
        if p.startswith("V.") or p.startswith("5."): return 5
        try:
            return int(p.split(".")[0])
        except:
            return 10

    def get_speed(r):
        dist = float(r.get("distance", 1900))
        km_time = parse_km_time(r.get("km_time"))
        if km_time and km_time > 0:
            return 1000.0 / km_time # speed in m/s
        return None

    def get_percentile(r):
        rid = r.get("race_id")
        f_size = race_field_map.get(str(rid), 10)
        rank = get_rank_num(r)
        if f_size <= 1: return 0.5
        return 1.0 - (min(rank, f_size) - 1) / (f_size - 1)

    # Life stats
    runs = len(past_races)
    wins = sum(1 for r in past_races if get_rank_num(r) == 1)
    top3 = sum(1 for r in past_races if get_rank_num(r) <= 3)
    total_prize = sum(float(r.get("prize", 0) or 0) for r in past_races)
    
    speeds = [get_speed(r) for r in past_races if get_speed(r) is not None]
    avg_speed = sum(speeds) / len(speeds) if speeds else None
    best_speed_life = max(speeds) if speeds else None
    
    percentiles = [get_percentile(r) for r in past_races]
    avg_percentile = sum(percentiles) / len(percentiles) if percentiles else 0.5
    
    # Days since last
    last_race_date = past_races[-1].get("date", "").replace(".", "-")
    try:
        dt_last = datetime.strptime(last_race_date, "%Y-%m-%d")
        days_since = (dt_target - dt_last).days
    except:
        days_since = 100
    
    # L5 stats
    l5 = past_races[-5:]
    runs_l5 = len(l5)
    wins_l5 = sum(1 for r in l5 if get_rank_num(r) == 1)
    top3_l5 = sum(1 for r in l5 if get_rank_num(r) <= 3)
    speeds_l5 = [get_speed(r) for r in l5 if get_speed(r) is not None]
    avg_speed_l5 = sum(speeds_l5) / len(speeds_l5) if speeds_l5 else None
    
    # Points L5: 1st=5, 2nd=3, 3rd=2, 4th=1
    def get_points(rank):
        if rank == 1: return 5
        if rank == 2: return 3
        if rank == 3: return 2
        if rank == 4: return 1
        return 0
    points_l5 = sum(get_points(get_rank_num(r)) for r in l5)

    # L3 Top 3
    l3 = past_races[-3:]
    top3_l3 = sum(1 for r in l3 if get_rank_num(r) <= 3) / len(l3) if l3 else 0

    percentiles_l5 = [get_percentile(r) for r in l5]
    avg_percentile_l5 = sum(percentiles_l5) / len(percentiles_l5) if percentiles_l5 else 0.5
    
    speed_ratio = (avg_speed_l5 / best_speed_life) if (avg_speed_l5 and best_speed_life) else 1.0

    return {
        "win_rate": wins / runs,
        "top_3_rate": top3 / runs,
        "avg_percentile": avg_percentile,
        "avg_speed": avg_speed,
        "total_prize": total_prize,
        "days_since_last": min(max(days_since, 0), 365),
        "win_rate_l5": wins_l5 / runs_l5,
        "top_3_rate_l5": top3_l5 / runs_l5,
        "avg_percentile_l5": avg_percentile_l5,
        "avg_speed_l5": avg_speed_l5,
        "best_speed_life": best_speed_life,
        "speed_ratio": speed_ratio,
        "points_l5": points_l5,
        "top3_l3": top3_l3
    }

def engineer_features():
    results, horse_data, driver_data, race_field_map = load_data()
    
    rows = []
    print(f"Processing {len(results['races'])} races with V4 features (Age, Sex, Trainer, Reliability)...")

    # Pre-calculate Trainer Stats
    all_trainers = {}
    for race in results["races"]:
        for p in race.get("participants", []):
            tid = str(p.get("trainer_id"))
            if tid not in all_trainers:
                all_trainers[tid] = {"runs": 0, "wins": 0, "top3": 0}
            all_trainers[tid]["runs"] += 1
            if p.get("rank") in ["I.", "1."]: all_trainers[tid]["wins"] += 1
            if p.get("rank") in ["I.", "1.", "II.", "2.", "III.", "3."]: all_trainers[tid]["top3"] += 1
    
    trainer_stats_map = {
        tid: {
            "win_rate": v["wins"]/v["runs"] if v["runs"] > 0 else 0.05,
            "top3_rate": v["top3"]/v["runs"] if v["runs"] > 0 else 0.15
        } for tid, v in all_trainers.items()
    }

    track_pattern = re.compile(r'Track:\s*([^.]+)', re.IGNORECASE)
    temp_pattern = re.compile(r'Hőmérséklet:\s*([-\d]+)', re.IGNORECASE)

    pair_experience = {}

    for race in results["races"]:
        race_id = race.get("race_id")
        date_str = race.get("race_date")
        desc = race.get("description", "")
        
        track_match = track_pattern.search(desc)
        track_str = track_match.group(1).strip().lower() if track_match else "jó"
        track_val = 0
        if "puha" in track_str: track_val = 1
        elif "sáros" in track_str or "nehéz" in track_str: track_val = 2
        elif "fagyos" in track_str: track_val = 3

        temp_match = temp_pattern.search(desc)
        temp_val = float(temp_match.group(1)) if temp_match else 15.0

        try:
            race_dist = float(race.get("distance", "1900").replace("A", "").replace("G", ""))
        except:
            race_dist = 1900.0
        
        for p in race.get("participants", []):
            h_id = str(p.get("id"))
            d_id = str(p.get("driver_jockey_id"))
            t_id = str(p.get("trainer_id"))
            
            rank = p.get("rank")
            target_win = 1 if rank in ["I.", "1."] else 0
            
            pair_key = (h_id, d_id)
            hd_pair_runs = pair_experience.get(pair_key, 0)
            pair_experience[pair_key] = hd_pair_runs + 1
            
            h_history = horse_data.get(h_id, {}).get("results", [])
            h_stats = calculate_point_in_time_stats(h_history, date_str, race_field_map)
            d_stats = calculate_point_in_time_stats(driver_data.get(d_id, {}).get("results", []), date_str, race_field_map)
            
            # V4 New Features
            h_age = p.get("age", 5)
            h_sex_str = str(p.get("sex", "male")).lower()
            h_sex_val = 0 # male/mén
            if "female" in h_sex_str or "kanca" in h_sex_str: h_sex_val = 1
            elif "gelding" in h_sex_str or "herelt" in h_sex_str: h_sex_val = 2
            
            # Reliability (Gallop rate in last 10)
            past_h = [r for r in h_history if r.get("date") and r["date"] < date_str][-10:]
            gallops = sum(1 for r in past_h if "gal" in str(r.get("placement", r.get("rank", ""))).lower())
            h_gallop_rate = gallops / len(past_h) if past_h else 0
            
            # Distance Preference (current vs median winning distance)
            win_dists = [float(r.get("distance", 1900)) for r in h_history 
                         if r.get("date") and r["date"] < date_str and str(r.get("rank")).startswith(("1.", "I."))]
            avg_win_dist = sum(win_dists)/len(win_dists) if win_dists else 1900.0
            dist_diff = abs(race_dist - avg_win_dist)

            t_stats = trainer_stats_map.get(t_id, {"win_rate": 0.05, "top3_rate": 0.15})
            
            row = {
                "race_id": race_id, "date": date_str, "distance": race_dist,
                "track_quality": track_val, "temperature": temp_val,
                "horse_id": int(h_id) if h_id else None,
                "driver_id": int(d_id) if d_id else None,
                "h_age": h_age, "h_sex": h_sex_val,
                "h_win_rate": h_stats["win_rate"], "h_top_3_rate": h_stats["top_3_rate"],
                "h_avg_percentile": h_stats["avg_percentile"], "h_avg_speed": h_stats["avg_speed"],
                "h_best_speed": h_stats["best_speed_life"], "h_speed_ratio": h_stats["speed_ratio"],
                "h_win_rate_l5": h_stats["win_rate_l5"], "h_top_3_rate_l5": h_stats["top_3_rate_l5"],
                "h_avg_percentile_l5": h_stats["avg_percentile_l5"], "h_avg_speed_l5": h_stats["avg_speed_l5"],
                "h_points_l5": h_stats["points_l5"], "h_top3_l3": h_stats["top3_l3"],
                "h_gallop_rate": h_gallop_rate,
                "h_total_prize": h_stats["total_prize"], "h_days_since": h_stats["days_since_last"],
                "dist_diff": dist_diff,
                "d_win_rate": d_stats["win_rate"], "d_top_3_rate": d_stats["top_3_rate"],
                "t_win_rate": t_stats["win_rate"], "t_top3_rate": t_stats["top3_rate"],
                "hd_pair_runs": hd_pair_runs,
                "win": target_win
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["h_avg_speed", "h_best_speed"]:
        df[col] = df[col].fillna(df[col].mean())
    df["h_speed_ratio"] = df["h_speed_ratio"].fillna(1.0)
    
    output_path = r"E:\Data\TSDP\scripts\TipForge\horse\data\training_set_v4.csv"
    df.to_csv(output_path, index=False)
    print(f"Success. dataset with {len(df)} entries saved to {output_path}")

    # Export Trainer Stats for the App
    trainer_json_path = r"E:\Data\TSDP\scripts\TipForge\horse\data\trainer_stats.json"
    with open(trainer_json_path, "w", encoding="utf-8") as f:
        json.dump(trainer_stats_map, f, indent=2)
    print(f"Trainer stats exported to {trainer_json_path}")

if __name__ == "__main__":
    base_path = r"E:\Data\TSDP\scripts\TipForge\horse\data"
    if os.path.exists(os.path.join(base_path, "historical_results_combined.json")) and \
       os.path.exists(os.path.join(base_path, "all_horses.json")) and \
       os.path.exists(os.path.join(base_path, "all_drivers.json")):
        engineer_features()
    else:
        print("Required JSON files for feature engineering not found yet.")

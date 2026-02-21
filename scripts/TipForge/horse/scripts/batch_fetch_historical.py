import json
import os
import time
from scripts.batch_fetch_today import update_consolidated_file
from scripts.convert_career import fetch_career_data

def collect_historical_participants(results_file):
    """
    Extracts unique horse and driver IDs from the historical results file.
    """
    if not os.path.exists(results_file):
        print(f"File not found: {results_file}")
        return None, None

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    horses = {}  # id: name
    drivers = {} # id: name

    for race in data.get("races", []):
        for p in race.get("participants", []):
            h_id = p.get("id")
            h_name = p.get("name")
            d_id = p.get("driver_jockey_id")
            d_name = p.get("driver_jockey")

            if h_id: horses[str(h_id)] = h_name
            if d_id: drivers[str(d_id)] = d_name

    return horses, drivers

def batch_fetch_historical(results_file):
    horses, drivers = collect_historical_participants(results_file)
    if not horses:
        return

    print(f"Starting bulk fetch for historical participants...")
    print(f"Total Unique Horses: {len(horses)}")
    print(f"Total Unique Drivers: {len(drivers)}")

    # Fetch Drivers
    print("\n--- Fetching Historical Drivers ---")
    for i, (d_id, d_name) in enumerate(drivers.items()):
        # Skip if already in today's drivers (optional, but saves time if there's overlap)
        # For now, let's just fetch all to be sure we have the full historical set
        print(f"[{i+1}/{len(drivers)}] Driver: {d_name} (ID: {d_id})")
        data = fetch_career_data(d_id, "driver_jockey")
        if data:
            update_consolidated_file("data/all_drivers.json", d_id, data, {"total": len(drivers)})
        time.sleep(0.4) # Slightly faster rate since it's a large batch

    # Fetch Horses
    print("\n--- Fetching Historical Horses ---")
    for i, (h_id, h_name) in enumerate(horses.items()):
        print(f"[{i+1}/{len(horses)}] Horse: {h_name} (ID: {h_id})")
        data = fetch_career_data(h_id, "participant")
        if data:
            update_consolidated_file("data/all_horses.json", h_id, data, {"total": len(horses)})
        time.sleep(0.4)

    print("\nHistorical career data collection complete.")

if __name__ == "__main__":
    batch_fetch_historical("data/historical_results_combined.json")

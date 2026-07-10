import json
import os
import time
from tqdm import tqdm
from scripts.batch_fetch_today import update_consolidated_file
from scripts.convert_career import fetch_career_data

def load_existing_ids(file_path):
    """Loads existing IDs from consolidated JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    return set(str(k) for k in data["data"].keys())
        except Exception as e:
            print(f"Warning loading {file_path}: {e}")
    return set()

def collect_historical_participants(results_file):
    """Extracts unique horse and driver IDs from the historical results file."""
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

    # Load already fetched IDs
    existing_drivers = load_existing_ids("data/all_drivers.json")
    existing_horses = load_existing_ids("data/all_horses.json")

    # Filter out already fetched participants
    drivers_to_fetch = {d_id: d_name for d_id, d_name in drivers.items() if str(d_id) not in existing_drivers}
    horses_to_fetch = {h_id: h_name for h_id, h_name in horses.items() if str(h_id) not in existing_horses}

    print(f"Total Unique Drivers in results: {len(drivers)}")
    print(f"Already fetched drivers: {len(drivers) - len(drivers_to_fetch)}")
    print(f"Drivers to fetch: {len(drivers_to_fetch)}")

    print(f"\nTotal Unique Horses in results: {len(horses)}")
    print(f"Already fetched horses: {len(horses) - len(horses_to_fetch)}")
    print(f"Horses to fetch: {len(horses_to_fetch)}")

    # Fetch Drivers
    if drivers_to_fetch:
        print("\n--- Fetching Historical Drivers ---")
        for d_id, d_name in tqdm(drivers_to_fetch.items(), desc="Fetching drivers"):
            data = fetch_career_data(d_id, "driver_jockey")
            if data:
                update_consolidated_file("data/all_drivers.json", d_id, data, {"total": len(drivers)})
            time.sleep(0.4)

    # Fetch Horses
    if horses_to_fetch:
        print("\n--- Fetching Historical Horses ---")
        for h_id, h_name in tqdm(horses_to_fetch.items(), desc="Fetching horses"):
            data = fetch_career_data(h_id, "participant")
            if data:
                update_consolidated_file("data/all_horses.json", h_id, data, {"total": len(horses)})
            time.sleep(0.4)

    print("\nHistorical career data collection complete.")

if __name__ == "__main__":
    batch_fetch_historical("data/historical_results_combined.json")

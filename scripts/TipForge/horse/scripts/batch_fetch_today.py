import json
import os
import time
from datetime import datetime
from scripts.convert_career import fetch_career_data

def update_consolidated_file(file_path, entity_id, data, metadata_info):
    """
    Loads existing file, updates with new data, and saves back.
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
    else:
        content = {
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "count": metadata_info["total"],
                "discipline": "trotting"
            },
            "data": {}
        }
    
    content["data"][str(entity_id)] = data
    content["metadata"]["last_updated"] = datetime.now().isoformat()
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

def batch_fetch_consolidated(ids_file):
    if not os.path.exists(ids_file):
        print(f"File not found: {ids_file}")
        return

    with open(ids_file, "r", encoding="utf-8") as f:
        id_registry = json.load(f)

    horses_registry = id_registry.get("horses", {})
    drivers_registry = id_registry.get("drivers", {})

    print(f"Starting incremental batch fetch...")

    # Fetch Drivers
    print("\n--- Fetching Drivers ---")
    for i, (d_id, d_name) in enumerate(drivers_registry.items()):
        print(f"[{i+1}/{len(drivers_registry)}] Driver: {d_name} (ID: {d_id})")
        data = fetch_career_data(d_id, "driver_jockey")
        if data:
            update_consolidated_file("data/today_drivers.json", d_id, data, {"total": len(drivers_registry)})
        time.sleep(0.5)

    # Fetch Horses
    print("\n--- Fetching Horses ---")
    for i, (h_id, h_name) in enumerate(horses_registry.items()):
        print(f"[{i+1}/{len(horses_registry)}] Horse: {h_name} (ID: {h_id})")
        data = fetch_career_data(h_id, "participant")
        if data:
            update_consolidated_file("data/today_horses.json", h_id, data, {"total": len(horses_registry)})
        time.sleep(0.5)

    print("\nRefined data collection complete.")

if __name__ == "__main__":
    batch_fetch_consolidated("data/today_ids.json")

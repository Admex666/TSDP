import json
import csv
import os
import requests
import pandas as pd

def fetch_career_data(entity_id, entity_type="driver_jockey", discipline="trotting"):
    """
    Fetches career data from Kincsem Park API.
    Returns: Dict containing the JSON response or None on error.
    """
    base_url = f"https://mla.kincsempark.hu/api/v1/{entity_type}_race_form_stats/{discipline}/{entity_id}"
    
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data for {entity_type} {entity_id}: {e}")
        return None

def fetch_and_convert(entity_id, entity_type="driver_jockey", discipline="trotting", save_files=True):
    """
    Fetches career data and optionally converts/saves it.
    Returns: The raw JSON data.
    """
    data = fetch_career_data(entity_id, entity_type, discipline)
    if not data:
        return None

    if save_files:
        # Create data directory if not exists
        os.makedirs("data", exist_ok=True)

        # Save as JSON
        json_path = f"data/sample_{entity_type}_{entity_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved JSON to {json_path}")

        # Convert results to CSV
        if 'results' in data and data['results']:
            results = data['results']
            df = pd.DataFrame(results)
            
            csv_path = f"data/sample_{entity_type}_{entity_id}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Saved CSV to {csv_path}")
        else:
            print("No results found in the data.")
    
    return data

if __name__ == "__main__":
    # Test with Fazekas Imre (ID 5)
    fetch_and_convert(5, "driver_jockey")

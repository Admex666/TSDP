import re
import json
import os

def parse_racecard(html_file):
    """
    Parses the racecard HTML to extract structured race data and pairings.
    """
    if not os.path.exists(html_file):
        print(f"File not found: {html_file}")
        return

    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all races_table_divs assignments
    pattern = r'races_table_divs\[".*?"\]\s*=\s*(\{.*?\});'
    matches = re.findall(pattern, content, re.DOTALL)

    horses = {}  # id: name
    drivers = {} # id: name
    racecard = []

    for match in matches:
        try:
            race_data = json.loads(match)
            race_info = {
                "race_id": race_data.get("race_id"),
                "race_name": race_data.get("race_name"),
                "start_time": race_data.get("start"),
                "distance": race_data.get("distance"),
                "participants": []
            }
            
            for p in race_data.get("participants", []):
                h_id = p.get("id")
                h_name = p.get("name")
                d_id = p.get("driver_jockey_id")
                d_name = p.get("driver_jockey")
                
                if h_id: horses[h_id] = h_name
                if d_id: drivers[d_id] = d_name
                
                race_info["participants"].append({
                    "horse_id": h_id,
                    "horse_name": h_name,
                    "driver_id": d_id,
                    "driver_name": d_name
                })
            
            racecard.append(race_info)
        except Exception as e:
            print(f"Error parsing race data: {e}")

    return racecard, horses, drivers

if __name__ == "__main__":
    html_path = "research/racecard_today.html"
    racecard, horses, drivers = parse_racecard(html_path)

    # Save unique IDs for batch fetching
    ids_data = {"horses": horses, "drivers": drivers}
    with open("data/today_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids_data, f, indent=4, ensure_ascii=False)

    # Save full racecard for pairing reference
    with open("data/today_racecard.json", "w", encoding="utf-8") as f:
        json.dump(racecard, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(racecard)} races.")
    print(f"Captured {len(horses)} horses and {len(drivers)} drivers.")
    print("Full racecard saved to data/today_racecard.json")

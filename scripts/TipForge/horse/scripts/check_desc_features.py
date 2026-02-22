import json
import os
import re

def analyze_descriptions():
    path = 'data/historical_results_combined.json'
    if not os.path.exists(path):
        print("Data file missing.")
        return

    print(f"Loading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    races = data.get('races', [])
    total = len(races)
    desc_count = 0
    track_count = 0
    temp_count = 0
    
    # Patterns
    # "Pálya minősége/Track: jó. Hőmérséklet: 0 °C"
    # "Pálya minősége/Track: jó"
    track_pattern = re.compile(r'Track:\s*([^.]+)', re.IGNORECASE)
    temp_pattern = re.compile(r'Hőmérséklet:\s*([-\d]+)', re.IGNORECASE)

    for r in races:
        desc = r.get('description', '')
        if desc:
            desc_count += 1
            if track_pattern.search(desc):
                track_count += 1
            if temp_pattern.search(desc):
                temp_count += 1
                
    print(f"Total Races: {total}")
    print(f"Races with description: {desc_count} ({desc_count/total:.1%})")
    print(f"Races with Track info: {track_count} ({track_count/total:.1%})")
    print(f"Races with Temp info: {temp_count} ({temp_count/total:.1%})")

if __name__ == "__main__":
    analyze_descriptions()

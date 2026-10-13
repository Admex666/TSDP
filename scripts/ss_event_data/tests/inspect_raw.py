import json
import re
import glob

raw_files = glob.glob('data/raw/*.json')
print(f"Found raw files: {raw_files}")

for path in raw_files:
    print(f"\n============================\nAnalyzing {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total recorded frames: {len(data)}")
    
    types = set()
    names = set()
    player_mentions = 0
    
    for entry in data:
        raw = entry.get('raw_payload', '')
        for t in re.findall(r'"type":"([^"]+)"', raw):
            types.add(t)
        for n in re.findall(r'"name":"([^"]+)"', raw):
            names.add(n)
        if 'player' in raw.lower():
            player_mentions += 1
            
    print(f"Types found: {types}")
    print(f"Names found: {names}")
    print(f"Player mentions in frames: {player_mentions}")

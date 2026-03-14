
import sys
import json
import pandas as pd
from typing import List, Dict

# Import sofascore module
sys.path.append(r"e:\Data\TSDP\modules")
from SofaScore_module import scrape_sofascore


leagues = [
    ("English Premier League", "Premier League", "England"),
    ("Spanish La Liga", "LaLiga", "Spain"),
    ("Italian Serie A", "Serie A", "Italy"),
    ("German Bundesliga", "Bundesliga", "Germany"),
    ("French Ligue 1", "Ligue 1", "France"),
    ("Belgian Jupiler Pro League", "Jupiler Pro League", "Belgium"),
    ("English Championship", "Championship", "England"),
    ("Liga Profesional Argentina", "Liga Profesional de Fútbol", "Argentina"),
    ("Brazilian Serie A", "Brasileiro Serie A", "Brazil"),
    ("Portuguese Primeira Liga", "Liga Portugal", "Portugal"),
    ("Danish Superligaen", "Superliga", "Denmark"),
    ("Polish Ekstraklasa", "Ekstraklasa", "Poland"),
    ("US Major League Soccer", "MLS", "USA"),
    ("Croatia Prva HNL", "HNL", "Croatia"),
    ("Ecuador Liga Pro", "LigaPro Primera A", "Ecuador"),
    ("Colombia Primera A Apertura", "Primera A", "Colombia"),
    ("Turkish Super Lig", "Süper Lig", "Turkey"),
    ("Norwegian Eliteserien", "Eliteserien", "Norway"),
    ("Japanese J1 League", "J1 League", "Japan"),
    ("Dutch Eredivisie", "Eredivisie", "Netherlands"),
    ("Spanish Segunda Division", "LaLiga 2", "Spain"),
    ("Greek Super League", "Super League 1", "Greece"),
    ("Czech First League", "1. Liga", "Czech Republic"),
    ("Brack Super League", "Super League", "Switzerland"),
    ("Russian Premier League", "Premier League", "Russia"),
    ("Division Profesional", "Division Profesional", "Bolivia"),
    ("German Bundesliga Zwei", "2. Bundesliga", "Germany"),
    ("Cypriot Division 1", "1. Division", "Cyprus"),
    ("Liga MX", "Liga MX", "Mexico"),
    ("Hungarian Liga NB I", "NB I", "Hungary"),
    ("Liga AUF", "Primera División", "Uruguay"),
    ("Swedish Allsvenskan", "Allsvenskan", "Sweden"),
    ("Austrian Bundesliga", "Bundesliga", "Austria"),
    ("Italian Serie B", "Serie B", "Italy"),
    ("Chile Primera", "Primera División", "Chile"),
    ("Saudi Arabian League", "Saudi Pro League", "Saudi Arabia"),
    ("Botola Pro", "Botola Pro 1", "Morocco"),
    ("French Ligue 2", "Ligue 2", "France"),
    ("Romania Liga I", "Superliga", "Romania"),
    ("Scottish Premiership", "Premiership", "Scotland")
]

def search_league(search_name: str, expected_country: str):
    # Try searching for unique tournaments first
    url = f"https://www.sofascore.com/api/v1/search/unique-tournaments?q={search_name.replace(' ', '%20')}&page=0"
    data = scrape_sofascore(url)
    
    if data and 'results' in data:
        # Filter for football tournaments
        football_results = [r for r in data['results'] if r.get('category', {}).get('sport', {}).get('name') == 'Football']
        
        # Try to match country
        for res in football_results:
            country = res.get('category', {}).get('country', {}).get('name')
            if country == expected_country:
                return res
            if expected_country == "USA" and country == "USA":
                return res
            if expected_country == "Netherlands" and country == "Netherlands":
                return res
        
        # Fallback to first football result if no country match
        if football_results:
            return football_results[0]
            
    return None

results = []
for i, (orig_name, search_name, country) in enumerate(leagues):
    print(f"[{i+1}/{len(leagues)}] Searching for: {search_name} ({country})")
    match = search_league(search_name, country)
    if match:
        results.append({
            "rank": i + 1,
            "input_name": orig_name,
            "sofascore_name": match.get('name'),
            "id": match.get('id'),
            "category": match.get('category', {}).get('name'),
            "country": match.get('category', {}).get('country', {}).get('name')
        })
        print(f"  Found: {match.get('name')} (ID: {match.get('id')}) in {match.get('category', {}).get('name')}")
    else:
        print(f"  Not found: {search_name} in {country}")
        results.append({
            "rank": i + 1,
            "input_name": orig_name,
            "sofascore_name": None,
            "id": None,
            "category": None,
            "country": country
        })

# Save to JSON
with open(r"e:\Data\TSDP\scripts\league_ids.json", "w", encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(r"e:\Data\TSDP\scripts\league_ids.csv", index=False, encoding='utf-8-sig')


print("\nDone! Results saved to league_ids.json and league_ids.csv")

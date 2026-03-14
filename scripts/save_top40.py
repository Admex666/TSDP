
import json
import pandas as pd

league_data = [
    {"rank": 1, "league": "English Premier League", "sofascore_name": "Premier League", "id": 17, "country": "England"},
    {"rank": 2, "league": "Spanish La Liga", "sofascore_name": "LaLiga", "id": 8, "country": "Spain"},
    {"rank": 3, "league": "Italian Serie A", "sofascore_name": "Serie A", "id": 23, "country": "Italy"},
    {"rank": 4, "league": "German Bundesliga", "sofascore_name": "Bundesliga", "id": 35, "country": "Germany"},
    {"rank": 5, "league": "French Ligue 1", "sofascore_name": "Ligue 1", "id": 34, "country": "France"},
    {"rank": 6, "league": "Belgian Jupiler Pro League", "sofascore_name": "Jupiler Pro League", "id": 38, "country": "Belgium"},
    {"rank": 7, "league": "English Championship", "sofascore_name": "Championship", "id": 18, "country": "England"},
    {"rank": 8, "league": "Liga Profesional Argentina", "sofascore_name": "Liga Profesional de Fútbol", "id": 155, "country": "Argentina"},
    {"rank": 9, "league": "Brazilian Serie A", "sofascore_name": "Brasileiro Serie A", "id": 325, "country": "Brazil"},
    {"rank": 10, "league": "Portuguese Primeira Liga", "sofascore_name": "Liga Portugal", "id": 238, "country": "Portugal"},
    {"rank": 11, "league": "Danish Superligaen", "sofascore_name": "Superliga", "id": 39, "country": "Denmark"},
    {"rank": 12, "league": "Polish Ekstraklasa", "sofascore_name": "Ekstraklasa", "id": 202, "country": "Poland"},
    {"rank": 13, "league": "US Major League Soccer", "sofascore_name": "MLS", "id": 242, "country": "USA"},
    {"rank": 14, "league": "Croatia Prva HNL", "sofascore_name": "HNL", "id": 170, "country": "Croatia"},
    {"rank": 15, "league": "Ecuador Liga Pro", "sofascore_name": "LigaPro Primera A", "id": 240, "country": "Ecuador"},
    {"rank": 16, "league": "Colombia Primera A Apertura", "sofascore_name": "Primera A", "id": 11539, "country": "Colombia"},
    {"rank": 17, "league": "Turkish Super Lig", "sofascore_name": "Süper Lig", "id": 52, "country": "Turkey"},
    {"rank": 18, "league": "Norwegian Eliteserien", "sofascore_name": "Eliteserien", "id": 20, "country": "Norway"},
    {"rank": 19, "league": "Japanese J1 League", "sofascore_name": "J1 League", "id": 196, "country": "Japan"},
    {"rank": 20, "league": "Dutch Eredivisie", "sofascore_name": "Eredivisie", "id": 37, "country": "Netherlands"},
    {"rank": 21, "league": "Spanish Segunda Division", "sofascore_name": "LaLiga 2", "id": 54, "country": "Spain"},
    {"rank": 22, "league": "Greek Super League", "sofascore_name": "Super League 1", "id": 185, "country": "Greece"},
    {"rank": 23, "league": "Czech First League", "sofascore_name": "1. Liga", "id": 229, "country": "Czech Republic"},
    {"rank": 24, "league": "Brack Super League", "sofascore_name": "Super League", "id": 215, "country": "Switzerland"},
    {"rank": 25, "league": "Russian Premier League", "sofascore_name": "Premier League", "id": 203, "country": "Russia"},
    {"rank": 26, "league": "Division Profesional", "sofascore_name": "Division Profesional", "id": 16736, "country": "Bolivia"},
    {"rank": 27, "league": "German Bundesliga Zwei", "sofascore_name": "2. Bundesliga", "id": 44, "country": "Germany"},
    {"rank": 28, "league": "Cypriot Division 1", "sofascore_name": "1. Division", "id": 1416, "country": "Cyprus"},
    {"rank": 29, "league": "Liga MX", "sofascore_name": "Liga MX", "id": 11621, "country": "Mexico"},
    {"rank": 30, "league": "Hungarian Liga NB I", "sofascore_name": "NB I", "id": 187, "country": "Hungary"},
    {"rank": 31, "league": "Liga AUF", "sofascore_name": "Primera División", "id": 278, "country": "Uruguay"},
    {"rank": 32, "league": "Swedish Allsvenskan", "sofascore_name": "Allsvenskan", "id": 40, "country": "Sweden"},
    {"rank": 33, "league": "Austrian Bundesliga", "sofascore_name": "Bundesliga", "id": 45, "country": "Austria"},
    {"rank": 34, "league": "Italian Serie B", "sofascore_name": "Serie B", "id": 53, "country": "Italy"},
    {"rank": 35, "league": "Chile Primera", "sofascore_name": "Primera División", "id": 11653, "country": "Chile"},
    {"rank": 36, "league": "Saudi Arabian League", "sofascore_name": "Saudi Pro League", "id": 955, "country": "Saudi Arabia"},
    {"rank": 37, "league": "Botola Pro", "sofascore_name": "Botola Pro 1", "id": 937, "country": "Morocco"},
    {"rank": 38, "league": "French Ligue 2", "sofascore_name": "Ligue 2", "id": 182, "country": "France"},
    {"rank": 39, "league": "Romania Liga I", "sofascore_name": "Superliga", "id": 188, "country": "Romania"},
    {"rank": 40, "league": "Scottish Premiership", "sofascore_name": "Premiership", "id": 36, "country": "Scotland"}
]

# Save to JSON
with open(r"e:\Data\TSDP\scripts\top40_leagues.json", "w", encoding='utf-8') as f:
    json.dump(league_data, f, indent=4, ensure_ascii=False)

# Save to CSV
df = pd.DataFrame(league_data)
df.to_csv(r"e:\Data\TSDP\scripts\top40_leagues.csv", index=False, encoding='utf-8-sig')

print("Top 40 leagues SofaScore IDs saved to top40_leagues.json and top40_leagues.csv")

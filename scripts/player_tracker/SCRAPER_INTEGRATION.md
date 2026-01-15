# Player Tracker - Scraper Integration

## FBref és SofaScore Scraper Integráció

A Player Tracker most már támogatja a közvetlen adatimportot a meglévő FBref és SofaScore scraper moduljaidból!

## Telepítés

```powershell
cd backend
pip install -r requirements_scrapers.txt
```

Ez telepíti:
- `selenium` - FBref scraping (Chrome WebDriver)
- `webdriver-manager` - Automatikus driver kezelés
- `tls-client` - SofaScore scraping

## Használat

### 1. FBref Liga Import

**Web UI:**
1. Nyisd meg: `http://localhost:8003/scraper_import.html`
2. Válaszd ki a ligát (pl. HUN, GER, ENG)
3. Opcionálisan add meg a szezont (pl. 2024-2025)
4. Kattints "FBref Import Indítása"

**API:**
```http
POST /api/v1/scraper/fbref/league?countrycode=HUN&season=2024-2025&auto_create_players=true
```

**Támogatott ligák:**
- `HUN` - NB I (Magyar)
- `GER` - Bundesliga
- `ENG` - Premier League
- `ESP` - La Liga
- `ITA` - Serie A
- `FRA` - Ligue 1
- `NED` - Eredivisie
- `POR` - Primeira Liga
- `AUT` - Austrian Bundesliga
- `BEL` - Belgian Pro League
- `Big5` - Top 5 ligák kombinálva

### 2. SofaScore Meccs Import

**Web UI:**
1. Nyisd meg: `http://localhost:8003/scraper_import.html`
2. Add meg a SofaScore Event ID-t
3. Kattints "SofaScore Import Indítása"

**API:**
```http
POST /api/v1/scraper/sofascore/match?event_id=12345678&auto_create_players=true
```

**Event ID megtalálása:**
- SofaScore URL: `sofascore.com/match/12345678`
- Az Event ID az URL-ben található

## Funkciók

### FBref Scraper
- ✅ Teljes liga összes játékosának statisztikái
- ✅ Selenium-alapú (Chrome WebDriver)
- ✅ Automatikus liga/csapat/játékos létrehozás
- ✅ Fuzzy matching meglévő játékosokkal
- ✅ Komprehenzív statisztikák (standard, passing, defense, stb.)

### SofaScore Scraper
- ✅ Meccs-szintű részletes statisztikák
- ✅ TLS client-alapú (gyors)
- ✅ Lineups, average positions, shotmap
- ✅ Játékos-szintű adatok
- ✅ JSON formátumban tárolt advanced stats

## Architektúra

```
backend/app/services/
├── fbref_scraper.py          # FBref modul
├── sofascore_scraper.py      # SofaScore modul
└── enhanced_import.py        # Unified import service

backend/app/api/
└── scraper.py                # Scraper API endpoints
```

## Példa Használat

### Python Script

```python
from app.database import SessionLocal
from app.services.enhanced_import import EnhancedImportService

db = SessionLocal()
service = EnhancedImportService(db)

# FBref import
result = service.import_fbref_league('HUN', '2024-2025')
print(f"Imported {result['imported_records']} records")

# SofaScore import
result = service.import_sofascore_match(12345678)
print(f"Matched {result['matched_players']} players")
```

## Megjegyzések

- **FBref import lassú lehet** (2-5 perc) sok adat miatt
- **Chrome WebDriver** szükséges FBref-hez (automatikusan települ)
- **Fuzzy matching** 80% threshold-dal párosít
- **Auto-create** opció új játékosokat hoz létre ha nincs találat

## Következő Lépések

Most már importálhatsz adatokat közvetlenül a scraperekből! Próbáld ki:
1. Importálj NB I adatokat FBref-ről
2. Kövesd a magyar játékosokat
3. Nézd meg a statisztikáikat a dashboardon

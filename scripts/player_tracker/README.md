# ⚽ Player Tracker

Futball játékos teljesítmény-követő és értékelő rendszer magyar játékosokhoz.

## Funkciók

- 📊 **Játékos statisztikák követése** - FBref és SofaScore adatok importálása
- 📈 **Teljesítmény értékelés** - Többdimenziós értékelés pozíció és liga szerint
- 🎯 **Forma trend** - Historikus teljesítmény követése
- 🌐 **Web interfész** - Modern, dark-themed dashboard
- 🔄 **Automatikus párosítás** - Fuzzy matching algoritmus játékosokhoz

## Gyors Indítás

### 1. Telepítés

```powershell
# Navigálj a backend mappába
cd backend

# Telepítsd a függőségeket
pip install -r requirements.txt
```

### 2. Adatbázis inicializálás

```powershell
# Futtasd az inicializáló scriptet
python ..\scripts\init_db.py
```

### 3. Szerver indítása

```powershell
# Indítsd el a FastAPI szervert (port 8003)
python -m app.main
```

A szerver elindul a `http://localhost:8003` címen.

### 4. Web interfész megnyitása

Nyisd meg a böngészőben: `http://localhost:8003`

## Projekt Struktúra

```
player_tracker/
├── backend/
│   ├── app/
│   │   ├── api/              # API endpointok
│   │   ├── services/         # Üzleti logika
│   │   ├── utils/            # Segédfüggvények
│   │   ├── database.py       # DB kapcsolat
│   │   ├── models.py         # SQLAlchemy modellek
│   │   ├── schemas.py        # Pydantic sémák
│   │   └── main.py           # FastAPI app
│   ├── config.py             # Konfiguráció
│   └── requirements.txt      # Python függőségek
├── frontend/
│   ├── index.html            # Dashboard
│   ├── players.html          # Játékos lista
│   ├── player_detail.html    # Játékos részletek
│   ├── import.html           # Adatimport
│   ├── css/styles.css        # Stílusok
│   └── js/api.js             # API kliens
├── data/
│   ├── player_tracker.db     # SQLite adatbázis
│   └── imports/              # Import fájlok
└── scripts/
    └── init_db.py            # DB inicializálás
```

## Használat

### Adatimport

1. Navigálj az **Adatimport** oldalra
2. Válaszd ki az adatforrást (FBref vagy SofaScore)
3. Töltsd fel a CSV fájlt
4. Kattints az **Import Indítása** gombra

### Játékosok követése

1. Menj a **Játékosok** oldalra
2. Kattints a **+ Követés** gombra a kívánt játékosoknál
3. A követett játékosok megjelennek a Dashboard-on

### Játékos részletek

- Kattints a **Részletek** gombra bármelyik játékosnál
- Megtekintheted:
  - Szezon statisztikákat
  - Értékelési metrikákat
  - Meccs-by-meccs statisztikákat

## API Dokumentáció

A teljes API dokumentáció elérhető: `http://localhost:8003/docs`

### Főbb Endpointok

- `GET /api/v1/players` - Játékosok listája
- `GET /api/v1/players/{id}` - Játékos részletei
- `GET /api/v1/players/{id}/stats` - Játékos statisztikák
- `PATCH /api/v1/players/{id}` - Játékos frissítése
- `POST /api/v1/import/stats` - Statisztikák importálása
- `GET /api/v1/leagues` - Ligák listája

## Konfiguráció

Szerkeszd a `backend/config.py` fájlt:

```python
class Config:
    API_PORT = 8003                    # API port
    MIN_MINUTES_FOR_EVALUATION = 450   # Min. játékperc értékeléshez
    PLAYER_MATCH_THRESHOLD = 80        # Fuzzy matching küszöb
    FORM_TREND_WINDOW = 10             # Forma trend ablak (meccsek)
```

## Következő Lépések (Fázis 2)

- [ ] ClubElo API integráció
- [ ] Liga-súlyozott értékelés
- [ ] Kompozit index implementálása
- [ ] Chart.js vizualizációk
- [ ] Player comparison funkció

## Technológiák

- **Backend**: Python, FastAPI, SQLAlchemy, Pandas
- **Frontend**: HTML, CSS, Vanilla JavaScript
- **Database**: SQLite
- **Matching**: FuzzyWuzzy

## Licenc

Személyes használatra.

---

**Készítette**: Player Tracker System  
**Verzió**: 1.0.0 (MVP)  
**Dátum**: 2026-01-15

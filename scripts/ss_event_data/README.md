# SofaScore 3D Live Event & Tracking Engine (`ss_event_data`)

Ez a projekt a **SofaScore és Sportradar valós idejű élő mikro-eseményeinek, 3D labda trajektóriáinak és valós GPS tracking adatainak** elfogására, tárolására, feldolgozására és interaktív vizualizációjára szolgál.

---

## 📁 Mappa Struktúra

```text
ss_event_data/
├── docs/                                  # Részletes technikai és architektúra dokumentáció
│   └── SOFASCORE_LIVE_TRACKING_ARCHITECTURE.md
├── data/                                  # Adattárolás
│   ├── raw/                               # Nyers WebSocket JSON stream mentések
│   │   ├── live_stream_16483643.json
│   │   └── live_stream_16939017.json
│   ├── processed/                         # Feldolgozott CSV táblázatok
│   │   ├── live_events_16483643.csv
│   │   ├── live_events_16939017.csv
│   │   ├── player_passes_16483643.csv
│   │   └── matched_events_16483643.csv
│   └── output/                            # Generált vizualizációk (PNG képek, interaktív HTML-ek)
│       ├── tracking_16939017_pitch.png
│       └── tracking_visualizer_16939017.html
├── tests/                                 # Tesztek és kísérleti szkriptek
│   └── test_stealth.py
├── live_cdp_recorder.py                   # Valós idejű WebSocket rögzítő (Chrome DevTools Protocol)
├── live_event_scraper.py                  # Standalone headless scraper (SofaScore_module wrapper)
├── match_events_with_passes.py            # Térbeli euklideszi összerendelő motor (tracking + passes)
├── visualize_tracking.py                  # Pitch vizualizáció: Dark mplsoccer PNG és interaktív HTML player
└── README.md
```

---

## 🚀 Használati Útmutató

### 1. Élő Meccs Tracking Rögzítése (Chrome CDP-n keresztül)

A Cloudflare / Turnstile védelem megkerüléséhez indítsunk egy Chrome böngészőt nyitott remote debugging porttal:
```bash
chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\ChromeProfile"
```
Nyissuk meg a kívánt SofaScore mérkőzést (pl. `https://www.sofascore.com/...#id:16939017`), majd futtassuk a rögzítőt:
```bash
python live_cdp_recorder.py
```
* **Kimenet**:
  - `data/raw/live_stream_{match_id}.json` (nyers frame-ek)
  - `data/processed/live_events_{match_id}.csv` (parse-olt mikro-események és koordináták)

---

### 2. Tracking Adatok Vizualizációja

A rögzített CSV adatok automatikus vizualizálásához:
```bash
python visualize_tracking.py
```
A script két kimenetet generál a `data/output/` mappába:
1. **Statikus pálya (`tracking_{id}_pitch.png`)**:
   - `mplsoccer` és `matplotlib` alapú, sötét tónusú prémium pályaábra.
   - Labdamozgási ívek (spline/nyilak), zónák színkódolása (`safe`, `attack`, `dangerous`), időlépések és koordináták.
2. **Interaktív lejátszó (`tracking_visualizer_{id}.html`)**:
   - Standalone HTML5 + JavaScript futballpálya szimulátor.
   - **Funkciók**: Lejátszás (Play/Pause), lépésenkénti léptetés, csúszka (Slider), valós idejű pulzáló labda, koordináta feliratok, interaktív oldalsó eseménylista.

---

### 3. Összekapcsolás Játékos Passzokkal és Akciókkal

A rögzített 3D labda koordinátákat idő és térbeli (4D-s euklideszi távolság) alapján hozzárendeli a SofaScore Opta passz adatbázisához:
```bash
python match_events_with_passes.py
```
* **Kimenet**: `data/processed/matched_events_{id}.csv` (tartalmazza a játékosnevet, passz kimenetelt, akciótípust és a távolsági hibát).

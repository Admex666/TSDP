# Liga-Összefoglaló Automatikus Generátor

Automatizált futball liga riport készítő rendszer Python + LaTeX alapon.

## 📋 Rendszerkövetelmények

### Szükséges telepítések:

**Python csomagok:**
```bash
pip install pandas pyyaml jinja2
```

**LaTeX disztribúció:**
- Windows: MiKTeX vagy TeX Live
- Linux: `sudo apt-get install texlive-full`
- macOS: MacTeX

**LaTeX csomagok** (általában előre telepítve):
- babel, booktabs, xcolor, tikz, pgfplots, tcolorbox, fontawesome5

## 📁 Projekt Struktúra

```
liga-report-generator/
├── report_generator.py       # Fő Python modul
├── config.yaml                # Konfiguráció
├── templates/
│   └── main_template.tex      # LaTeX sablon
├── data/
│   ├── round_template.json    # Minta input
│   └── round_10.json          # Aktuális forduló adatok
├── output/                    # Generált fájlok
│   ├── liga_report_round_10.tex
│   └── liga_report_round_10.pdf
└── README.md
```

## 🚀 Használat

### 1. Adatok előkészítése

Hozz létre egy JSON fájlt a forduló adataival a `data/` mappában (lásd: `round_template.json` példa).

**Minimális szükséges adatok:**
- Liga alapadatok (név, forduló, dátum)
- Csapatok (helyezés, pontok, forma, xG/xGA)
- Mérkőzések (eredmény, xG értékek)
- Következő forduló meccsek (opcionális)

### 2. Konfiguráció testreszabása

Szerkeszd a `config.yaml` fájlt a ligád sajátosságaihoz:

```yaml
zone_limits:
  champions_league: 4      # Pl. 2-re módosítás Bundesliga esetén
  relegation_start: 18     # Pl. 16-ra 18 csapatos ligánál

thresholds:
  good_form: 2.5           # Jó forma küszöb
  overperforming: 5        # Felülteljesítés határ
```

### 3. Riport generálás

```python
from report_generator import LigaReportGenerator

# Inicializálás
generator = LigaReportGenerator("config.yaml")

# Riport készítés
pdf_path = generator.generate_report(
    data_path="data/round_10.json",
    output_name="premier_league_round_10"
)

print(f"PDF elkészült: {pdf_path}")
```

**Vagy parancssorból:**

```bash
python report_generator.py
```

### 4. Kimenet

A `output/` mappában megtalálod:
- `liga_report_round_10.tex` - LaTeX forráskód
- `liga_report_round_10.pdf` - Kész PDF dokumentum

## 📊 Input Adatstruktúra

### Csapat objektum formátum:

```json
{
  "team_id": "mci",
  "team_name": "Manchester City",
  "position": 1,
  "position_history": [2, 1, 1, 1],
  "matches_played": 10,
  "wins": 7,
  "draws": 2,
  "losses": 1,
  "points": 23,
  "goals_for": 25,
  "goals_against": 8,
  "goal_difference": 17,
  "xg_total": 23.5,
  "xga_total": 7.2,
  "last_5_results": "WWDWW",
  "last_5_points": 13
}
```

### Mérkőzés objektum formátum:

```json
{
  "match_id": "m101",
  "date": "2024-11-02",
  "home_team": "Manchester City",
  "away_team": "Arsenal",
  "home_goals": 2,
  "away_goals": 1,
  "home_xg": 1.8,
  "away_xg": 1.2,
  "home_shots": 15,
  "away_shots": 10,
  "home_possession": 58,
  "away_possession": 42
}
```

## 🔧 Testreszabás

### Színek módosítása

`config.yaml`-ban:

```yaml
colors:
  champions_league: "green!30"
  relegation: "red!25"
  positive: "green!50"
```

### Narratíva template-ek

```yaml
narratives:
  best_form: "{team} kiváló formában: {form_index}, {position}. hely"
  danger_zone: "{team} veszélyben: rossz forma, sürgős beavatkozás!"
```

### Szekciók ki/bekapcsolása

```yaml
sections:
  executive_summary: true
  table_momentum: true
  trend_analysis: false    # Kikapcsolás
```

## 📈 Számított Metrikák

A rendszer automatikusan kiszámítja:

- **xPoints**: Várható pontok xG/xGA alapján
- **Forma index**: Súlyozott átlag utolsó 5 meccsből
- **Performance gap**: Tényleges vs várható pontok különbsége
- **Momentum**: Pozícióváltozás trendek
- **Kategorizálás**: Upward/Danger/Over-Under performing

## 🎨 Vizualizációk

A riport tartalmazza:

- ✅ Színkódolt tabella (BL/EL/Kiesés zónák)
- ✅ Momentum hőtérkép
- ✅ xG vs Goals scatter plot
- ✅ Forma grafikonok (sparklines)
- ✅ Top 5 listák kategóriánként

## 🔍 Hibakeresés

### LaTeX fordítási hiba

```bash
# Ellenőrizd a LaTeX telepítést
pdflatex --version

# Manuális fordítás teszteléshez
cd output
pdflatex liga_report_round_10.tex
```

### Python import hibák

```bash
# Csomagok újratelepítése
pip install --upgrade pandas pyyaml jinja2
```

### Hiányzó adatok

A generátor automatikusan kezel hiányzó értékeket (0-ra vagy üres string-re állítja).

## 📝 Példa Workflow

```python
# 1. Adatok gyűjtése (te implementálod)
from your_scraper import get_league_data

raw_data = get_league_data(league="EPL", round=10)

# 2. Formázás a rendszer formátumára
formatted_data = format_to_template(raw_data)

# 3. Mentés JSON-be
import json
with open('data/round_10.json', 'w') as f:
    json.dump(formatted_data, f, indent=2)

# 4. Riport generálás
from report_generator import LigaReportGenerator
generator = LigaReportGenerator()
generator.generate_report('data/round_10.json')
```

## 🌍 Több Liga Támogatás

Hozz létre liga-specifikus konfigokat:

```bash
config_premier_league.yaml
config_bundesliga.yaml
config_serie_a.yaml
```

Használat:

```python
generator = LigaReportGenerator("config_bundesliga.yaml")
```

## 📧 Automatizálás

Cron job beállítás (Linux/Mac):

```bash
# Minden hétfő reggel 8-kor
0 8 * * 1 /usr/bin/python3 /path/to/report_generator.py
```

Windows Task Scheduler-rel is működik hasonlóan.

## 🆘 Támogatás

Kérdések esetén ellenőrizd:
1. Input JSON formátum helyes-e
2. Minden kötelező mező kitöltött
3. LaTeX telepítés működik
4. Python csomagok verziókompatibilitás

---

**Verzió:** 1.0  
**Utolsó frissítés:** 2024 November  
**Licenc:** MIT

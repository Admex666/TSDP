# SofaScore Fantasy PL Scripts

Ez a mappa tartalmazza a SofaScore Fantasy Premier League adatgyűjtéséhez és a pontszámok előrejelzéséhez szükséges scripteket.

## Fájlok és Funkciók

### 1. `price_scraper.py`
Ez a script gyűjti le az aktuális játékos árakat és egyéb fantasy adatokat a SofaScore API-n keresztül.
- **Bemenet**: Közvetlenül a SofaScore API-t hívja (`/api/v1/fantasy/round/.../players`).
- **Kimenet**: `player_prices.csv`
- **Működés**: 
  - Végigiterál az összes elérhető oldalon.
  - Kezeli a "rate limit"-et és a 403-as hibákat a `SofaScore_module` segítségével.
  - Kimenti a játékos nevét, csapatát, árát, ID-ját és slug-ját.
- **Használat**: 
  ```bash
  python price_scraper.py
  ```

### 2. `data_collector.py`
A történelmi adatok (korábbi szezonok és az aktuális szezon lezárt fordulói) gyűjtésére szolgál.
- **Kimenet**: Nyers mérkőzés statisztikák (parquet vagy csv formátumban a `data/raw` mappába).
- **Működés**: Háttérben fut, és folyamatosan tölti le a hiányzó fordulók adatait.

### 3. `predict_next_round.py`
A következő forduló pontszámainak előrejelzését végzi.
- **Bemenet**: 
  - `player_prices.csv` (az árakhoz és csapatokhoz)
  - Történelmi adatok (a formák és átlagok számításához)
  - Következő forduló sorsolása (fixtures)
- **Kimenet**: `predictions.csv` (Játékos, Becsült Pont, Ár, Value)
- **Státusz**: Fejlesztés alatt (jelenleg a fixture-ök lekérése és a feature engineering vázlata kész).

## Specifikus Modulok

- **`../../modules/SofaScore_module.py`**: Közös modul az API kérések kezelésére (User-Agent rotálás, TLS fingerprinting, retry logika).

## Telepítés és Követelmények

A scriptek futtatásához szükséges Python csomagok:
- `requests`
- `pandas`
- `numpy`
- `joblib`
- `tqdm`

Futtatás a gyökérkönyvtárból vagy a script mappájából:
```bash
python scripts/sofa_fantasy/price_scraper.py
```

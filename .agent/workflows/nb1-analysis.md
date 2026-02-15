---
description: Hogyan futtasd az NB1 Odds vs Teljesítmény elemzést
---

### Előfeltételek
Győződj meg róla, hogy a szükséges Python csomagok telepítve vannak:
- `pandas`
- `streamlit`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
- `tls-client` (a SofaScore modulhoz)

### Lépések

1. **Adatgyűjtés**:
   Futtasd az adatgyűjtő scriptet, ami lekéri a 2024-25-ös szezon adatait a SofaScore-ról:
   ```powershell
   python scripts/odds_vs_performance/data_fetcher.py
   ```
   *Megjegyzés: A script inkrementálisan ment, így ha megszakad, onnan folytatja ahol abbahagyta.*

2. **Dashboard indítása**:
   Ha már van elegendő adat (legalább pár forduló), indítsd el a Streamlit dashboardot:
   ```powershell
   streamlit run scripts/odds_vs_performance/nb1_analysis_dashboard.py
   ```

3. **Használat**:
   - A bal oldali sávban válaszd ki azokat a csapatokat, akiket össze akarsz vetni.
   - Az **Idősoros Elemzés** fülön láthatod, hogyan alakult a csapatok elvárt vs tényleges pontszáma.
   - A **Fogadóiroda Precízió** fül megmutatja, mennyire pontosak az odds-ok statisztikailag.
   - A **Teljesítmény Rangsor** fülön láthatod a "szerencse" (vagy taktikai túlteljesítés) szerinti sorrendet.
   - A **Csapat Profil** fülön egy-egy csapat mérkőzéseit elemezheted mélyebben.

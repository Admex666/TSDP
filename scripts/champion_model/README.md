# NB I Champion Model & Generatív Scoreline / Betting Engine

Ez a projekt a magyar labdarúgó-bajnokság (NB I) matematikai, statisztikai és gépi tanulási modellezésére készült a 2015–2026 közötti időszakra.

## 📁 Projekt Struktúra

- **`app.py`**: A fő Streamlit Dashboard alkalmazás (kettős nézettel: Champion Model & Bajnoki Szimuláció, valamint Betting & Odds Dokumentáció / Vizualizáció).
- **`data/`**: Minden kanonikus meccsadat, élő szezon mérkőzések, történelmi Oddsportal odds-ok és modellszintű predikciók.
- **`models/`**:
  - `dynamic_elo.py`: 1D Dinamikus Elo rating motor szezonszintű lecsengéssel és döntetlen modellezéssel.
  - `dynamic_dixon_coles.py`: Dinamikus Támadó/Védekező rating Poisson gólfolyamattal és Dixon–Coles $\tau$ korrekcióval ($\rho=-0.10$).
  - `scoreline_engine.py`: V3 Generatív Full Scoreline Engine (teljes 2D valószínűségi mátrix, Over/Under 0.5–5.5, BTTS, fair odds-ok).
- **`scrapers/`**:
  - `scrape_last11_nb1.py`: Transfermarkt történelmi 11 szezon meccskaparó.
  - `scrape_current_season.py`: Transfermarkt élő szezonkaparó automatikus frissítéshez.
  - `scrape_oddsportal_nb1.py`: Oddsportal történelmi 1X2 odds-kaparó (10 szezon).
- **`backtest/`**:
  - `full_scoreline_audit.py`: V3 Proper scoring rule és multi-threshold O/U audit.
  - `v1_validation.py` & `v2_ablation_and_season_sim.py`: Elo és DC modell validációs és ablációs vizsgálatai.
  - `historical_season_backtest.py`: 8 szezonos out-of-sample Monte Carlo backtest.
- **`simulations/`**:
  - `generate_all_season_simulations.py`: Szezon- és fordulószintű szimulációk pre-generálása.
  - `generate_multi_model_dataset.py`: Multi-modell összehasonlító szimulációk.

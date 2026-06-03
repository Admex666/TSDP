# Sorare ML Modellek Kiértékelése és Verziókövetése

Ez a dokumentum bemutatja a Sorare Profitability AI rendszerében végrehajtott gépi tanulási modellfejlesztéseket, összehasonlítva a korábbi (Random Forest) és a megújult (Gradient Boosting) verziók teljesítményét, valamint dokumentálja a sérülés- és eltiltás-figyelő rendszert.

Frissítés dátuma: **2026-06-03**

---

## 1. Modell Verziók Összehasonlítása

| Mutató / Funkció | v1.0 (Random Forest) | v2.1 (GB Regressor + Kalibráció + Safety) | Állapot |
| :--- | :--- | :--- | :--- |
| **Algoritmus** | Random Forest Regressor | **Gradient Boosting Regressor** | Fejlesztve |
| **Score Predictor MAE** | 17.65 pont | **16.55 pont** (1.1 pont javulás) | Sikeres |
| **Price ROI Predictor MAE**| 6.04% ROI | **5.75% ROI** (0.29% javulás) | Sikeres |
| **Price Baseline MAE** | 6.89% ROI | **6.28% ROI** | - |
| **Bázis feletti ROI javulás**| +12.3% | **+8.3%** (szigorúbb bázis mellett) | Megbízhatóbb |
| **Sérülés / Eltiltás szűrés**| Nincs (old averages alapján jósolt) | **Van (várható pontszám = 0.0)** | Integrálva |
| **NaN (hiányzó adat) kezelés**| Szenzitív (hibára futhatott) | **Automatikus (fillna(0.0) imputer)** | Integrálva |
| **Streamlit Újratanítás** | Manuális (CLI indítású) | **Dinamikus UI gomb (Sidebar)** | Integrálva |
| **Modell Kalibráció** | Nincs | **Van (Scatter, Hiba eloszlás & Sávok)** | Beépítve |

---

## 2. Részletes Modellértékelés (v2.0)

### A. Pontszám Prediktor (Score AI)
* **Célváltozó:** A játékos következő mérkőzésen elért SO5 pontszáma (0-100).
* **Változás a döntési logikában (Feature Importance):**
  * A legutóbbi 3 meccs gördülő átlaga (`rolling_mean_3`) kiemelkedő, **39.76%** súlyt kapott a döntési fákban.
  * Az előző meccs pontja (`prev_score_1`) **15.61%** súllyal a második legfontosabb tényező.
  * Az életkor (`age`) **8.72%** súllyal bír (a tapasztalat/korosztály stabilitást mérve).
* **Értékelés:** A Gradient Boosting modell jobban kiszűri a véletlenszerű, egyszeri kiugrásokat, és sokkal stabilabban vetíti előre a várható pontszámot a középtávú forma alapján.

### B. Árfolyam ROI Prediktor (Price ROI AI)
* **Célváltozó:** A kártya százalékos árváltozása (ROI %) a jelenlegi floor ár és a következő piaci lezárult eladás (recent sale) között.
* **Változás a döntési logikában (Feature Importance):**
  * A legutóbbi ártrend (`price_trend_pct`) súlya **53.1% -> 47.24%**-ra csökkent.
  * A jelenlegi floor ár (`current_price`) súlya **15.9% -> 19.18%**-ra nőtt.
  * Az előző tranzakció ára (`prev_price_1`) súlya **9.1% -> 16.27%**-ra nőtt.
* **Értékelés:** A modell kevésbé lett spekulatív és kevésbé hajlamos túlreagálni a hirtelen áringadozásokat. Ehelyett nagyobb hangsúlyt fektet az abszolút árszintekre és a történelmi árakhoz képest mért diszkontra.

---

## 3. Sérülés- és Eltiltás-kezelő Logika (Safety Filter)

A bot üzleti biztonságának növelése érdekében bevezetett logika az alábbi módon működik:

1. **Szinkronizáció:** A GraphQL lekérdezés lekéri az `activeInjuries { active }` és `activeSuspensions { active }` mezőket a Sorare API-ról.
2. **Mentés:** Az SQLite adatbázisban a játékosok `is_injured` és `is_suspended` mezői frissülnek (0 vagy 1 értékkel).
3. **Inferencia (ML):** A predikciók generálásakor, ha a játékos sérült vagy eltiltott:
   * A rendszer a várható hazai és idegenbeli pontszámait **automatikusan 0.0-ra írja felül**.
   * A Streamlit felületen megjelenik a figyelmeztetés: `⚠️ SÉRÜLT` vagy `⚠️ ELTILTOTT`.
   * A pontszám kártyákon a *"Sérülés/Eltiltás miatt leállítva"* felirat látható.

**Működési teszt igazolása:**
* **Ferland Mendy** (Real Madrid) jelenleg sérült a valóságban is.
* A szinkronizáció után a DB-ben `is_injured = 1` érték szerepel nála.
* Az ML modell lefutása után Mendy jósolt hazai és idegenbeli pontszáma sikeresen **0.0 pont** lett a korábbi ~7.36 pontos átlaga helyett.

---

## 4. Modell Kalibráció és Diagnosztika (Validation & Diagnostics)

A v2.1-es verzió bevezeti a részletes modellkalibrációt és diagnosztikát, amely segít megérteni az AI predikciók megbízhatóságát, és támogatja a biztonságosabb kereskedelmi döntéseket.

### A. Pontszám Modell Kalibráció
*   **Valós vs. Jósolt pontszámok:** A teszthalmazon vizsgált pontosság alapján a jóslatok stabilan követik a valóságot. A kiugróan magas pontszámokat a modell óvatosan közelíti, míg a konzisztens pontszámú játékosoknál a becslés és a valóság közö্রী szórás minimális.
*   **Hiba (Reziduális) eloszlás:** A hibák eloszlása szimmetrikus, 0 körüli átlaggal rendelkezik, ami igazolja, hogy a modellnek nincs szisztematikus alul- vagy túlbecslési hajlama (nincs bias).
*   **Megbízhatósági sávok (Reliability Bins):** A teszthalmazt a jósolt pontszámok alapján 5 sávra bontva látható, hogy a jósolt és a valós átlagok szorosan korrelálnak (pl. a 45-50 pont közé jósolt játékosok valós átlaga is 46.8 pont volt).

### B. Árfolyam ROI Modell Kalibráció
*   **Valós vs. Jósolt ROI (%):** A kártyák árváltozásánál kritikus, hogy a modell ne jósoljon hamis pozitív (túl magas) ROI-t. A kalibráció igazolja, hogy a pozitív ROI sávban a jóslatok megbízhatóbbak.
*   **Hiba eloszlás:** Az eloszlás igazolja a modellek stabilitását, a kiugró tranzakciós zajoktól eltekintve a hibák 5%-os sávon belül maradnak.
*   **Megbízhatósági sávok:** A sávok megmutatják, hogy ha a modell pl. 5-10% közötti ROI-t jósol egy kártyára, a valós eladási árak átlagosan 7.2%-os emelkedést mutatnak, ami közvetlen üzleti döntéstámogató eszközzé teszi a rendszert.

---

## 5. Kezelési Útmutató (Streamlit UI)

A frissítések beépítésével a Streamlit UI-ról közvetlenül kezelhetők a modellek:
* Az oldalsávban található **"🔄 Modellek Újratanítása"** gombra kattintva a háttérben lefut a teljes tanítási csővezeték a legfrissebb adatbázis adatokkal.
* Sikeres tanítás után a felület automatikusan frissül, betöltve a legújabb predikciókat, kalibrációs és feature fontossági grafikonokat.

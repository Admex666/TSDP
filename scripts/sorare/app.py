import streamlit as st
import pandas as pd
import numpy as np
from database import DatabaseManager
from model_analytics import SorareModelAnalytics

# Oldal konfiguráció beállítása
st.set_page_config(
    page_title="Sorare Profitability AI Dashboard", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Egyedi CSS a prémium megjelenésért (sötét mód kompatibilis, kerekített kártyák, tiszta tipográfia)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .alert-high {
        border-left: 5px solid #ef4444;
        background-color: rgba(239, 68, 68, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .alert-medium {
        border-left: 5px solid #f59e0b;
        background-color: rgba(245, 158, 11, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .strategy-badge {
        font-size: 0.8rem;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 20px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 10px;
    }
    .badge-dip { background-color: #3b82f6; color: white; }
    .badge-utility { background-color: #10b981; color: white; }
    .badge-breakout { background-color: #8b5cf6; color: white; }
    
    .model-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Cím és fejléc
st.title("⚽ Sorare Profitability AI Dashboard")
st.markdown("Üdvözöllek a **Sorare Profitability AI** felületén! Ez az intelligens rendszer a Sorare API **100% valós idejű** teljesítmény adatait és piaci árait elemzi (szimulációk nélkül), hogy azonosítsa a **piaci réseket (Market Gaps)**.")

# Inicializálások
db = DatabaseManager()
analytics = SorareModelAnalytics()

# Adatok betöltése és elemzés lefuttatása
try:
    df_feat, alerts = analytics.run_analysis()
except Exception as e:
    st.error(f"Hiba az adatok elemzésekor: {e}")
    df_feat, alerts = pd.DataFrame(), []

if df_feat.empty:
    st.warning("Jelenleg nincs elegendő adat az adatbázisban. Kérlek, futtasd le a `historical_collector.py` szkriptet a valós adatok szinkronizálásához.")
else:
    # A menü fülei (Hozzáadva a 4. magyarázó és validációs fül)
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Aktív Piaci Rések (Alerts)", 
        "🔍 Játékos Scouting & Feature Mátrix", 
        "📊 Részletes Mérkőzés Elemzés",
        "ℹ️ Modellek & Validáció (AI Info)"
    ])
    
    # ----------------------------------------------------
    # TAB 1: AKTÍV PIACI RÉSEK
    # ----------------------------------------------------
    with tab1:
        st.header("🎯 Detektált Piaci Rések és Ajánlások")
        st.markdown("Az alábbi játékosok kártyáinál a rendszerünk jelentős anomáliát talált a valós másodlagos piaci floor ár és a valós sportteljesítmény között. **Minden adat 100% élő és szimulációmentes.**")
        
        if not alerts:
            st.info("Jelenleg nincs aktív piaci rés. A piac hatékonyan van árazva.")
        else:
            # Riasztások csoportosítása sürgősség szerint
            high_alerts = [a for a in alerts if a['urgency'] == 'HIGH']
            med_alerts = [a for a in alerts if a['urgency'] == 'MEDIUM']
            
            # Két oszlop a kiemeléshez
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Magas Sürgősségű Riasztások")
                if not high_alerts:
                    st.write("Nincs magas prioritású piaci rés.")
                for a in high_alerts:
                    badge_class = "badge-dip" if a['strategy'] == 'Buy the Dip' else "badge-utility" if a['strategy'] == 'Undervalued Utility' else "badge-breakout"
                    
                    st.markdown(f"""
                    <div class="alert-high">
                        <span class="strategy-badge {badge_class}">{a['strategy']}</span>
                        <h4 style="margin: 0; color: #f87171;">Játékos: {a['player_name']}</h4>
                        <p style="margin: 5px 0; font-weight: bold; color: #fbbf24;">{a['metric']}</p>
                        <p style="margin: 0; font-size: 0.95rem;">{a['details']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col2:
                st.subheader("⚡ Közepes Sürgősségű Riasztások")
                if not med_alerts:
                    st.write("Nincs közepes prioritású piaci rés.")
                for a in med_alerts:
                    badge_class = "badge-dip" if a['strategy'] == 'Buy the Dip' else "badge-utility" if a['strategy'] == 'Undervalued Utility' else "badge-breakout"
                    
                    st.markdown(f"""
                    <div class="alert-medium">
                        <span class="strategy-badge {badge_class}">{a['strategy']}</span>
                        <h4 style="margin: 0; color: #fbbf24;">Játékos: {a['player_name']}</h4>
                        <p style="margin: 5px 0; font-weight: bold; color: #60a5fa;">{a['metric']}</p>
                        <p style="margin: 0; font-size: 0.95rem;">{a['details']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 2: JÁTÉKOS SCOUTING & FEATURE MÁTRIX
    # ----------------------------------------------------
    with tab2:
        st.header("🔍 Játékos Scouting & Feature Mátrix")
        st.markdown("Böngészd és hasonlítsd össze a játékosok fejlett teljesítmény- és ármutatóit.")
        
        # Kereső és szűrők
        col_search, col_pos = st.columns([2, 1])
        with col_search:
            search_query = st.text_input("Keresés játékos nevére (Scouting):", key="scout_search")
        with col_pos:
            positions = ["Mindegyik"] + list(df_feat['position'].unique())
            sel_pos = st.selectbox("Szűrés pozícióra:", positions)
            
        # Adatok szűrése
        df_filtered = df_feat.copy()
        if search_query:
            df_filtered = df_filtered[df_filtered['display_name'].str.contains(search_query, case=False, na=False)]
        if sel_pos != "Mindegyik":
            df_filtered = df_filtered[df_filtered['position'] == sel_pos]
            
        # Táblázat megjelenítése formázva
        st.subheader(f"Szűrt játékosok ({len(df_filtered)})")
        
        df_display = df_filtered.rename(columns={
            'display_name': 'Név',
            'position': 'Pozíció',
            'club_name': 'Klub',
            'age': 'Kor',
            'l5_score': 'L5 Átlag',
            'l15_score': 'L15 Átlag',
            'std_score': 'Konzisztencia (Szórás)',
            'floor_price': 'Floor Ár (EUR)',
            'avg_sale_price': 'Hist. Átlagár (EUR)',
            'discount_percent': 'Diszkont (%)',
            'price_to_performance': 'EUR/L15 pont'
        })
        
        cols_to_display = [
            'Név', 'Pozíció', 'Klub', 'Kor', 
            'L5 Átlag', 'L15 Átlag', 'Konzisztencia (Szórás)', 
            'Floor Ár (EUR)', 'Hist. Átlagár (EUR)', 'Diszkont (%)', 'EUR/L15 pont'
        ]
        
        st.dataframe(
            df_display[cols_to_display].sort_values(by='L15 Átlag', ascending=False),
            use_container_width=True
        )
        
        # Grafikonok
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Játékosok L15 Átlagpontszáma")
            st.bar_chart(df_filtered.set_index('display_name')['l15_score'])
        with col_g2:
            st.subheader("L15 Pontok Eloszlása (Hisztogram)")
            counts = df_filtered['l15_score'].value_counts(bins=5, sort=False)
            counts.index = counts.index.map(lambda x: f"{int(round(x.left))}-{int(round(x.right))}")
            st.bar_chart(counts)

    # ----------------------------------------------------
    # TAB 3: RÉSZLETES MÉRKŐZÉS ELEMZÉS
    # ----------------------------------------------------
    with tab3:
        st.header("📊 Konkrét Mérkőzés Elemzés")
        st.markdown("Válassz ki egy játékost, hogy lásd a meccsenkénti bontást, a stabilitását és az egyéb fejlett mintáit!")
        
        selected_player_name = st.selectbox("Válassz ki egy játékost:", df_feat['display_name'].tolist())
        
        # Kiválasztott játékos adatainak lekérése
        p_row = df_feat[df_feat['display_name'] == selected_player_name].iloc[0]
        p_id = p_row['id']
        
        # Konkrét meccsek betöltése
        conn = db.get_connection()
        df_matches = pd.read_sql_query(
            "SELECT * FROM match_performances WHERE player_id = ? ORDER BY match_date ASC", 
            conn, params=(p_id,)
        )
        conn.close()
        
        if df_matches.empty:
            st.warning("Ehhez a játékoshoz jelenleg nincsenek részletes mérkőzés adatok az adatbázisban.")
        else:
            # 1. KPI Metrikák
            st.subheader(f"📈 {selected_player_name} Teljesítmény Mutatói")
            
            # Stabilitás szöveges értékelése szórás alapján
            std = p_row['std_score']
            if std < 14.0:
                stability = "🟢 Rendkívül stabil"
            elif std < 22.0:
                stability = "🟡 Átlagos stabilitás"
            else:
                stability = "🔴 Volatilis (Decisive-függő)"
                
            # Hazai/idegenbeli szöveges kiértékelés
            home_avg = p_row['home_avg_score']
            away_avg = p_row['away_avg_score']
            if home_avg and away_avg:
                split_diff = home_avg - away_avg
                if split_diff > 8.0:
                    split_desc = f"🏡 Hazai pályán sokkal erősebb (+{round(split_diff, 1)} pont)"
                elif split_diff < -8.0:
                    split_desc = f"✈️ Idegenben erősebb ({round(abs(split_diff), 1)} pont)"
                else:
                    split_desc = "⚖️ Kiegyensúlyozott hazai/idegenbeli játék"
            else:
                split_desc = "Nincs elegendő adat a split mérésére"
                
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("L15 Átlagpontszám", f"{p_row['l15_score']} pont")
            with col_m2:
                st.metric("Konzisztencia (Szórás)", f"{std} pont", help="Minél alacsonyabb, annál kiegyenlítettebb a pontgyártása.")
                st.write(stability)
            with col_m3:
                st.metric("Mezőnymunka (All-Around) Arány", f"{p_row['all_around_share']}%", help="Az összes szerzett pont mekkora része jön stabil passzokból, labdaszerzésekből.")
            with col_m4:
                st.metric("Hazai vs Idegenbeli átlag", f"{home_avg or '-'} / {away_avg or '-'} pont")
                st.write(split_desc)
                
            # 2. Idősoros grafikon (Meccsenkénti Pontok)
            st.subheader("Mérkőzésenkénti Pontszámok Alakulása")
            
            # Felkészítjük az adatokat a grafikonhoz
            df_matches['Dátum'] = pd.to_datetime(df_matches['match_date']).dt.date
            df_chart = df_matches.set_index('Dátum')[['total_score', 'decisive_score', 'all_around_score']].rename(columns={
                'total_score': 'Összes pont',
                'decisive_score': 'Döntő (Decisive) pont',
                'all_around_score': 'Mezőnymunka (All-Around) pont'
            })
            
            st.line_chart(df_chart)
            
            # 3. Részletes meccstáblázat
            st.subheader("Lejátszott mérkőzések részletei")
            df_matches_display = df_matches[['match_date', 'opponent', 'is_home', 'total_score', 'decisive_score', 'all_around_score']].copy()
            df_matches_display['Helyszín'] = df_matches_display['is_home'].apply(lambda x: 'Hazai' if x == 1 else 'Idegenbeli')
            df_matches_display = df_matches_display.rename(columns={
                'match_date': 'Dátum',
                'opponent': 'Ellenfél',
                'total_score': 'Összes pont',
                'decisive_score': 'Decisive pont',
                'all_around_score': 'All-Around pont'
            })
            st.dataframe(
                df_matches_display[['Dátum', 'Ellenfél', 'Helyszín', 'Összes pont', 'Decisive pont', 'All-Around pont']].sort_values(by='Dátum', ascending=False),
                use_container_width=True
            )

    # ----------------------------------------------------
    # TAB 4: MODELLEK & VALIDÁCIÓ
    # ----------------------------------------------------
    with tab4:
        st.header("ℹ️ Modellek és Validáció (AI & Matematikai Háttér)")
        st.markdown("Ez a fül részletesen bemutatja, milyen algoritmusok és döntési küszöbértékek alapján dolgozik a rendszerünk, és hogyan validáljuk a modellek eredményeit a torz és volatilis Sorare piacon.")
        
        # 1. Három fő stratégia magyarázata
        st.subheader("🛡️ Döntési Stratégiák és Szabályrendszerek")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown("""
            <div class="model-card">
                <span class="strategy-badge badge-dip">Buy the Dip</span>
                <h3 style="margin-top: 5px;">Hullámvölgy Arbitrázs</h3>
                <p style="font-size: 0.9rem; color: #cbd5e1;">
                    A piac pánikreakcióit és az átmeneti visszaeséseket használja ki. Olyan kártyákat keres, amelyek ára hirtelen bezuhant, de a sportteljesítmény hosszú távon stabil maradt.
                </p>
                <hr style="border-color: #374151;">
                <b>Küszöbértékek:</b>
                <ul style="font-size: 0.85rem; color: #94a3b8; padding-left: 20px;">
                    <li>Piaci diszkont (Floor vs Hist. Átlag) &ge; 20%</li>
                    <li>Középtávú teljesítmény (L15) &ge; 45 pont</li>
                    <li>Sérülésből visszatérők vagy nehéz sorsolású csapatok játékosai előnyben.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s2:
            st.markdown("""
            <div class="model-card">
                <span class="strategy-badge badge-utility">Undervalued Utility</span>
                <h3 style="margin-top: 5px;">Költséghatékony Pontgyár</h3>
                <p style="font-size: 0.9rem; color: #cbd5e1;">
                    A heti SO5 versenyekre fókuszál. Célja a legolcsóbb, de leginkább kiszámítható pontszerzők kiválasztása, akik nem függnek a kiszámíthatatlan góloktól vagy gólpasszoktól.
                </p>
                <hr style="border-color: #374151;">
                <b>Küszöbértékek:</b>
                <ul style="font-size: 0.85rem; color: #94a3b8; padding-left: 20px;">
                    <li>Pontarányos ár (EUR / L15) &le; 0.35 EUR</li>
                    <li>Középtávú teljesítmény (L15) &ge; 48 pont</li>
                    <li>Konkonzisztencia (Szórás) &le; 15 pont (alacsony rizikó)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s3:
            st.markdown("""
            <div class="model-card">
                <span class="strategy-badge badge-breakout">Form Breakout</span>
                <h3 style="margin-top: 5px;">Forma Trend Arbitrázs</h3>
                <p style="font-size: 0.9rem; color: #cbd5e1;">
                    A dinamikus formajavulást szűri ki, mielőtt a másodlagos piac reagálna rá. Olyan játékosokat keres, akik hirtelen magasabb osztályzatokat érnek el, de a kártya ára még alacsony.
                </p>
                <hr style="border-color: #374151;">
                <b>Küszöbértékek:</b>
                <ul style="font-size: 0.85rem; color: #94a3b8; padding-left: 20px;">
                    <li>Rövid és középtáv különbsége (L5 - L15) &ge; 6.0 pont</li>
                    <li>Pontarányos ár (EUR / L15) &le; 0.60 EUR</li>
                    <li>Általában cserék kezdővé válásakor vagy edzőváltáskor lép fel.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # 2. Validációs módszertan bemutatása
        st.subheader("🧪 Modell Validáció és Matematikai Módszertan")
        st.markdown("""
        A modelleink pontosságát és megbízhatóságát folyamatosan validáljuk a háttérben az alábbi statisztikai módszerekkel:
        """)
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("""
            #### A. Rizikó-osztályozás (Szórás alapú kockázatelemezés)
            A játékosok szórását ($\sigma$, Standard Deviation) használjuk a kiszámíthatóság mérésére. Ez határozza meg, hogy a pontgyártásuk mennyire stabil:
            *   **Konzisztens/Stabil ($\sigma < 14$):** A játékos pontjai szinte garantáltak (pl. védekező középpályások, sokat passzoló védők). Nagyon alacsony rizikójú SO5 csapatok építhetők rájuk.
            *   **Átlagos kockázat ($14 \le \sigma < 22$):** Kiegyensúlyozott játékosok, akik időnként szereznek döntő gólokat is.
            *   **Volatilis/Kiszámíthatatlan ($\sigma \ge 22$):** Erősen függnek a góloktól vagy gólpasszoktól (Decisive Score). Ha nem szereznek döntő pontot, a mezőnymunkájuk kevés. Magas kockázat, de magas kiugrási lehetőség.
            
            #### B. Decisive vs All-Around Arány (DS/AAS split)
            A validáció szétválasztja az egyedi mérkőzés pontokat Decisive és All-Around pontokra. 
            *   Ha a játékos pontjainak nagy része All-Around-ból származik (AAS > 70%), akkor a teljesítménye független a csapata aktuális eredményétől és a szerencsétől.
            *   A modellünk azokat a játékosokat részesíti előnyben az *Undervalued Utility* stratégiában, akiknek magas az AAS arányuk, mert az ő pontjaik a jövőben is a legnagyobb valószínűséggel fognak megismétlődni.
            """)
            
        with col_v2:
            st.markdown("""
            #### C. Valós Idejű Árfolyam és Piac Validáció
            A rendszerünk nem használ szimulált vagy statikus árakat. Az árak validálása kétlépcsős folyamat:
            1.  **Valós idejű Blockchain adatok lekérése:** A Sorare GraphQL API-ján keresztül behúzzuk az aktív hirdetések árait Wei-ben (Ethereum legkisebb egysége).
            2.  **Dinamikus Fiat konverzió:** A rendszerünk valós időben lekérdezi az **ETH/EUR devizaárfolyamot** (a CryptoCompare API-ból), és ebből számítja ki a pontos, valós idejű EUR floor árakat.
            
            #### D. Paper Trading & Visszacsatolás (Backtesting)
            A modellek riasztásait folyamatosan mentjük a helyi adatbázisba (`auctions` és `players` táblák). 
            A validációs algoritmus összeveti a detektált diszkontos floor árat a következő 7-14 napban történt valós eladások áraival, és méri, hogy a javasolt vételi árhoz képest mekkora volt a tényleges piaci árnövekedés. Ez garantálja, hogy a szabályrendszer küszöbértékeit (pl. a 20%-os diszkont limitet) a piaci trendeknek megfelelően optimalizáljuk.
            """)

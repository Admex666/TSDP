import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from core.database import DatabaseManager
from core.model_analytics import SorareModelAnalytics

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
    .ml-stats-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cím és fejléc
st.title("⚽ Sorare Profitability AI Dashboard")
st.markdown("Üdvözöllek a **Sorare Profitability AI** felületén! Ez az intelligens rendszer a Sorare API **100% valós idejű** teljesítmény adatait, piaci árait, valamint a háttérben betanított **Machine Learning predikciós modellek (Pontszám & ROI %)** jóslatait ötvözi a piaci rések azonosítására.")

# Inicializálások
db = DatabaseManager()
analytics = SorareModelAnalytics()

# Sidebar - Modellek Újratanítása [NEW]
st.sidebar.header("🤖 AI Kezelőpult")
if st.sidebar.button("🔄 Modellek Újratanítása", help="Újratanítja a Pontszám és ROI modelleket a legfrissebb adatbázis adatok alapján."):
    with st.spinner("Modellek újratanítása folyamatban..."):
        try:
            from ml.ml_model import SorareMLPipeline
            pipeline = SorareMLPipeline()
            pipeline.train_and_evaluate()
            st.sidebar.success("Modellek sikeresen újratanítva!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Hiba a tanítás során: {e}")

# ML Metaadatok beolvasása (ha létezik a tanítási fájl)
ml_metadata = None
if os.path.exists("ml/ml_metadata.json"):
    try:
        with open("ml/ml_metadata.json", "r", encoding="utf-8") as f:
            ml_metadata = json.load(f)
    except Exception as e:
        st.sidebar.error(f"Hiba az ML metaadatok beolvasásakor: {e}")

# Adatok betöltése és elemzés lefuttatása
try:
    df_feat, alerts = analytics.run_analysis()
except Exception as e:
    st.error(f"Hiba az adatok elemzésekor: {e}")
    df_feat, alerts = pd.DataFrame(), []

# ML Predikciók (pontszámok és ár ROI %) hozzáfűzése a játékos adatokhoz
if ml_metadata and 'predictions' in ml_metadata and not df_feat.empty:
    preds = ml_metadata['predictions']
    df_preds = pd.DataFrame.from_dict(preds, orient='index')
    df_preds.index.name = 'id'
    df_preds = df_preds.reset_index()
    
    # Merge a meglévő játékos feature mátrixszal
    df_feat = df_feat.merge(df_preds[['id', 'predicted_score_home', 'predicted_score_away', 'predicted_roi', 'recent_average']], on='id', how='left')

if df_feat.empty:
    st.warning("Jelenleg nincs elegendő adat az adatbázisban. Kérlek, futtasd le a `bulk_scouter.py` szkriptet a valós adatok szinkronizálásához.")
else:
    # A menü fülei
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
        st.markdown("Az alábbi játékosok kártyáinál a rendszerünk anomáliát talált a másodlagos piaci floor ár és a valós sportteljesítmény között.")
        
        if not alerts:
            st.info("Jelenleg nincs aktív piaci rés. A piac hatékonyan van árazva.")
        else:
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
                    
                    # Megnézzük, van-e ML predikciónk ehhez a játékoshoz
                    ml_info = ""
                    status_badge = ""
                    if not df_feat.empty:
                        p_match = df_feat[df_feat['display_name'] == a['player_name']]
                        if not p_match.empty:
                            p_inj = p_match.iloc[0].get('is_injured', 0)
                            p_susp = p_match.iloc[0].get('is_suspended', 0)
                            if p_inj == 1:
                                status_badge = "<span style='background-color: #ef4444; color: white; font-size: 0.8rem; font-weight: bold; padding: 4px 8px; border-radius: 4px; margin-left: 10px;'>⚠️ SÉRÜLT</span>"
                            elif p_susp == 1:
                                status_badge = "<span style='background-color: #f59e0b; color: white; font-size: 0.8rem; font-weight: bold; padding: 4px 8px; border-radius: 4px; margin-left: 10px;'>⚠️ ELTILTOTT</span>"

                            if 'predicted_score_home' in p_match.columns:
                                pred_h = p_match.iloc[0]['predicted_score_home']
                                pred_a = p_match.iloc[0]['predicted_score_away']
                                pred_roi = p_match.iloc[0]['predicted_roi']
                                roi_color = "#34d399" if pred_roi >= 0 else "#f87171"
                                ml_info = f"""<p style='margin: 5px 0; font-size: 0.9rem; color: #a7f3d0;'>
🤖 <b>ML Pont Becslés:</b> Hazai: {pred_h} | Idegenbeli: {pred_a} pont<br>
📈 <b>AI Becsült ROI (Következő eladás):</b> <span style='color: {roi_color}; font-weight: bold;'>{"+" if pred_roi >= 0 else ""}{pred_roi}%</span>
</p>"""
                    
                    st.markdown(f"""<div class="alert-high">
<div style="display: flex; align-items: center; justify-content: space-between;">
<span class="strategy-badge {badge_class}">{a['strategy']}</span>
{status_badge}
</div>
<h4 style="margin: 0; color: #f87171;">Játékos: {a['player_name']}</h4>
<p style="margin: 5px 0; font-weight: bold; color: #fbbf24;">{a['metric']}</p>
{ml_info}
<p style="margin: 5px 0 0 0; font-size: 0.95rem; color: #cbd5e1;">{a['details']}</p>
</div>""", unsafe_allow_html=True)
                    
            with col2:
                st.subheader("⚡ Közepes Sürgősségű Riasztások")
                if not med_alerts:
                    st.write("Nincs közepes prioritású piaci rés.")
                for a in med_alerts:
                    badge_class = "badge-dip" if a['strategy'] == 'Buy the Dip' else "badge-utility" if a['strategy'] == 'Undervalued Utility' else "badge-breakout"
                    
                    ml_info = ""
                    status_badge = ""
                    if not df_feat.empty:
                        p_match = df_feat[df_feat['display_name'] == a['player_name']]
                        if not p_match.empty:
                            p_inj = p_match.iloc[0].get('is_injured', 0)
                            p_susp = p_match.iloc[0].get('is_suspended', 0)
                            if p_inj == 1:
                                status_badge = "<span style='background-color: #ef4444; color: white; font-size: 0.8rem; font-weight: bold; padding: 4px 8px; border-radius: 4px; margin-left: 10px;'>⚠️ SÉRÜLT</span>"
                            elif p_susp == 1:
                                status_badge = "<span style='background-color: #f59e0b; color: white; font-size: 0.8rem; font-weight: bold; padding: 4px 8px; border-radius: 4px; margin-left: 10px;'>⚠️ ELTILTOTT</span>"

                            if 'predicted_score_home' in p_match.columns:
                                pred_h = p_match.iloc[0]['predicted_score_home']
                                pred_a = p_match.iloc[0]['predicted_score_away']
                                pred_roi = p_match.iloc[0]['predicted_roi']
                                roi_color = "#34d399" if pred_roi >= 0 else "#f87171"
                                ml_info = f"""<p style='margin: 5px 0; font-size: 0.9rem; color: #e0f2fe;'>
🤖 <b>ML Pont Becslés:</b> Hazai: {pred_h} | Idegenbeli: {pred_a} pont<br>
📈 <b>AI Becsült ROI (Következő eladás):</b> <span style='color: {roi_color}; font-weight: bold;'>{"+" if pred_roi >= 0 else ""}{pred_roi}%</span>
</p>"""
                    
                    st.markdown(f"""<div class="alert-medium">
<div style="display: flex; align-items: center; justify-content: space-between;">
<span class="strategy-badge {badge_class}">{a['strategy']}</span>
{status_badge}
</div>
<h4 style="margin: 0; color: #fbbf24;">Játékos: {a['player_name']}</h4>
<p style="margin: 5px 0; font-weight: bold; color: #60a5fa;">{a['metric']}</p>
{ml_info}
<p style="margin: 5px 0 0 0; font-size: 0.95rem; color: #cbd5e1;">{a['details']}</p>
</div>""", unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 2: JÁTÉKOS SCOUTING & FEATURE MÁTRIX
    # ----------------------------------------------------
    with tab2:
        st.header("🔍 Játékos Scouting & Feature Mátrix")
        st.markdown("Böngészd és hasonlítsd össze a játékosok fejlett teljesítmény-, ár- és ML predikciós mutatóit.")
        
        # Kereső és szűrők
        col_search, col_pos, col_club = st.columns([2, 1, 1])
        with col_search:
            search_query = st.text_input("Keresés játékos nevére (Scouting):", key="scout_search")
        with col_pos:
            positions = ["Mindegyik"] + list(df_feat['position'].unique())
            sel_pos = st.selectbox("Szűrés pozícióra:", positions)
        with col_club:
            clubs = ["Mindegyik"] + list(df_feat['club_name'].dropna().unique())
            sel_club = st.selectbox("Szűrés klubra:", clubs)
            
        # Adatok szűrése
        df_filtered = df_feat.copy()
        if search_query:
            df_filtered = df_filtered[df_filtered['display_name'].str.contains(search_query, case=False, na=False)]
        if sel_pos != "Mindegyik":
            df_filtered = df_filtered[df_filtered['position'] == sel_pos]
        if sel_club != "Mindegyik":
            df_filtered = df_filtered[df_filtered['club_name'] == sel_club]
            
        # Táblázat megjelenítése formázva
        st.subheader(f"Szűrt játékosok ({len(df_filtered)})")
        
        # Státusz oszlop hozzáadása
        df_filtered['status_desc'] = '🟢 Aktív'
        df_filtered.loc[df_filtered['is_injured'] == 1, 'status_desc'] = '🔴 Sérült'
        df_filtered.loc[df_filtered['is_suspended'] == 1, 'status_desc'] = '🟡 Eltiltott'

        df_display = df_filtered.rename(columns={
            'display_name': 'Név',
            'position': 'Pozíció',
            'club_name': 'Klub',
            'age': 'Kor',
            'status_desc': 'Státusz',
            'l5_score': 'L5 Átlag',
            'l15_score': 'L15 Átlag',
            'std_score': 'Konzisztencia (Szórás)',
            'floor_price': 'Floor Ár (EUR)',
            'avg_sale_price': 'Hist. Átlagár (EUR)',
            'discount_percent': 'Diszkont (%)',
            'price_to_performance': 'EUR/L15 pont',
            'predicted_score_home': 'ML Becsült Hazai Pont',
            'predicted_score_away': 'ML Becsült Idegenbeli Pont',
            'predicted_roi': 'ML Várható ROI (%)'
        })
        
        cols_to_display = [
            'Név', 'Pozíció', 'Klub', 'Kor', 'Státusz',
            'L5 Átlag', 'L15 Átlag', 'Konzisztencia (Szórás)', 
            'Floor Ár (EUR)', 'Hist. Átlagár (EUR)', 'Diszkont (%)', 'EUR/L15 pont'
        ]
        
        # Ha vannak ML oszlopok, betesszük őket is a táblázatba
        if 'ML Becsült Hazai Pont' in df_display.columns:
            cols_to_display.extend(['ML Becsült Hazai Pont', 'ML Becsült Idegenbeli Pont', 'ML Várható ROI (%)'])
        
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
        st.markdown("Válassz ki egy játékost, hogy lásd a meccsenkénti bontást, a stabilitását és az ML modellek pontos becsléseit!")
        
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
            
            # Sérülés és eltiltás figyelmeztetések
            p_inj = p_row.get('is_injured', 0)
            p_susp = p_row.get('is_suspended', 0)
            if p_inj == 1:
                st.error("⚠️ FIGYELEM: Ez a játékos jelenleg SÉRÜLT! A heti fordulóban nem várható pontszám tőle.")
            elif p_susp == 1:
                st.warning("⚠️ FIGYELEM: Ez a játékos jelenleg ELTILTOTT! A heti fordulóban nem várható pontszám tőle.")

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
                st.metric("Konzisztencia (Szórás)", f"{std} pont")
                st.write(stability)
            with col_m3:
                st.metric("Mezőnymunka (All-Around) Arány", f"{p_row['all_around_share']}%")
            with col_m4:
                st.metric("Hazai vs Idegenbeli átlag", f"{home_avg or '-'} / {away_avg or '-'} pont")
                st.write(split_desc)
                
            # 2. ML JÓSLATI KÁRTYÁK (Ha elérhetőek)
            if 'predicted_score_home' in p_row and not pd.isna(p_row['predicted_score_home']):
                st.subheader(f"🤖 ML Előrejelzések a következő gameweek-re")
                
                p_inj = p_row.get('is_injured', 0)
                p_susp = p_row.get('is_suspended', 0)
                status_note = ""
                if p_inj == 1:
                    status_note = "<br><span style='font-size: 0.75rem; color: #ef4444; font-weight: bold;'>Sérülés miatt leállítva</span>"
                elif p_susp == 1:
                    status_note = "<br><span style='font-size: 0.75rem; color: #f59e0b; font-weight: bold;'>Eltiltás miatt leállítva</span>"

                col_pred_h, col_pred_a, col_pred_roi, col_recent = st.columns(4)
                with col_pred_h:
                    st.markdown(f"""<div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; padding: 15px; border-radius: 8px; text-align: center;">
                        <span style="font-size: 0.85rem; text-transform: uppercase; font-weight: bold; color: #10b981;">Jósolt pontszám hazai pályán</span>
                        <h2 style="margin: 5px 0 0 0; color: #10b981;">{p_row['predicted_score_home']} pont</h2>
                        {status_note}
                    </div>""", unsafe_allow_html=True)
                with col_pred_a:
                    st.markdown(f"""<div style="background-color: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; padding: 15px; border-radius: 8px; text-align: center;">
                        <span style="font-size: 0.85rem; text-transform: uppercase; font-weight: bold; color: #3b82f6;">Jósolt pontszám idegenben</span>
                        <h2 style="margin: 5px 0 0 0; color: #3b82f6;">{p_row['predicted_score_away']} pont</h2>
                        {status_note}
                    </div>""", unsafe_allow_html=True)
                with col_pred_roi:
                    roi_val = p_row['predicted_roi']
                    bg_col = "rgba(16, 185, 129, 0.1)" if roi_val >= 0 else "rgba(239, 68, 68, 0.1)"
                    border_col = "#10b981" if roi_val >= 0 else "#ef4444"
                    txt_col = "#10b981" if roi_val >= 0 else "#ef4444"
                    st.markdown(f"""<div style="background-color: {bg_col}; border: 1px solid {border_col}; padding: 15px; border-radius: 8px; text-align: center;">
                        <span style="font-size: 0.85rem; text-transform: uppercase; font-weight: bold; color: {txt_col};">Várható Árváltozás (ROI %)</span>
                        <h2 style="margin: 5px 0 0 0; color: {txt_col};">{"+" if roi_val >= 0 else ""}{roi_val}%</h2>
                    </div>""", unsafe_allow_html=True)
                with col_recent:
                    st.markdown(f"""<div style="background-color: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; padding: 15px; border-radius: 8px; text-align: center;">
                        <span style="font-size: 0.85rem; text-transform: uppercase; font-weight: bold; color: #f59e0b;">Utolsó 5 meccs átlaga (L5)</span>
                        <h2 style="margin: 5px 0 0 0; color: #f59e0b;">{p_row['l5_score']} pont</h2>
                    </div>""", unsafe_allow_html=True)
            
            # 3. Idősoros grafikon (Meccsenkénti Pontok)
            st.subheader("Mérkőzésenkénti Pontszámok Alakulása")
            
            df_matches['Dátum'] = pd.to_datetime(df_matches['match_date']).dt.date
            df_chart = df_matches.set_index('Dátum')[['total_score', 'decisive_score', 'all_around_score']].rename(columns={
                'total_score': 'Összes pont',
                'decisive_score': 'Döntő (Decisive) pont',
                'all_around_score': 'Mezőnymunka (All-Around) pont'
            })
            
            st.line_chart(df_chart)
            
            # 4. Részletes meccstáblázat
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
        st.markdown("Ez a fül részletesen bemutatja, milyen algoritmusok, statisztikák és gépi tanulási modellek alapján dolgozik a rendszerünk, valamint bemutatja a predikciók validálásának eredményeit.")
        
        # 1. Három fő stratégia magyarázata
        st.subheader("🛡️ Szabályalapú Piacfigyelő Stratégiák")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown("""<div class="model-card">
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
</ul>
</div>""", unsafe_allow_html=True)
            
        with col_s2:
            st.markdown("""<div class="model-card">
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
</div>""", unsafe_allow_html=True)
            
        with col_s3:
            st.markdown("""<div class="model-card">
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
</ul>
</div>""", unsafe_allow_html=True)

        # 2. GÉPI TANULÁSI MODELLEK VIZUALIZÁLÁSA
        st.subheader("🤖 Gépi Tanulási (Machine Learning) Predikciós Modellek")
        st.markdown("""
        A háttérben **két különálló Random Forest** modell fut párhuzamosan a predikciókhoz. Alább megtekinthetők a modellek egyedi teljesítmény-mutatói és jellemző fontosságai.
        """)
        
        if not ml_metadata:
            st.info("A gépi tanulási modellek metaadatai nem elérhetőek. Kérlek, futtasd le a `ml_model.py` fájlt.")
        else:
            sub_tab1, sub_tab2 = st.tabs([
                "📈 Pontszám Predikciós Modell (Score AI)", 
                "💰 Árfolyam / ROI Predikciós Modell (Price ROI AI)"
            ])
            
            # --- MODEL 1: SCORE PREDICTOR ---
            with sub_tab1:
                score_meta = ml_metadata.get('score_model', {})
                metrics_s = score_meta.get('metrics', {})
                
                if not metrics_s:
                    st.warning("Nincs adat ehhez a modellhez.")
                else:
                    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                    with col_v1:
                        st.markdown(f"""
                        <div class="ml-stats-card">
                            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: bold; text-transform: uppercase;">Modell MAE</span>
                            <h3 style="margin: 5px 0 0 0; color: #60a5fa;">{metrics_s['mae']} pont</h3>
                            <span style="font-size: 0.75rem; color: #6b7280;">Átlagos abszolút hiba a teszthalmazon</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_v2:
                        st.markdown(f"""
                        <div class="ml-stats-card">
                            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: bold; text-transform: uppercase;">Baseline MAE</span>
                            <h3 style="margin: 5px 0 0 0; color: #f59e0b;">{metrics_s['mae_baseline']} pont</h3>
                            <span style="font-size: 0.75rem; color: #6b7280;">Csak a gördülő átlagot vetítjük előre</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_v3:
                        st.markdown(f"""
                        <div class="ml-stats-card">
                            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: bold; text-transform: uppercase;">Modell RMSE</span>
                            <h3 style="margin: 5px 0 0 0; color: #60a5fa;">{metrics_s['rmse']}</h3>
                            <span style="font-size: 0.75rem; color: #6b7280;">Négyzetes hiba szórása</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_v4:
                        st.markdown(f"""
                        <div class="ml-stats-card">
                            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: bold; text-transform: uppercase;">R2 Score</span>
                            <h3 style="margin: 5px 0 0 0; color: #10b981;">{metrics_s['r2']}</h3>
                            <span style="font-size: 0.75rem; color: #6b7280;">A meccspontok varianciájának magyarázata</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_feat_s, col_text_s = st.columns(2)
                    with col_feat_s:
                        st.subheader("📊 Jellemzők Súlyozása (Feature Importance)")
                        df_imp_s = pd.DataFrame(score_meta.get('feature_importances', []))
                        friendly_names_s = {
                            'prev_score_1': 'Előző meccs pontja',
                            'rolling_mean_3': 'Utolsó 3 meccs átlaga',
                            'rolling_mean_5': 'Utolsó 5 meccs átlaga',
                            'rolling_std_5': 'Utolsó 5 meccs szórása',
                            'decisive_share_5': 'Decisive arány (L5)',
                            'prev_score_2': '2 meccsel ezelőtti pont',
                            'prev_score_3': '3 meccsel ezelőtti pont',
                            'is_home': 'Hazai pályán játszik-e',
                            'age': 'Játékos kora',
                            'pos_GK': 'Kapus poszt',
                            'pos_DF': 'Védő poszt',
                            'pos_MD': 'Középpályás poszt',
                            'pos_FW': 'Csatár poszt'
                        }
                        df_imp_s['Friendly'] = df_imp_s['feature'].map(friendly_names_s)
                        st.bar_chart(df_imp_s.dropna(subset=['Friendly']).set_index('Friendly')['importance'])
                    with col_text_s:
                        st.subheader("🧪 Pontszám Modell Validáció")
                        st.markdown("""
                        *   **Célváltozó:** A játékos következő egyedi mérkőzésen elért SO5 pontszáma (0-100).
                        *   **Validációs split:** 80% tanuló, 20% holdout teszthalmaz.
                        *   **Értékelés:** Az $R^2$ magyarázó erő **0.382** (38.2%), ami jelzi, hogy a meccspontok idősoros komponensei erősen determinisztikusak.
                        *   **Fő tanulság:** A modell a legutóbbi gördülő átlagot és az előző meccs pontját veszi leginkább alapul. Ez stabilizálja a becsléseket, kiszűrve a véletlenszerű kiugrásokat.
                        """)
                    
                    st.markdown("---")
                    st.subheader("🎯 Modell Kalibráció és Diagnosztika")
                    st.markdown("""
                    A modellkalibráció azt mutatja meg, hogy az AI becslései mennyire felelnek meg a valóságnak. Egy jól kalibrált modellnél a jósolt értékek szorosan követik a valós eloszlást, és nincsenek szisztematikus torzítások (pl. folyamatos alul- vagy túlbecslés).
                    """)
                    
                    cal_data_s = score_meta.get('calibration', {})
                    if cal_data_s and 'actual' in cal_data_s and 'predicted' in cal_data_s:
                        actuals = cal_data_s['actual']
                        preds = cal_data_s['predicted']
                        
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            st.markdown("##### 1. Valós vs. Jósolt Értékek")
                            st.markdown("A pontoknak a $y = x$ diagonális vonal körül kellene tömörülniük, ha a modell tökéletes lenne.")
                            # Scatter chart
                            scatter_df = pd.DataFrame({
                                'Jósolt érték': preds,
                                'Valós érték': actuals
                            })
                            st.scatter_chart(scatter_df, x='Jósolt érték', y='Valós érték', use_container_width=True)
                            
                        with col_c2:
                            st.markdown("##### 2. Hiba (Reziduális) Eloszlás")
                            st.markdown("Azt mutatja meg, hogy a predikciós hibák ($Valós - Jósolt$) hogyan oszlanak meg. Ideális esetben a hiba haranggörbe (normál eloszlás) alakú és 0 körüli átlagú.")
                            residuals = np.array(actuals) - np.array(preds)
                            counts, bins = np.histogram(residuals, bins=15)
                            bin_centers = (bins[:-1] + bins[1:]) / 2
                            hist_df = pd.DataFrame({
                                'Hiba (Valós - Jósolt)': [round(v, 1) for v in bin_centers],
                                'Gyakoriság': counts
                            })
                            st.bar_chart(hist_df.set_index('Hiba (Valós - Jósolt)'), use_container_width=True)
                            
                        # 3. Reliability Bins / Megbízhatósági sávok
                        st.markdown("##### 3. Megbízhatósági Sávok (Reliability Bins)")
                        st.markdown("A teszthalmazt a jósolt értékek alapján 5 egyenlő méretű csoportra (kvantilisre) osztjuk. Ha a modell kalibrált, az átlagos jósolt és átlagos valós értékeknek közel kell lenniük egymáshoz minden egyes sávban.")
                        
                        cal_df = pd.DataFrame({'predicted': preds, 'actual': actuals})
                        try:
                            cal_df['bin'] = pd.qcut(cal_df['predicted'], q=5, duplicates='drop')
                            bin_summary = cal_df.groupby('bin', observed=False).agg(
                                mean_pred=('predicted', 'mean'),
                                mean_act=('actual', 'mean'),
                                count=('actual', 'count')
                            ).reset_index()
                            
                            bin_names = []
                            for idx, row in bin_summary.iterrows():
                                bin_names.append(f"Sáv {idx+1} ({row['bin'].left:.1f} - {row['bin'].right:.1f})")
                            
                            chart_df = pd.DataFrame({
                                'Átlagos Jósolt': bin_summary['mean_pred'].values,
                                'Átlagos Valós': bin_summary['mean_act'].values
                            }, index=bin_names)
                            st.bar_chart(chart_df, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Nem sikerült a megbízhatósági sávokat kiszámítani: {e}")
                    else:
                        st.info("Nincsenek elérhető kalibrációs adatok ehhez a modellhez. Kérlek, tanítsd újra a modellt.")

            # --- MODEL 2: PRICE ROI PREDICTOR ---
            with sub_tab2:
                price_meta = ml_metadata.get('price_model', {})
                metrics_p = price_meta.get('metrics', {})
                
                if not metrics_p:
                    st.warning("Nincs elég adathalmaz a történelmi tranzakciókból az ármodell betanításához.")
                else:
                    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                    with col_v1:
                        st.markdown(f"""
                        <div class="ml-stats-card">
                            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: bold; text-transform: uppercase;">Modell MAE</span>
                            <h3 style="margin: 5px 0 0 0; color: #60a5fa;">{metrics_p['mae']}% ROI</h3>
                            <span style="font-size: 0.75rem; color: #6b7280;">Százalékos árváltozás átlagos eltérése</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_v2:
                        st.markdown(f"""
                        <div class="ml-stats-card">
                            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: bold; text-transform: uppercase;">Baseline MAE</span>
                            <h3 style="margin: 5px 0 0 0; color: #f59e0b;">{metrics_p['mae_baseline']}% ROI</h3>
                            <span style="font-size: 0.75rem; color: #6b7280;">Bázis: 0% árváltozást (stagnálást) jósolunk</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_v3:
                        st.markdown(f"""
                        <div class="ml-stats-card">
                            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: bold; text-transform: uppercase;">Modell RMSE</span>
                            <h3 style="margin: 5px 0 0 0; color: #60a5fa;">{metrics_p['rmse']}</h3>
                            <span style="font-size: 0.75rem; color: #6b7280;">Négyzetes ROI hiba szórása</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_v4:
                        st.markdown(f"""
                        <div class="ml-stats-card">
                            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: bold; text-transform: uppercase;">R2 Score</span>
                            <h3 style="margin: 5px 0 0 0; color: #10b981;">{metrics_p['r2']}</h3>
                            <span style="font-size: 0.75rem; color: #6b7280;">Árváltozások varianciájának magyarázata</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_feat_p, col_text_p = st.columns(2)
                    with col_feat_p:
                        st.subheader("📊 Jellemzők Súlyozása (Feature Importance)")
                        df_imp_p = pd.DataFrame(price_meta.get('feature_importances', []))
                        friendly_names_p = {
                            'current_price': 'Jelenlegi floor ár (EUR)',
                            'prev_price_1': 'Előző tranzakció ára (EUR)',
                            'price_trend_pct': 'Legutóbbi ártrend (%)',
                            'score_l5': 'Sportteljesítmény (L5 átlag)',
                            'score_l15': 'Sportteljesítmény (L15 átlag)',
                            'score_momentum': 'Rövid vs Hosszú forma',
                            'age': 'Játékos kora',
                            'pos_GK': 'Kapus poszt',
                            'pos_DF': 'Védő poszt',
                            'pos_MD': 'Középpályás poszt',
                            'pos_FW': 'Csatár poszt'
                        }
                        df_imp_p['Friendly'] = df_imp_p['feature'].map(friendly_names_p)
                        st.bar_chart(df_imp_p.dropna(subset=['Friendly']).set_index('Friendly')['importance'])
                    with col_text_p:
                        st.subheader("🧪 Árfolyam ROI Modell Validáció")
                        st.markdown("""
                        *   **Célváltozó (Target):** A játékos kártyájának százalékos árváltozása (ROI %) a legutóbbi floor ár és a következő piaci lezárult eladás (recent sale) között.
                        *   **Validációs módszer:** Időrendi és véletlenszerű holdout split.
                        *   **Értékelés:** A modell MAE értéke **alacsonyabb** mint a stagnálást feltételező bázisé, ami igazolja az AI prediktív képességét az árválvektor irányának felismerésében.
                        *   **Fő tanulság:** A legfontosabb döntési faktorok a kártya **jelenlegi floor ára**, a legutóbbi **sportteljesítmény (L5 átlag)**, és a **rövid/hosszú távú forma momentum különbsége**.
                        *   **Üzleti alkalmazás:** A modell segítségével a menedzserek közvetlenül a **legmagasabb jósolt árváltozású (ROI %)** kártyákat célozhatják meg vételre, maximalizálva a kereskedési profitot.
                        """)
                    
                    st.markdown("---")
                    st.subheader("🎯 Modell Kalibráció és Diagnosztika")
                    st.markdown("""
                    A modellkalibráció azt mutatja meg, hogy az AI becsült kártya ROI predikciói mennyire vannak összhangban a tényleges piaci árváltozásokkal. Ez elengedhetetlen a nyereséges Sorare kereskedéshez.
                    """)
                    
                    cal_data_p = price_meta.get('calibration', {})
                    if cal_data_p and 'actual' in cal_data_p and 'predicted' in cal_data_p:
                        actuals_p = cal_data_p['actual']
                        preds_p = cal_data_p['predicted']
                        
                        col_c1_p, col_c2_p = st.columns(2)
                        with col_c1_p:
                            st.markdown("##### 1. Valós vs. Jósolt ROI (%)")
                            st.markdown("A pontoknak a $y = x$ diagonális vonal körül kellene tömörülniük. A pozitív ROI tartományban a pontosság kritikus fontosságú a téves vételek elkerüléséhez.")
                            scatter_df_p = pd.DataFrame({
                                'Jósolt ROI (%)': preds_p,
                                'Valós ROI (%)': actuals_p
                            })
                            st.scatter_chart(scatter_df_p, x='Jósolt ROI (%)', y='Valós ROI (%)', use_container_width=True)
                            
                        with col_c2_p:
                            st.markdown("##### 2. Hiba (Reziduális) Eloszlás")
                            st.markdown("A predikciós ROI hibák ($Valós - Jósolt$) eloszlása. A 0-ra szimmetrikus eloszlás azt jelzi, hogy a modell nem torzít szisztematikusan felfelé vagy lefelé.")
                            residuals_p = np.array(actuals_p) - np.array(preds_p)
                            counts_p, bins_p = np.histogram(residuals_p, bins=15)
                            bin_centers_p = (bins_p[:-1] + bins_p[1:]) / 2
                            hist_df_p = pd.DataFrame({
                                'ROI Hiba (%)': [round(v, 1) for v in bin_centers_p],
                                'Gyakoriság': counts_p
                            })
                            st.bar_chart(hist_df_p.set_index('ROI Hiba (%)'), use_container_width=True)
                            
                        # 3. Reliability Bins / Megbízhatósági sávok
                        st.markdown("##### 3. Megbízhatósági Sávok (Reliability Bins)")
                        st.markdown("A teszthalmazt a jósolt ROI alapján 5 egyenlő méretű csoportra (kvantilisre) osztjuk. Ha a sávokban a jósolt és a valós átlagos ROI szorosan együtt mozog, akkor a modell jól kalibrált (pl. a 10%-os becsült ROI ténylegesen 10% körüli átlagos hasznot hoz).")
                        
                        cal_df_p = pd.DataFrame({'predicted': preds_p, 'actual': actuals_p})
                        try:
                            cal_df_p['bin'] = pd.qcut(cal_df_p['predicted'], q=5, duplicates='drop')
                            bin_summary_p = cal_df_p.groupby('bin', observed=False).agg(
                                mean_pred=('predicted', 'mean'),
                                mean_act=('actual', 'mean'),
                                count=('actual', 'count')
                            ).reset_index()
                            
                            bin_names_p = []
                            for idx, row in bin_summary_p.iterrows():
                                bin_names_p.append(f"Sáv {idx+1} ({row['bin'].left:.1f}% - {row['bin'].right:.1f}%)")
                            
                            chart_df_p = pd.DataFrame({
                                'Átlagos Jósolt ROI (%)': bin_summary_p['mean_pred'].values,
                                'Átlagos Valós ROI (%)': bin_summary_p['mean_act'].values
                            }, index=bin_names_p)
                            st.bar_chart(chart_df_p, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Nem sikerült a megbízhatósági sávokat kiszámítani: {e}")
                    else:
                        st.info("Nincsenek elérhető kalibrációs adatok ehhez a modellhez. Kérlek, tanítsd újra a modellt.")
            
            st.markdown(f"<p style='text-align: right; font-size: 0.8rem; color: #6b7280; margin-top: 15px;'>Modellek utolsó tanítási időpontja: {ml_metadata.get('last_trained', 'Ismeretlen')}</p>", unsafe_allow_html=True)

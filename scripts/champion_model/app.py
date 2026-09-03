import os
import sys
import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure modules path is accessible
modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "modules"))
if modules_path not in sys.path:
    sys.path.append(modules_path)

from scrape_current_season import scrape_current_season_matches
from scoreline_engine import FullScorelineEngine

# Set page config
st.set_page_config(
    page_title="NB I Bajnoki Cím & Full Scoreline Prediktor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Ensure local subdirectories are in sys.path
base_dir = os.path.dirname(__file__)
for sub in ["models", "scrapers", "data", "backtest", "simulations"]:
    p = os.path.abspath(os.path.join(base_dir, sub))
    if p not in sys.path:
        sys.path.append(p)

def find_data_file(filename):
    p_sub = os.path.join(os.path.dirname(__file__), "data", filename)
    if os.path.exists(p_sub):
        return p_sub
    return os.path.join(os.path.dirname(__file__), filename)

CSV_PATH_HISTORICAL = find_data_file("nbi_canonical_matches_2015_2025.csv")
CSV_PATH_CURRENT = find_data_file("nbi_matches_2026_current.csv")
CSV_PATH_ODDS = find_data_file("nbi_historical_odds_2015_2026.csv")

@st.cache_data
def load_all_matches():
    df_hist = pd.read_csv(CSV_PATH_HISTORICAL)
    df_hist['parsed_date'] = pd.to_datetime(df_hist['date'])
    df_hist['is_played'] = True
    
    if os.path.exists(CSV_PATH_CURRENT):
        df_curr = pd.read_csv(CSV_PATH_CURRENT)
        df_curr['parsed_date'] = pd.to_datetime(df_curr['date'])
        
        std_map = {
            "ETO FC Győr": "ETO FC",
            "ETO FC Györ": "ETO FC",
            "Videoton FC": "Fehérvár FC",
            "MOL Vidi FC": "Fehérvár FC",
            "Kispest–Honvéd FC": "Budapest Honvéd FC"
        }
        df_curr['home_team'] = df_curr['home_team'].replace(std_map)
        df_curr['away_team'] = df_curr['away_team'].replace(std_map)
        
        combined_df = pd.concat([df_hist, df_curr], ignore_index=True)
    else:
        combined_df = df_hist
        
    combined_df = combined_df.sort_values(by=['season_id', 'matchday', 'parsed_date', 'match_id']).reset_index(drop=True)
    return combined_df

def get_warmup_engine(season_id):
    """Initializes and trains FullScorelineEngine up to season_id."""
    df = load_all_matches()
    prior_matches = df[(df['season_id'] < season_id) & (df['is_played'] == True)]
    
    engine = FullScorelineEngine(
        lr_att=0.02,
        lr_def=0.02,
        home_adv=0.25,
        base_mu=0.30,
        dc_rho=-0.10
    )
    
    for r in prior_matches.itertuples():
        h, a = str(r.home_team), str(r.away_team)
        hs, ascore = int(r.home_score), int(r.away_score)
        engine.update_ratings(h, a, hs, ascore)
        
    return engine

@st.cache_data
def simulate_state_by_date(season_id, model_name, target_date_str, n_sims=1500):
    df = load_all_matches()
    s_df = df[df['season_id'] == season_id].copy()
    if s_df.empty:
        return pd.DataFrame(), 0, 0
        
    teams = sorted(list(set(s_df['home_team'].dropna().unique())))
    target_dt = pd.to_datetime(target_date_str)
    
    engine = get_warmup_engine(season_id)
    
    played_up_to_date = s_df[(s_df['parsed_date'] <= target_dt) & (s_df['is_played'] == True)].sort_values(by='parsed_date')
    rem_fixtures = s_df[(s_df['parsed_date'] > target_dt) | (s_df['is_played'] == False)].sort_values(by=['matchday', 'parsed_date'])
    
    actual_pts = {t: 0 for t in teams}
    actual_gd  = {t: 0 for t in teams}
    actual_played = {t: 0 for t in teams}
    
    for r in played_up_to_date.itertuples():
        h, a = str(r.home_team), str(r.away_team)
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            hs, ascore = int(r.home_score), int(r.away_score)
            res = str(r.result)
            actual_played[h] += 1
            actual_played[a] += 1
            if res == 'H': actual_pts[h] += 3
            elif res == 'D': actual_pts[h] += 1; actual_pts[a] += 1
            else: actual_pts[a] += 3
            actual_gd[h] += (hs - ascore)
            actual_gd[a] += (ascore - hs)
            engine.update_ratings(h, a, hs, ascore)
            
    curr_sorted = sorted(teams, key=lambda t: (actual_pts[t], actual_gd[t]), reverse=True)
    curr_rank_map = {t: r for r, t in enumerate(curr_sorted, 1)}
    
    np.random.seed(42)
    
    if len(rem_fixtures) == 0:
        records = []
        for rank, t in enumerate(curr_sorted, 1):
            records.append({
                'current_rank': rank,
                'team': t,
                'played': actual_played[t],
                'current_pts': actual_pts[t],
                'current_gd': actual_gd[t],
                'exp_pts': float(actual_pts[t]),
                'p_champion': 100.0 if rank == 1 else 0.0,
                'p_top4': 100.0 if rank <= 4 else 0.0,
                'p_relegation': 100.0 if rank >= 11 else 0.0
            })
    else:
        sim_pts = {t: np.zeros(n_sims, dtype=np.int16) for t in teams}
        sim_ranks = {t: np.zeros(n_sims, dtype=np.int8) for t in teams}
        
        rem_probs = []
        for r in rem_fixtures.itertuples():
            h, a = str(r.home_team), str(r.away_team)
            dist = engine.predict_match_full_distribution(h, a)
            rem_probs.append((h, a, dist['p_home'], dist['p_draw']))
            
        rnds = np.random.rand(n_sims, len(rem_probs))
        for s_i in range(n_sims):
            s_pts = dict(actual_pts)
            s_gd  = dict(actual_gd)
            
            for f_i, (h, a, ph, pd_) in enumerate(rem_probs):
                rnd = rnds[s_i, f_i]
                if rnd < ph:
                    s_pts[h] += 3; s_gd[h] += 1; s_gd[a] -= 1
                elif rnd < ph + pd_:
                    s_pts[h] += 1; s_pts[a] += 1
                else:
                    s_pts[a] += 3; s_gd[a] += 1; s_gd[h] -= 1
                    
            sorted_sim = sorted(teams, key=lambda t: (s_pts[t], s_gd[t]), reverse=True)
            for r_idx, t in enumerate(sorted_sim, 1):
                sim_pts[t][s_i] = s_pts[t]
                sim_ranks[t][s_i] = r_idx
                
        records = []
        for t in teams:
            records.append({
                'current_rank': curr_rank_map[t],
                'team': t,
                'played': actual_played[t],
                'current_pts': actual_pts[t],
                'current_gd': actual_gd[t],
                'exp_pts': round(float(np.mean(sim_pts[t])), 1),
                'p_champion': round(float(np.mean(sim_ranks[t] == 1) * 100), 1),
                'p_top4': round(float(np.mean(sim_ranks[t] <= 4) * 100), 1),
                'p_relegation': round(float(np.mean(sim_ranks[t] >= 11) * 100), 1)
            })
            
    df_res = pd.DataFrame(records).sort_values(by='current_rank')
    return df_res, len(played_up_to_date), len(rem_fixtures)

@st.cache_data
def get_cached_matchday_simulation(season_id, model_name):
    df = load_all_matches()
    s_df = df[df['season_id'] == season_id].copy()
    if s_df.empty:
        return pd.DataFrame()
        
    teams = sorted(list(set(s_df['home_team'].dropna().unique())))
    max_md = int(s_df['matchday'].max())
    
    engine = get_warmup_engine(season_id)
    
    actual_pts = {t: 0 for t in teams}
    actual_gd  = {t: 0 for t in teams}
    actual_played = {t: 0 for t in teams}
    
    records = []
    n_sims = 1500
    np.random.seed(42)
    
    for md in range(0, max_md + 1):
        if md > 0:
            md_matches = s_df[s_df['matchday'] == md]
            for r in md_matches.itertuples():
                h, a = str(r.home_team), str(r.away_team)
                if r.is_played and pd.notna(r.home_score) and pd.notna(r.away_score):
                    hs, ascore = int(r.home_score), int(r.away_score)
                    res = str(r.result)
                    actual_played[h] += 1
                    actual_played[a] += 1
                    if res == 'H': actual_pts[h] += 3
                    elif res == 'D': actual_pts[h] += 1; actual_pts[a] += 1
                    else: actual_pts[a] += 3
                    actual_gd[h] += (hs - ascore)
                    actual_gd[a] += (ascore - hs)
                    engine.update_ratings(h, a, hs, ascore)
                    
        curr_sorted = sorted(teams, key=lambda t: (actual_pts[t], actual_gd[t]), reverse=True)
        curr_rank_map = {t: r for r, t in enumerate(curr_sorted, 1)}
        
        rem_fixtures = s_df[(s_df['matchday'] > md) | ((s_df['matchday'] == md) & (~s_df['is_played']))]
        
        if len(rem_fixtures) == 0:
            for rank, t in enumerate(curr_sorted, 1):
                records.append({
                    'matchday': md,
                    'team': t,
                    'current_rank': rank,
                    'played': actual_played[t],
                    'current_pts': actual_pts[t],
                    'current_gd': actual_gd[t],
                    'exp_pts': float(actual_pts[t]),
                    'p_champion': 100.0 if rank == 1 else 0.0,
                    'p_top4': 100.0 if rank <= 4 else 0.0,
                    'p_relegation': 100.0 if rank >= 11 else 0.0
                })
        else:
            sim_pts = {t: np.zeros(n_sims, dtype=np.int16) for t in teams}
            sim_ranks = {t: np.zeros(n_sims, dtype=np.int8) for t in teams}
            
            rem_probs = []
            for r in rem_fixtures.itertuples():
                h, a = str(r.home_team), str(r.away_team)
                dist = engine.predict_match_full_distribution(h, a)
                rem_probs.append((h, a, dist['p_home'], dist['p_draw']))
                
            rnds = np.random.rand(n_sims, len(rem_probs))
            for s_i in range(n_sims):
                s_pts = dict(actual_pts)
                s_gd  = dict(actual_gd)
                
                for f_i, (h, a, ph, pd_) in enumerate(rem_probs):
                    rnd = rnds[s_i, f_i]
                    if rnd < ph:
                        s_pts[h] += 3; s_gd[h] += 1; s_gd[a] -= 1
                    elif rnd < ph + pd_:
                        s_pts[h] += 1; s_pts[a] += 1
                    else:
                        s_pts[a] += 3; s_gd[a] += 1; s_gd[h] -= 1
                        
                sorted_sim = sorted(teams, key=lambda t: (s_pts[t], s_gd[t]), reverse=True)
                for r_idx, t in enumerate(sorted_sim, 1):
                    sim_pts[t][s_i] = s_pts[t]
                    sim_ranks[t][s_i] = r_idx
                    
            for t in teams:
                records.append({
                    'matchday': md,
                    'team': t,
                    'current_rank': curr_rank_map[t],
                    'played': actual_played[t],
                    'current_pts': actual_pts[t],
                    'current_gd': actual_gd[t],
                    'exp_pts': round(float(np.mean(sim_pts[t])), 1),
                    'p_champion': round(float(np.mean(sim_ranks[t] == 1) * 100), 1),
                    'p_top4': round(float(np.mean(sim_ranks[t] <= 4) * 100), 1),
                    'p_relegation': round(float(np.mean(sim_ranks[t] >= 11) * 100), 1)
                })
                
    return pd.DataFrame(records)



# -------------------------------------------------------------
# MODE SELECTOR IN SIDEBAR
# -------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Nemzeti_Bajnoks%C3%A1g_I_logo.svg/800px-Nemzeti_Bajnoks%C3%A1g_I_logo.svg.png", width=110)
st.sidebar.title("⚽ NB I Analytics")

app_mode = st.sidebar.radio(
    "🧭 Alkalmazás Üzemmód",
    [
        "🏆 Champion Model & Tabella Szimuláció",
        "💰 Betting Modell & Odds Értékelő (ÚJ)"
    ],
    index=0
)

# -------------------------------------------------------------
# BETTING DATA PIPELINE
# -------------------------------------------------------------
@st.cache_data
def get_betting_evaluation_dataset():
    if not os.path.exists(CSV_PATH_ODDS):
        return pd.DataFrame()
        
    df_odds = pd.read_csv(CSV_PATH_ODDS)
    df_odds['parsed_date'] = pd.to_datetime(df_odds['date'])
    df_odds = df_odds.sort_values(by=['parsed_date', 'season']).reset_index(drop=True)
    
    engine = FullScorelineEngine(lr_att=0.02, lr_def=0.02, home_adv=0.25, base_mu=0.30, dc_rho=-0.10)
    
    records = []
    for r in df_odds.itertuples():
        h = str(r.home_team)
        a = str(r.away_team)
        preds = engine.predict_match_full_distribution(h, a)
        
        ph = preds['p_home']
        pd_ = preds['p_draw']
        pa = preds['p_away']
        
        o1 = float(r.odds_1) if pd.notna(r.odds_1) and r.odds_1 > 1.0 else 1.0
        ox = float(r.odds_x) if pd.notna(r.odds_x) and r.odds_x > 1.0 else 1.0
        o2 = float(r.odds_2) if pd.notna(r.odds_2) and r.odds_2 > 1.0 else 1.0
        
        margin = (1.0/o1 + 1.0/ox + 1.0/o2 - 1.0) * 100.0
        
        edge_1 = (ph * o1 - 1.0) * 100.0
        edge_x = (pd_ * ox - 1.0) * 100.0
        edge_2 = (pa * o2 - 1.0) * 100.0
        
        candidates = [
            ('Hazai (1)', edge_1, ph, o1, 'H'),
            ('Döntetlen (X)', edge_x, pd_, ox, 'D'),
            ('Vendég (2)', edge_2, pa, o2, 'A')
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_market, best_edge, best_p, best_odds, target_res = candidates[0]
        
        res = str(r.result) if pd.notna(r.result) else None
        hit = None
        pnl = None
        if res in ['H', 'D', 'A']:
            hit = (res == target_res)
            pnl = (best_odds - 1.0) if hit else -1.0
            
        records.append({
            'season': str(r.season),
            'date': str(r.date) if pd.notna(r.date) else '',
            'parsed_date': r.parsed_date,
            'match': f"{h} - {a}",
            'home_team': h,
            'away_team': a,
            'home_score': r.home_score,
            'away_score': r.away_score,
            'result': res,
            'odds_1': o1,
            'odds_x': ox,
            'odds_2': o2,
            'margin': margin,
            'p_fair_1': (1.0/o1) / (1.0 + margin/100.0) * 100.0,
            'p_fair_x': (1.0/ox) / (1.0 + margin/100.0) * 100.0,
            'p_fair_2': (1.0/o2) / (1.0 + margin/100.0) * 100.0,
            'model_p_h': ph * 100.0,
            'model_p_d': pd_ * 100.0,
            'model_p_a': pa * 100.0,
            'model_fair_1': round(1.0 / ph, 2) if ph > 0.01 else 99.0,
            'model_fair_x': round(1.0 / pd_, 2) if pd_ > 0.01 else 99.0,
            'model_fair_2': round(1.0 / pa, 2) if pa > 0.01 else 99.0,
            'edge_1': edge_1,
            'edge_x': edge_x,
            'edge_2': edge_2,
            'best_market': best_market,
            'best_edge': best_edge,
            'best_p': best_p * 100.0,
            'best_odds': best_odds,
            'target_res': target_res,
            'hit': hit,
            'pnl': pnl
        })
        
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            engine.update_ratings(h, a, int(r.home_score), int(r.away_score))
            
    return pd.DataFrame(records)

if app_mode == "💰 Betting Modell & Odds Értékelő (ÚJ)":
    st.markdown('<div class="main-header">💰 NB I Betting Modell & Odds Értékelő</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fogadóirodai piaci záró odds-ok (Oddsportal 1X2, 2015–2026, N=2 170 meccs) és a Dixon-Coles generatív modell szisztematikus értékelése</div>', unsafe_allow_html=True)
    
    df_betting = get_betting_evaluation_dataset()
    
    if df_betting.empty:
        st.warning("⚠️ Az odds adatbázis még nem áll rendelkezésre. Futtasd a scrapers/scrape_oddsportal_nb1.py szkriptet!")
    else:
        # Sidebar filters for Betting Mode
        st.sidebar.markdown("### 🔍 Fogadási Szűrők")
        all_betting_seasons = ["Összes Szezon (2015–2026)"] + sorted(list(df_betting['season'].unique()), reverse=True)
        selected_bet_season = st.sidebar.selectbox("📅 Szezon Szűrése", all_betting_seasons, index=0)
        
        min_edge = st.sidebar.slider("💎 Minimális Várható Érték (Edge %)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
        
        selected_market_filter = st.sidebar.selectbox(
            "🎯 Piac Szűrése",
            ["Minden Piac (1, X, 2)", "Csak Hazai (1)", "Csak Döntetlen (X)", "Csak Vendég (2)"],
            index=0
        )
        
        # Filter dataframe
        df_filtered = df_betting.copy()
        if selected_bet_season != "Összes Szezon (2015–2026)":
            df_filtered = df_filtered[df_filtered['season'] == selected_bet_season]
            
        if selected_market_filter == "Csak Hazai (1)":
            df_filtered = df_filtered[df_filtered['best_market'] == 'Hazai (1)']
        elif selected_market_filter == "Csak Döntetlen (X)":
            df_filtered = df_filtered[df_filtered['best_market'] == 'Döntetlen (X)']
        elif selected_market_filter == "Csak Vendég (2)":
            df_filtered = df_filtered[df_filtered['best_market'] == 'Vendég (2)']
            
        value_bets = df_filtered[df_filtered['best_edge'] >= min_edge].copy()
        
        # Calculate key performance indicators
        n_total = len(df_filtered)
        n_bets = len(value_bets)
        avg_margin = df_filtered['margin'].mean()
        
        played_bets = value_bets[value_bets['pnl'].notna()].copy()
        n_played = len(played_bets)
        
        if n_played > 0:
            hits = played_bets['hit'].sum()
            hit_rate = (hits / n_played) * 100.0
            total_profit = played_bets['pnl'].sum()
            roi = (total_profit / n_played) * 100.0
        else:
            hit_rate = 0.0
            total_profit = 0.0
            roi = 0.0
            
        # Top KPI cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #94A3B8;">⚽ Elemzett Meccsek</div>
                <div style="font-size: 1.7rem; font-weight: 800;">{n_total:,}</div>
                <div style="font-size: 0.8rem; color: #CBD5E1;">NB I mérkőzés</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #94A3B8;">🏢 Piaci Margin</div>
                <div style="font-size: 1.7rem; font-weight: 800;">{avg_margin:.2f}%</div>
                <div style="font-size: 0.8rem; color: #CBD5E1;">Átlagos Overround</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #94A3B8;">💎 Value Fogadások</div>
                <div style="font-size: 1.7rem; font-weight: 800;">{n_bets:,}</div>
                <div style="font-size: 0.8rem; color: #CBD5E1;">Edge ≥ {min_edge:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #94A3B8;">🎯 Találati Arány</div>
                <div style="font-size: 1.7rem; font-weight: 800;">{hit_rate:.1f}%</div>
                <div style="font-size: 0.8rem; color: #CBD5E1;">{hits if n_played>0 else 0} / {n_played} nyertes</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            pnl_color = "#10B981" if total_profit >= 0 else "#EF4444"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #94A3B8;">📈 Backtest ROI / Profit</div>
                <div style="font-size: 1.7rem; font-weight: 800; color: {pnl_color};">{roi:+.1f}%</div>
                <div style="font-size: 0.8rem; color: #CBD5E1;">Profit: {total_profit:+.1f} egység</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # Tabs for Betting View
        b_tab1, b_tab2, b_tab3, b_tab4 = st.tabs([
            "🎯 Value Bet Kereső & Részletes Oddsok",
            "📈 Kumulatív PnL & Tőke Görbe",
            "⚖️ Fogadóirodai Precízió & Margin Analízis",
            "📑 Betting Modell Módszertan & Elmélet"
        ])
        
        with b_tab1:
            st.markdown("### 🎯 Pozitív Várható Értékű (Value Bet) Lehetőségek")
            st.caption(f"Azon mérkőzések és kimenetelek, ahol a Dixon-Coles generatív modell becsült valószínűsége meghaladta a fogadóirodai záró odds által implikált valószínűséget (Min. Edge: +{min_edge}%):")
            
            if value_bets.empty:
                st.info("A megadott szűrési feltételekkel nem található Value Bet lehetőség.")
            else:
                display_cols = [
                    'season', 'date', 'match', 'result',
                    'best_market', 'best_odds', 'best_p',
                    'best_edge', 'hit', 'pnl'
                ]
                show_df = value_bets[display_cols].copy()
                show_df.columns = [
                    'Szezon', 'Dátum', 'Mérkőzés', 'Eredmény',
                    'Ajánlott Piac', 'Záró Odds', 'Modell P (%)',
                    'Várható Érték (Edge %)', 'Nyert?', 'Profit (Egység)'
                ]
                
                def style_pnl(val):
                    if pd.isna(val): return ''
                    color = '#10B981' if val > 0 else '#EF4444'
                    return f'color: {color}; font-weight: 600;'
                    
                def style_edge(val):
                    if pd.isna(val): return ''
                    return 'font-weight: bold; color: #F59E0B;' if val >= 5.0 else 'font-weight: 600;'
                    
                st.dataframe(
                    show_df.style.format({
                        'Záró Odds': '{:.2f}',
                        'Modell P (%)': '{:.1f}%',
                        'Várható Érték (Edge %)': '+{:.1f}%',
                        'Profit (Egység)': '{:+.2f}'
                    }).map(style_pnl, subset=['Profit (Egység)']).map(style_edge, subset=['Várható Érték (Edge %)']),
                    use_container_width=True,
                    height=450
                )
                
        with b_tab2:
            st.markdown("### 📈 Kumulatív Profit & Tőke Alakulása Idősorosan")
            st.caption("Flat-stake (1 egység tét fogadásonként) szimulált egyenleg alakulás a pozitív Edge-es tippeken:")
            
            if not played_bets.empty:
                played_bets['cum_pnl'] = played_bets['pnl'].cumsum()
                played_bets['bet_idx'] = range(1, len(played_bets) + 1)
                
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=played_bets['bet_idx'],
                    y=played_bets['cum_pnl'],
                    mode='lines',
                    line=dict(color='#10B981' if total_profit >= 0 else '#EF4444', width=2.5),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)' if total_profit >= 0 else 'rgba(239, 68, 68, 0.1)',
                    name='Kumulatív Profit (Flat Stake)',
                    hovertemplate="Fogadás sorszám: %{x}<br>Kumulatív Profit: %{y:+.2f} egység<extra></extra>"
                ))
                fig_pnl.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
                fig_pnl.update_layout(
                    title=f"Kumulatív PnL Teljesítmény (Összesen {len(played_bets)} lezárt fogadás, ROI: {roi:+.2f}%)",
                    xaxis_title="Fogadások Száma",
                    yaxis_title="Kumulatív Profit (Tétegység)",
                    template="plotly_white",
                    height=450
                )
                st.plotly_chart(fig_pnl, use_container_width=True)
                
                # Drawdown calculation
                cum_max = played_bets['cum_pnl'].cummax()
                drawdown = played_bets['cum_pnl'] - cum_max
                max_dd = drawdown.min()
                
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Legnagyobb Profit Csúcs", f"{played_bets['cum_pnl'].max():+.2f} egység")
                c_b.metric("Legnagyobb Visszaesés (Max Drawdown)", f"{max_dd:.2f} egység")
                c_c.metric("Végleges Egyenleg", f"{total_profit:+.2f} egység")
            else:
                st.info("Nincs elegendő lezárt mérkőzés a PnL grafikon megrajzolásához.")
                
        with b_tab3:
            st.markdown("### ⚖️ Fogadóirodai Precízió & Margin (Overround) Analízis")
            st.caption("A fogadóirodák beépített haszonkulcsa és árazási viselkedése az NB I-ben:")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                fig_margin = px.histogram(
                    df_filtered,
                    x='margin',
                    nbins=30,
                    title="Fogadóirodai Margin (Overround %) Eloszlása",
                    labels={'margin': 'Margin (%)', 'count': 'Meccsek Száma'},
                    color_discrete_sequence=['#3B82F6']
                )
                fig_margin.add_vline(x=avg_margin, line_dash="dash", line_color="#EF4444", annotation_text=f"Átlag: {avg_margin:.2f}%")
                fig_margin.update_layout(template="plotly_white")
                st.plotly_chart(fig_margin, use_container_width=True)
                
            with col_m2:
                # Margin by season trend
                margin_by_season = df_betting.groupby('season')['margin'].mean().reset_index()
                fig_season_margin = px.bar(
                    margin_by_season,
                    x='season',
                    y='margin',
                    title="Átlagos Fogadóirodai Margin Szezononként",
                    labels={'season': 'Szezon', 'margin': 'Átlagos Margin (%)'},
                    color='margin',
                    color_continuous_scale='Blues'
                )
                fig_season_margin.update_layout(template="plotly_white")
                st.plotly_chart(fig_season_margin, use_container_width=True)
                
            st.markdown("#### 🎯 Favourite-Longshot Bias Vizsgálat")
            st.info("""
            A sportfogadási piacok egyik legismertebb anomáliája a **Favourite-Longshot Bias**: a fogadók hajlamosak túlbecsülni a nagy oddsú kimeneteleket (outsiderek), 
            míg a favoritok odds-ai relatíve hatékonyabbak vagy alulértékeltek. 
            A Dixon-Coles generatív modellünk az NB I-ben képes azonosítani azokat a mérkőzéseket, ahol a döntetlen oddsok vagy az alulértékelt hazai favoritok jelentős pozitív matematikai előnyt kínálnak.
            """)
            
        with b_tab4:
            st.markdown("### 📑 Betting Modell Módszertan & Matematika")
            
            st.markdown(r"""
            #### 1. 💎 A Várható Érték (Expected Value / Edge) Meghatározása
            A sportfogadásban a hosszú távú profitabilitás egyetlen matematikai feltétele a pozitív várható érték ($EV > 0$):
            $$\text{EV} = P_{\text{modell}} \times \text{Odds} - 1$$
            Ha $P_{\text{modell}} \times \text{Odds} > 1.0$, a fogadás **Value Bet**-nek minősül.
            Például, ha a modellünk egy hazai győzelemre $P = 55\%$-ot ad, míg a fogadóiroda záró oddsa $2.10$, akkor:
            $$\text{EV} = 0.55 \times 2.10 - 1 = 1.155 - 1 = +15.5\% \text{ (Edge)}$$
            
            ---
            
            #### 2. 🏢 A Fogadóirodai Margin (Overround) és Eltávolítása (De-vigging)
            A fogadóirodák úgy biztosítják garantált hasznukat, hogy az odds-aik reciprokösszege meghaladja a 100%-ot:
            $$\sum_{k \in \{1, X, 2\}} \frac{1}{\text{Odds}_k} = 1 + \text{Margin}$$
            Az NB I-ben az átlagos margin **~5–8%** között mozog. A piac valódi (implicit fair) valószínűségeit arányos de-vigginggel számítjuk ki:
            $$P^*_i = \frac{\frac{1}{\text{Odds}_i}}{1 + \text{Margin}}$$
            
            ---
            
            #### 3. 🎯 Closing Line Value (CLV - Záró Odds) Jelentősége
            A sportfogadási szakirodalom egyértelmű bizonyítéka, hogy a mérkőzés kezdetekor érvényes **záró odds (Closing Odds)** tartalmazza a piac által elérhető összes információt 
            (kezdőcsapatok, sérültek, időjárás, profi szindikátusok tétjei). 
            Ha egy matematikai modell hosszú távon képes verni a záró oddsokat, az bizonyítja a modell statisztikai fölényét és prediktív erejét.
            
            ---
            
            #### 4. ⚖️ Kockázatkezelés & Tétméretezés (Kelly-kritérium)
            A tőkenövekedés elméleti optimumát a **Kelly-formula** adja meg:
            $$f^* = \frac{p \cdot (b - 1) - (1 - p)}{b - 1} = \frac{\text{Edge}}{b - 1}$$
            Ahol $b$ a decimális odds, $p$ a valós valószínűség. A gyakorlatban a piaci variancia tompítására **Fractional Kelly**-t (pl. $0.25 \times f^*$) vagy **Flat Stake**-et (fix 1 egység) alkalmazunk.
            """)

else:
    # -------------------------------------------------------------
    # CHAMPION MODEL DASHBOARD (ORIGINAL APP)
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # SIDEBAR CONTROLS & LIVE SCRAPER STATUS
    # -------------------------------------------------------------
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Nemzeti_Bajnoks%C3%A1g_I_logo.svg/800px-Nemzeti_Bajnoks%C3%A1g_I_logo.svg.png", width=110)
    st.sidebar.title("⚙️ Beállítások")

    selected_model = st.sidebar.selectbox(
        "🤖 Predikciós Modell",
        [
            "Model 2: Dynamic Dixon-Coles (V2/V3 Full Scoreline)",
            "Model 1: Dynamic Elo (V1 Baseline)",
            "Model 0: Static Poisson (Baseline)"
        ],
        index=0
    )

    all_seasons_list = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015]
    season_labels = {
        2026: "2026/27 (Jelenlegi szezon 🔴 LIVE)",
        2025: "2025/26 Szezon",
        2024: "2024/25 Szezon",
        2023: "2023/24 Szezon",
        2022: "2022/23 Szezon",
        2021: "2021/22 Szezon",
        2020: "2020/21 Szezon",
        2019: "2019/20 Szezon",
        2018: "2018/19 Szezon",
        2017: "2017/18 Szezon",
        2016: "2016/17 Szezon",
        2015: "2015/16 Szezon"
    }

    selected_season = st.sidebar.selectbox(
        "📅 Szezon Kiválasztása",
        all_seasons_list,
        format_func=lambda x: season_labels[x],
        index=0
    )

    st.sidebar.divider()

    # Live Data Status & Scraper Section
    st.sidebar.markdown("### 🔄 Élő Adatállapot & Frissítés")

    if os.path.exists(CSV_PATH_CURRENT):
        curr_df = pd.read_csv(CSV_PATH_CURRENT)
        curr_df['parsed_date'] = pd.to_datetime(curr_df['date'])
        played_matches = curr_df[curr_df['is_played'] == True].sort_values(by='parsed_date')
        upcoming_matches = curr_df[curr_df['is_played'] == False].sort_values(by='parsed_date')

        if not played_matches.empty:
            last_m = played_matches.iloc[-1]
            st.sidebar.caption(f"🕒 **Legutóbbi meccs:** {last_m['date']} {last_m['time'] if pd.notna(last_m['time']) else ''}")
            st.sidebar.markdown(f"*{last_m['home_team']} {int(last_m['home_score'])} - {int(last_m['away_score'])} {last_m['away_team']}*")

        if not upcoming_matches.empty:
            next_m = upcoming_matches.iloc[0]
            st.sidebar.caption(f"📅 **Következő meccs:** {next_m['date']} {next_m['time'] if pd.notna(next_m['time']) else ''}")
            st.sidebar.markdown(f"*{next_m['home_team']} vs {next_m['away_team']}*")

        now = datetime.now()
        overdue = upcoming_matches[upcoming_matches['parsed_date'] < now]
        if not overdue.empty:
            st.sidebar.warning(f"⚠️ **Frissítés javasolt!** {len(overdue)} mérkőzés időpontja már elmúlt.")
        else:
            st.sidebar.success("✅ **Az adatbázis naprakész.**")
    else:
        st.sidebar.warning("⚠️ Még nincsenek letöltve a 2026/27-es adatok.")

    if st.sidebar.button("🚀 Adatok Frissítése (Transfermarkt Scraper)", use_container_width=True):
        with st.sidebar:
            with st.spinner("⏳ Transfermarkt adatok letöltése Playwright segítségével..."):
                scraped_df = scrape_current_season_matches(season_id=2026, headless=True)
                if not scraped_df.empty:
                    scraped_df.to_csv(CSV_PATH_CURRENT, index=False, encoding='utf-8-sig')
                    st.cache_data.clear()
                    st.success("✅ Sikeres frissítés!")
                    st.rerun()
                else:
                    st.error("[-] Nem sikerült adatot kinyerni a Transfermarktról.")

    # -------------------------------------------------------------
    # MAIN APP HEADER
    # -------------------------------------------------------------
    st.markdown("<div class='main-header'>⚽ NB I Bajnoki Cím & Full Scoreline Prediktor</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>Kiválasztott Szezon: <b>{season_labels[selected_season]}</b> | Modell: <b>{selected_model}</b></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tabella & Bajnoki Esélyek (Független Dátum / Forduló)",
        "🎯 Meccs & Piac Predikciók (Gólok, Over/Under, BTTS)",
        "📈 Szezonális Valószínűségi Trendek (Lineplot)",
        "📑 Modell Dokumentáció & Benchmarkok"
    ])

    all_matches_df = load_all_matches()
    season_matches = all_matches_df[all_matches_df['season_id'] == selected_season].copy()

    if season_matches.empty:
        st.warning("⚠️ Nincs elérhető mérkőzésadat ehhez a szezonhoz. Kattints az 'Adatok Frissítése' gombra az oldalsávban!")
        st.stop()

    # -------------------------------------------------------------
    # TAB 1: TABELLA & ESÉLYEK
    # -------------------------------------------------------------
    with tab1:
        col_mode, col_selector = st.columns([1, 2.5])

        with col_mode:
            filter_mode = st.radio(
                "🔘 Szűrési Mód Kiválasztása",
                ["📅 Naptári Dátum szerint (Valós időpillanat)", "🔢 Forduló szerint (Fix játéknap)"],
                help="A dátum szerinti szűrés pontosan a megadott napon ténylegesen lejátszott meccseket veszi figyelembe."
            )

        valid_dates_series = season_matches['parsed_date'].dropna().sort_values()
        min_date = valid_dates_series.min() - pd.Timedelta(days=1)
        unique_dates = sorted(list(set(valid_dates_series.dt.strftime('%Y-%m-%d').tolist())))
        unique_dates = [(min_date.strftime('%Y-%m-%d'))] + unique_dates
        max_matchday_num = int(season_matches['matchday'].max())

        with col_selector:
            if "Naptári Dátum" in filter_mode:
                default_date_idx = len(unique_dates) - 1
                if selected_season == 2026:
                    played_dates = season_matches[season_matches['is_played']]['parsed_date'].dropna().sort_values()
                    if not played_dates.empty:
                        last_played_str = played_dates.max().strftime('%Y-%m-%d')
                        if last_played_str in unique_dates:
                            default_date_idx = unique_dates.index(last_played_str)

                selected_date_str = st.select_slider(
                    "📅 Válassz Naptári Dátumot:",
                    options=unique_dates,
                    value=unique_dates[default_date_idx]
                )
                df_state, n_played_matches, n_rem_fixtures = simulate_state_by_date(selected_season, selected_model, selected_date_str)
                st.markdown(f"#### 📌 Állás a **{selected_date_str}** naptári napon (Eddig lejátszva: **{n_played_matches} meccs**, Hátralévő: **{n_rem_fixtures} meccs**)")

            else:
                played_mds = season_matches[season_matches['is_played']]['matchday'].dropna().unique()
                default_md = int(max(played_mds)) if len(played_mds) > 0 and selected_season == 2026 else max_matchday_num

                selected_md_num = st.slider(
                    "🔢 Válassz Fordulószámot (0 = Rajt előtt):",
                    0, max_matchday_num, value=default_md
                )
                df_all_mds = get_cached_matchday_simulation(selected_season, selected_model)
                df_state = df_all_mds[df_all_mds['matchday'] == selected_md_num].sort_values(by='current_rank').copy()
                st.markdown(f"#### 📌 Állás a(z) **{selected_md_num}. Forduló** után")

        if not df_state.empty:
            top_champ = df_state.sort_values(by='p_champion', ascending=False).iloc[0]
            top_releg = df_state.sort_values(by='p_relegation', ascending=False).iloc[0]
            curr_leader = df_state.iloc[0]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("🏆 Bajnoki Favorit", f"{top_champ['team']}", f"{top_champ['p_champion']:.1f}% esély")
            with c2:
                st.metric("🥇 Tabella Első Helyezett", f"{curr_leader['team']}", f"{curr_leader['current_pts']} pont ({curr_leader['played']} meccs)")
            with c3:
                st.metric("🔴 Legnagyobb Kiesési Kockázat", f"{top_releg['team']}", f"{top_releg['p_relegation']:.1f}% kiesési esély")

            st.write("")

            display_df = df_state[[
                'current_rank', 'team', 'played', 'current_pts', 'current_gd',
                'exp_pts', 'p_champion', 'p_top4', 'p_relegation'
            ]].copy()

            display_df.columns = [
                'Hely', 'Csapat', 'Meccs', 'Pont', 'Gólkülönbség',
                'Várható Végső Pont', 'Bajnok %', 'Top 4 %', 'Kiesés %'
            ]

            def highlight_rows(row):
                rank = row['Hely']
                if rank == 1:
                    return ['background-color: rgba(34, 197, 94, 0.15); font-weight: bold'] * len(row)
                elif rank <= 4:
                    return ['background-color: rgba(59, 130, 246, 0.08)'] * len(row)
                elif rank >= 11:
                    return ['background-color: rgba(239, 68, 68, 0.15)'] * len(row)
                return [''] * len(row)

            styled_table = display_df.style.apply(highlight_rows, axis=1).format({
                'Várható Végső Pont': '{:.1f}',
                'Bajnok %': '{:.1f}%',
                'Top 4 %': '{:.1f}%',
                'Kiesés %': '{:.1f}%'
            })

            st.dataframe(styled_table, use_container_width=True, hide_index=True)

            fig_pts = go.Figure()
            fig_pts.add_trace(go.Bar(
                x=df_state['team'],
                y=df_state['current_pts'],
                name='Megszerzett Pontok',
                marker_color='#3B82F6'
            ))
            fig_pts.add_trace(go.Bar(
                x=df_state['team'],
                y=df_state['exp_pts'],
                name='Várható Végső Pontok (Szimulált)',
                marker_color='#10B981'
            ))
            fig_pts.update_layout(
                title="📊 Megszerzett Pontok vs. Várható Végső Pontszám",
                barmode='group',
                xaxis_title="Csapat",
                yaxis_title="Pontszám",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_pts, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: V3 FULL SCORELINE & PIAC PREDIKCIÓK
    # -------------------------------------------------------------
    with tab2:
        st.markdown("### 🎯 V3 Full Scoreline & Piac Valószínűségi Motor")
        st.markdown("Válassz ki egy konkrét mérkőzést a szezonból, vagy állíts be egyedi párosítást a **teljes $6 \\times 6$-os gólmátrix**, **Over/Under** és **BTTS** piacok kiszámításához:")

        engine_v3 = get_warmup_engine(selected_season)
        season_teams = sorted(list(set(season_matches['home_team'].dropna().unique())))

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            home_pick = st.selectbox("🏠 Hazai Csapat", season_teams, index=0)
        with col_m2:
            away_pick = st.selectbox("🚶 Vendég Csapat", season_teams, index=min(1, len(season_teams)-1))

        if home_pick == away_pick:
            st.warning("⚠️ Kérlek válassz két különböző csapatot a párosításhoz!")
        else:
            dist = engine_v3.predict_match_full_distribution(home_pick, away_pick)

            # 1. Big KPI Highlights
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("⚽ Várható Hazai Gól (λH)", f"{dist['lambda_home']:.2f}")
            with k2:
                st.metric("⚽ Várható Vendég Gól (λA)", f"{dist['lambda_away']:.2f}")
            with k3:
                st.metric("🎯 Legvalószínűbb Eredmény", f"{dist['most_likely_score']}", f"{dist['most_likely_score_prob']*100:.1f}% esély")
            with k4:
                st.metric("🔥 Várható Összes Gól", f"{dist['expected_total_goals']:.2f}")

            st.divider()

            c_left, c_right = st.columns([1.2, 1])

            with c_left:
                st.markdown("#### 📊 1. 2D Scoreline Valószínűségi Mátrix (Heatmap)")
                matrix_df = pd.DataFrame(
                    dist['matrix'][:6, :6] * 100,
                    index=[f"Hazai {x}" for x in range(6)],
                    columns=[f"Vendég {y}" for y in range(6)]
                )
                fig_heat = px.imshow(
                    matrix_df,
                    text_auto='.1f',
                    color_continuous_scale='Blues',
                    labels=dict(x="Vendég Gólok", y="Hazai Gólok", color="Valószínűség %")
                )
                fig_heat.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_heat, use_container_width=True)

            with c_right:
                st.markdown("#### 🥇 2. Top 5 Legvalószínűbb Pontos Eredmény")
                top_scores_df = pd.DataFrame(dist['top_scores'], columns=['Pontos Eredmény', 'Valószínűség'])
                top_scores_df['Valószínűség %'] = top_scores_df['Valószínűség'] * 100

                fig_top = px.bar(
                    top_scores_df,
                    x='Pontos Eredmény',
                    y='Valószínűség %',
                    text=top_scores_df['Valószínűség %'].apply(lambda x: f"{x:.1f}%"),
                    color='Valószínűség %',
                    color_continuous_scale='Greens'
                )
                fig_top.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_top, use_container_width=True)

            # 3. Market Summary Cards
            st.markdown("#### 💰 3. Fogadási Piacok Valószínűségei")
            p1, p2, p3, p4 = st.columns(4)

            with p1:
                st.markdown("##### 🏆 1X2 Piac")
                st.write(f"• **1 (Hazai):** `{dist['p_home']*100:.1f}%` *(fair odds: {1/dist['p_home']:.2f})*")
                st.write(f"• **X (Döntetlen):** `{dist['p_draw']*100:.1f}%` *(fair odds: {1/dist['p_draw']:.2f})*")
                st.write(f"• **2 (Vendég):** `{dist['p_away']*100:.1f}%` *(fair odds: {1/dist['p_away']:.2f})*")

            with p2:
                st.markdown("##### 📈 Over / Under 2.5")
                st.write(f"• **Over 2.5:** `{dist['p_over_2_5']*100:.1f}%` *(fair: {1/dist['p_over_2_5']:.2f})*")
                st.write(f"• **Under 2.5:** `{dist['p_under_2_5']*100:.1f}%` *(fair: {1/dist['p_under_2_5']:.2f})*")

            with p3:
                st.markdown("##### 🤝 Both Teams To Score")
                st.write(f"• **BTTS Igen:** `{dist['p_btts_yes']*100:.1f}%` *(fair: {1/dist['p_btts_yes']:.2f})*")
                st.write(f"• **BTTS Nem:** `{dist['p_btts_no']*100:.1f}%` *(fair: {1/dist['p_btts_no']:.2f})*")

            with p4:
                st.markdown("##### 📊 Egyéb Gólszám Piacok")
                st.write(f"• **Over 1.5:** `{dist['p_over_1_5']*100:.1f}%`")
                st.write(f"• **Over 3.5:** `{dist['p_over_3_5']*100:.1f}%`")
                st.write(f"• **0-0 esély:** `{dist['p_0_0']*100:.1f}%`")

    # -------------------------------------------------------------
    # TAB 3: LINEPLOT TRENDEK
    # -------------------------------------------------------------
    with tab3:
        st.markdown(f"### 📈 Szezonális Valószínűségi Trendek – **{season_labels[selected_season]}**")

        df_line_sim = get_cached_matchday_simulation(selected_season, selected_model)

        played_mds_in_s = season_matches[season_matches['is_played']]['matchday'].dropna().unique()
        max_played_md = int(max(played_mds_in_s)) if len(played_mds_in_s) > 0 else 0
        total_season_mds = int(season_matches['matchday'].max())

        is_in_progress = (max_played_md < total_season_mds)
        if is_in_progress:
            df_line_sim = df_line_sim[df_line_sim['matchday'] <= max_played_md].copy()
            st.info(f"ℹ️ **Folyamatban lévő szezon:** A grafikon a már lejátszott **0–{max_played_md}. fordulók** közötti esélyalakulást mutatja.")

        col_opt1, col_opt2 = st.columns([1, 2])
        with col_opt1:
            target_metric = st.selectbox(
                "🎯 Vizsgálandó Valószínűségi Kategória",
                ["🏆 Bajnok (Bajnoki cím %)", "🥉 Top 4 (Nemzetközi kupa %)", "🔴 Kieső (Kiesési zóna %)"]
            )

        metric_col_map = {
            "🏆 Bajnok (Bajnoki cím %)": ('p_champion', 'Bajnoki Cím Valószínűség (%)', '#F59E0B'),
            "🥉 Top 4 (Nemzetközi kupa %)": ('p_top4', 'Top 4 Valószínűség (%)', '#3B82F6'),
            "🔴 Kieső (Kiesési zóna %)": ('p_relegation', 'Kiesési Valószínűség (%)', '#EF4444')
        }
        col_name, title_name, primary_col = metric_col_map[target_metric]

        all_teams_in_season = sorted(df_line_sim['team'].unique())
        with col_opt2:
            selected_teams = st.multiselect(
                "Csapatok Szűrése (Üresen hagyva minden csapat látható):",
                all_teams_in_season,
                default=[]
            )

        plot_df = df_line_sim.copy()
        if len(selected_teams) > 0:
            plot_df = plot_df[plot_df['team'].isin(selected_teams)]

        max_md_line = int(plot_df['matchday'].max()) if not plot_df.empty else 0

        fig_line = px.line(
            plot_df,
            x='matchday',
            y=col_name,
            color='team',
            markers=True,
            title=f"<b>{title_name} Alakulása Fordulóról Fordulóra (0–{max_md_line}. forduló)</b>",
            labels={'matchday': 'Forduló', col_name: 'Valószínűség (%)', 'team': 'Csapat'},
            hover_data={'matchday': True, col_name: ':.1f%', 'team': True}
        )

        fig_line.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1 if max_md_line <= 10 else 2,
                title="<b>Forduló</b> (0 = Szezon előtt)",
                range=[-0.3, max_md_line + 0.3]
            ),
            yaxis=dict(title=f"<b>{title_name}</b>", range=[-2, 103]),
            height=580,
            hovermode="x unified",
            legend=dict(title="Csapatok", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )

        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("#### 📋 Fordulónkénti Részletes Valószínűségi Táblázat")
        pivot_df = plot_df.pivot(index='matchday', columns='team', values=col_name)
        pivot_df.index.name = 'Forduló'
        st.dataframe(pivot_df.style.format("{:.1f}%"), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: MODELL DOKUMENTÁCIÓ & BENCHMARKOK
    # -------------------------------------------------------------
    with tab4:
        st.markdown("### 📑 Modell Dokumentáció & Történelmi Benchmark Eredmények")
        st.markdown("A rendszerben megvalósított generatív modellek mélyreható statisztikai és valószínűségi auditja a **2015–2026** közötti NB I-es meccsadatokon:")

        st.markdown("#### 1. 🏆 Full Scoreline Log Loss & 1X2 Benchmark (Érintetlen Teszthalmaz 2022–2025, N=792)")
        st.caption("Proper scoring rule a teljes 2D scoreline mátrix felett: -log P(X=Goals_H, Y=Goals_A)")

        bench_data = [
            {
                "Modell": "Model 2: Dynamic Dixon-Coles (V3 Generatív)",
                "Full Scoreline Log Loss (↓)": "3.0335",
                "1X2 Match Log Loss (↓)": "1.0366",
                "Exact Score Top-1": "11.49%",
                "Exact Score Top-3": "32.70%",
                "BTTS Log Loss (↓)": "0.6848"
            },
            {
                "Modell": "Model 2-B: Dynamic Poisson (rho = 0.0)",
                "Full Scoreline Log Loss (↓)": "3.0362",
                "1X2 Match Log Loss (↓)": "1.0377",
                "Exact Score Top-1": "12.50%",
                "Exact Score Top-3": "31.31%",
                "BTTS Log Loss (↓)": "0.6857"
            },
            {
                "Modell": "Model 1-B: Dynamic Elo-Implied Poisson",
                "Full Scoreline Log Loss (↓)": "3.0470",
                "1X2 Match Log Loss (↓)": "1.0421",
                "Exact Score Top-1": "10.48%",
                "Exact Score Top-3": "30.68%",
                "BTTS Log Loss (↓)": "0.7001"
            },
            {
                "Modell": "Model 0: Static Poisson (Baseline)",
                "Full Scoreline Log Loss (↓)": "3.0864",
                "1X2 Match Log Loss (↓)": "1.0742",
                "Exact Score Top-1": "11.62%",
                "Exact Score Top-3": "29.92%",
                "BTTS Log Loss (↓)": "0.6873"
            }
        ]
        st.table(pd.DataFrame(bench_data))

        st.markdown("#### 2. 📈 Többküszöbös Over / Under Probabilisztikus Kiértékelés (Dynamic Dixon-Coles)")
        st.caption("A modell valószínűségi pontossága a teljes gólspektrum összes küszöbén:")
        ou_bench_table = [
            {"Piac (Küszöb)": "Over / Under 0.5", "Log Loss (↓)": "0.2582", "Brier Score (↓)": "0.0653", "Accuracy": "93.06%", "Predikált Over %": "93.0%", "Valós Over %": "93.1%", "Kalibrációs Eltérés": "+0.07%"},
            {"Piac (Küszöb)": "Over / Under 1.5", "Log Loss (↓)": "0.5281", "Brier Score (↓)": "0.1717", "Accuracy": "78.03%", "Predikált Over %": "78.8%", "Valós Over %": "78.0%", "Kalibrációs Eltérés": "-0.76%"},
            {"Piac (Küszöb)": "Over / Under 2.5", "Log Loss (↓)": "0.6919", "Brier Score (↓)": "0.2492", "Accuracy": "54.29%", "Predikált Over %": "54.8%", "Valós Over %": "55.9%", "Kalibrációs Eltérés": "+1.17%"},
            {"Piac (Küszöb)": "Over / Under 3.5", "Log Loss (↓)": "0.6426", "Brier Score (↓)": "0.2243", "Accuracy": "66.16%", "Predikált Over %": "33.0%", "Valós Over %": "32.4%", "Kalibrációs Eltérés": "-0.52%"},
            {"Piac (Küszöb)": "Over / Under 4.5", "Log Loss (↓)": "0.4352", "Brier Score (↓)": "0.1317", "Accuracy": "84.85%", "Predikált Over %": "17.2%", "Valós Over %": "15.2%", "Kalibrációs Eltérés": "-2.01%"},
            {"Piac (Küszöb)": "Over / Under 5.5", "Log Loss (↓)": "0.2960", "Brier Score (↓)": "0.0773", "Accuracy": "91.79%", "Predikált Over %": "7.8%", "Valós Over %": "8.2%", "Kalibrációs Eltérés": "+0.37%"}
        ]
        st.table(pd.DataFrame(ou_bench_table))

        st.markdown("#### 3. 🎯 Scoreline Valószínűségi Kalibráció (Binned Reliability Curve)")
        st.caption("19 800 cella-előrejelzés összevetése a valós bekövetkezési frekvenciákkal:")
        rel_table = [
            {"Predikált P(Score) Sáv": "0% - 2%", "Megfigyelések Száma": 8239, "Átlagos Predikció %": "0.84%", "Valós Gyakoriság %": "0.85%", "Kalibrációs Eltérés": "+0.01%"},
            {"Predikált P(Score) Sáv": "2% - 4%", "Megfigyelések Száma": 3796, "Átlagos Predikció %": "2.90%", "Valós Gyakoriság %": "2.79%", "Kalibrációs Eltérés": "-0.11%"},
            {"Predikált P(Score) Sáv": "4% - 6%", "Megfigyelések Száma": 2815, "Átlagos Predikció %": "4.99%", "Valós Gyakoriság %": "5.04%", "Kalibrációs Eltérés": "+0.06%"},
            {"Predikált P(Score) Sáv": "6% - 8%", "Megfigyelések Száma": 2050, "Átlagos Predikció %": "6.93%", "Valós Gyakoriság %": "6.20%", "Kalibrációs Eltérés": "-0.73%"},
            {"Predikált P(Score) Sáv": "8% - 10%", "Megfigyelések Száma": 1766, "Átlagos Predikció %": "9.01%", "Valós Gyakoriság %": "10.02%", "Kalibrációs Eltérés": "+1.01%"},
            {"Predikált P(Score) Sáv": "10% - 15%", "Megfigyelések Száma": 1132, "Átlagos Predikció %": "12.02%", "Valós Gyakoriság %": "11.31%", "Kalibrációs Eltérés": "-0.71%"}
        ]
        st.table(pd.DataFrame(rel_table))

        st.markdown("#### 4. ⚽ Top 8 Leggyakoribb NB I Pontos Eredmény Kalibrációja")
        sc_indiv_table = [
            {"Pontos Eredmény": "1-1 (NB I leggyakoribb)", "Predikált Átlag %": "11.93%", "Valós Gyakoriság %": "11.62%", "Eltérés": "-0.31%"},
            {"Pontos Eredmény": "2-1", "Predikált Átlag %": "8.54%", "Valós Gyakoriság %": "9.97%", "Eltérés": "+1.44%"},
            {"Pontos Eredmény": "1-0", "Predikált Átlag %": "7.99%", "Valós Gyakoriság %": "8.33%", "Eltérés": "+0.35%"},
            {"Pontos Eredmény": "1-2", "Predikált Átlag %": "6.89%", "Valós Gyakoriság %": "7.45%", "Eltérés": "+0.56%"},
            {"Pontos Eredmény": "0-0", "Predikált Átlag %": "7.01%", "Valós Gyakoriság %": "6.94%", "Eltérés": "-0.07%"},
            {"Pontos Eredmény": "2-0", "Predikált Átlag %": "7.34%", "Valós Gyakoriság %": "5.81%", "Eltérés": "-1.53%"},
            {"Pontos Eredmény": "0-1", "Predikált Átlag %": "6.21%", "Valós Gyakoriság %": "6.69%", "Eltérés": "+0.48%"},
            {"Pontos Eredmény": "2-2", "Predikált Átlag %": "5.28%", "Valós Gyakoriság %": "5.81%", "Eltérés": "+0.53%"}
        ]
        st.table(pd.DataFrame(sc_indiv_table))

        st.markdown(r"""
        #### 5. 📐 Matematikai Modell Architektúra & Gradiens Audit

        ##### ⚽ Várható Gólok (Expected Goals):
        $$\ln \lambda_{\text{home}} = \mu + \alpha_{\text{home}} + \beta_{\text{away}} + \gamma_{\text{home}}$$
        $$\ln \mu_{\text{away}} = \mu + \alpha_{\text{away}} + \beta_{\text{home}}$$

        ##### 🎯 Dixon-Coles Alacsony Gólszámú Korrekció (\tau):
        $$\tau(0,0) = 1 - \lambda \mu \rho, \quad \tau(0,1) = 1 + \lambda \rho, \quad \tau(1,0) = 1 + \mu \rho, \quad \tau(1,1) = 1 - \rho$$

        ##### 📈 Kétoldali Gradiens Frissítés (Dual Residual Update):
        $$e_H = \text{Goals}_H - \lambda_H, \quad e_A = \text{Goals}_A - \lambda_A$$
        $$\alpha_H \leftarrow \alpha_H + \eta_{\text{att}} e_H, \quad \beta_A \leftarrow \beta_A + \eta_{\text{def}} e_H$$
        $$\alpha_A \leftarrow \alpha_A + \eta_{\text{att}} e_A, \quad \beta_H \leftarrow \beta_H + \eta_{\text{def}} e_A$$
        """)
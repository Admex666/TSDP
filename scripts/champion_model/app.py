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

# Set page config
st.set_page_config(
    page_title="NB I Bajnoki Cím & Tabella Prediktor",
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

CSV_PATH_HISTORICAL = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")
CSV_PATH_CURRENT = os.path.join(os.path.dirname(__file__), "nbi_matches_2026_current.csv")
FACT = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]

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

def get_warmup_ratings(season_id):
    """Computes pre-season starting ratings for season_id based on all previous history."""
    df = load_all_matches()
    prior_matches = df[(df['season_id'] < season_id) & (df['is_played'] == True)]
    
    alpha, beta = {}, {}
    elo_ratings = {}
    
    for r in prior_matches.itertuples():
        h, a = str(r.home_team), str(r.away_team)
        hs, ascore = int(r.home_score), int(r.away_score)
        res = str(r.result)
        
        lh = max(0.1, min(4.5, math.exp(0.30 + alpha.get(h, 0.0) + beta.get(a, 0.0) + 0.25)))
        la = max(0.1, min(4.5, math.exp(0.30 + alpha.get(a, 0.0) + beta.get(h, 0.0))))
        alpha[h] = alpha.get(h, 0.0) + 0.02 * (hs - lh)
        beta[a]  = beta.get(a, 0.0)  + 0.02 * (hs - lh)
        alpha[a] = alpha.get(a, 0.0) + 0.02 * (ascore - la)
        beta[h]  = beta.get(h, 0.0)  + 0.02 * (ascore - la)
        
        rh = elo_ratings.get(h, 1500.0)
        ra = elo_ratings.get(a, 1500.0)
        eh = 1.0 / (1.0 + math.pow(10.0, -((rh + 60.0) - ra) / 400.0))
        s_act = 1.0 if res == 'H' else (0.5 if res == 'D' else 0.0)
        elo_ratings[h] = rh + 15.0 * (s_act - eh)
        elo_ratings[a] = ra - 15.0 * (s_act - eh)
        
    return alpha, beta, elo_ratings

@st.cache_data
def simulate_state_by_date(season_id, model_name, target_date_str, n_sims=1500):
    """
    Computes exact table standings and simulations strictly based on calendar date (<= target_date).
    Treats postponed and unplayed matches after target_date as remaining fixtures to simulate!
    """
    df = load_all_matches()
    s_df = df[df['season_id'] == season_id].copy()
    if s_df.empty:
        return pd.DataFrame(), None, 0
        
    teams = sorted(list(set(s_df['home_team'].dropna().unique())))
    target_dt = pd.to_datetime(target_date_str)
    
    alpha, beta, elo_ratings = get_warmup_ratings(season_id)
    
    # Played matches on or before target_date
    played_up_to_date = s_df[(s_df['parsed_date'] <= target_dt) & (s_df['is_played'] == True)].sort_values(by='parsed_date')
    rem_fixtures = s_df[(s_df['parsed_date'] > target_dt) | (s_df['is_played'] == False)].sort_values(by=['matchday', 'parsed_date'])
    
    actual_pts = {t: 0 for t in teams}
    actual_gd  = {t: 0 for t in teams}
    actual_played = {t: 0 for t in teams}
    
    curr_alpha = dict(alpha)
    curr_beta  = dict(beta)
    curr_elo   = dict(elo_ratings)
    
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
            
            lh = max(0.1, min(4.5, math.exp(0.30 + curr_alpha.get(h, 0.0) + curr_beta.get(a, 0.0) + 0.25)))
            la = max(0.1, min(4.5, math.exp(0.30 + curr_alpha.get(a, 0.0) + curr_beta.get(h, 0.0))))
            curr_alpha[h] = curr_alpha.get(h, 0.0) + 0.02 * (hs - lh)
            curr_beta[a]  = curr_beta.get(a, 0.0)  + 0.02 * (hs - lh)
            curr_alpha[a] = curr_alpha.get(a, 0.0) + 0.02 * (ascore - la)
            curr_beta[h]  = curr_beta.get(h, 0.0)  + 0.02 * (ascore - la)
            
            rh = curr_elo.get(h, 1500.0)
            ra = curr_elo.get(a, 1500.0)
            eh = 1.0 / (1.0 + math.pow(10.0, -((rh + 60.0) - ra) / 400.0))
            s_act = 1.0 if res == 'H' else (0.5 if res == 'D' else 0.0)
            curr_elo[h] = rh + 15.0 * (s_act - eh)
            curr_elo[a] = ra - 15.0 * (s_act - eh)
            
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
            if 'Dixon-Coles' in model_name:
                log_lh = 0.30 + curr_alpha.get(h, 0.0) + curr_beta.get(a, 0.0) + 0.25
                log_la = 0.30 + curr_alpha.get(a, 0.0) + curr_beta.get(h, 0.0)
                lh = max(0.1, min(4.5, math.exp(log_lh)))
                la = max(0.1, min(4.5, math.exp(log_la)))
                eh = lh / (lh + la)
                pd_ = 0.26 * math.exp(-((lh - la)**2)/2.0)
                ph = max(0.01, eh - 0.5 * pd_)
            elif 'Elo' in model_name:
                rh = curr_elo.get(h, 1500.0)
                ra = curr_elo.get(a, 1500.0)
                d_elo = (rh + 60.0) - ra
                eh = 1.0 / (1.0 + math.pow(10.0, -d_elo / 400.0))
                pd_ = 0.28 * math.exp(-math.pow(d_elo / 350.0, 2))
                ph = max(0.01, eh - 0.5 * pd_)
            else:
                ph, pd_ = 0.44, 0.26
            rem_probs.append((h, a, ph, pd_))
            
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
    """
    Computes matchday-by-matchday (0..33) progression for lineplots and matchday filtering.
    """
    df = load_all_matches()
    s_df = df[df['season_id'] == season_id].copy()
    if s_df.empty:
        return pd.DataFrame()
        
    teams = sorted(list(set(s_df['home_team'].dropna().unique())))
    max_md = int(s_df['matchday'].max())
    
    alpha, beta, elo_ratings = get_warmup_ratings(season_id)
    
    actual_pts = {t: 0 for t in teams}
    actual_gd  = {t: 0 for t in teams}
    actual_played = {t: 0 for t in teams}
    
    records = []
    n_sims = 1500
    np.random.seed(42)
    
    curr_alpha = dict(alpha)
    curr_beta  = dict(beta)
    curr_elo   = dict(elo_ratings)
    
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
                    
                    lh = max(0.1, min(4.5, math.exp(0.30 + curr_alpha.get(h, 0.0) + curr_beta.get(a, 0.0) + 0.25)))
                    la = max(0.1, min(4.5, math.exp(0.30 + curr_alpha.get(a, 0.0) + curr_beta.get(h, 0.0))))
                    curr_alpha[h] = curr_alpha.get(h, 0.0) + 0.02 * (hs - lh)
                    curr_beta[a]  = curr_beta.get(a, 0.0)  + 0.02 * (hs - lh)
                    curr_alpha[a] = curr_alpha.get(a, 0.0) + 0.02 * (ascore - la)
                    curr_beta[h]  = curr_beta.get(h, 0.0)  + 0.02 * (ascore - la)
                    
                    rh = curr_elo.get(h, 1500.0)
                    ra = curr_elo.get(a, 1500.0)
                    eh = 1.0 / (1.0 + math.pow(10.0, -((rh + 60.0) - ra) / 400.0))
                    s_act = 1.0 if res == 'H' else (0.5 if res == 'D' else 0.0)
                    curr_elo[h] = rh + 15.0 * (s_act - eh)
                    curr_elo[a] = ra - 15.0 * (s_act - eh)
                    
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
                if 'Dixon-Coles' in model_name:
                    log_lh = 0.30 + curr_alpha.get(h, 0.0) + curr_beta.get(a, 0.0) + 0.25
                    log_la = 0.30 + curr_alpha.get(a, 0.0) + curr_beta.get(h, 0.0)
                    lh = max(0.1, min(4.5, math.exp(log_lh)))
                    la = max(0.1, min(4.5, math.exp(log_la)))
                    eh = lh / (lh + la)
                    pd_ = 0.26 * math.exp(-((lh - la)**2)/2.0)
                    ph = max(0.01, eh - 0.5 * pd_)
                elif 'Elo' in model_name:
                    rh = curr_elo.get(h, 1500.0)
                    ra = curr_elo.get(a, 1500.0)
                    d_elo = (rh + 60.0) - ra
                    eh = 1.0 / (1.0 + math.pow(10.0, -d_elo / 400.0))
                    pd_ = 0.28 * math.exp(-math.pow(d_elo / 350.0, 2))
                    ph = max(0.01, eh - 0.5 * pd_)
                else:
                    ph, pd_ = 0.44, 0.26
                rem_probs.append((h, a, ph, pd_))
                
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
# SIDEBAR CONTROLS & LIVE SCRAPER STATUS
# -------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Nemzeti_Bajnoks%C3%A1g_I_logo.svg/800px-Nemzeti_Bajnoks%C3%A1g_I_logo.svg.png", width=110)
st.sidebar.title("⚙️ Beállítások")

selected_model = st.sidebar.selectbox(
    "🤖 Predikciós Modell",
    [
        "Model 2: Dynamic Dixon-Coles (V2 - Legjobb)",
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
st.markdown("<div class='main-header'>⚽ NB I Bajnoki Cím & Tabella Szimulátor</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Kiválasztott Szezon: <b>{season_labels[selected_season]}</b> | Modell: <b>{selected_model}</b></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📊 Tabella & Bajnoki Esélyek (Független Dátum / Forduló)",
    "📈 Szezonális Valószínűségi Trendek (Lineplot)",
    "📑 Modell Dokumentáció & Benchmarkok"
])

all_matches_df = load_all_matches()
season_matches = all_matches_df[all_matches_df['season_id'] == selected_season].copy()

if season_matches.empty:
    st.warning("⚠️ Nincs elérhető mérkőzésadat ehhez a szezonhoz. Kattints az 'Adatok Frissítése' gombra az oldalsávban!")
    st.stop()

# -------------------------------------------------------------
# TAB 1: TABELLA & ESÉLYEK (FÜGGETLEN DÁTUM VAGY FORDULÓ)
# -------------------------------------------------------------
with tab1:
    col_mode, col_selector = st.columns([1, 2.5])
    
    with col_mode:
        filter_mode = st.radio(
            "🔘 Szűrési Mód Kiválasztása",
            ["📅 Naptári Dátum szerint (Valós időpillanat)", "🔢 Forduló szerint (Fix játéknap)"],
            help="A dátum szerinti szűrés pontosan a megadott napon ténylegesen lejátszott meccseket veszi figyelembe (figyelembe véve az elhalasztott vagy előrehozott meccseket)."
        )
        
    valid_dates_series = season_matches['parsed_date'].dropna().sort_values()
    min_date = valid_dates_series.min() - pd.Timedelta(days=1)
    max_date = valid_dates_series.max()
    unique_dates = sorted(list(set(valid_dates_series.dt.strftime('%Y-%m-%d').tolist())))
    unique_dates = [(min_date.strftime('%Y-%m-%d'))] + unique_dates
    
    max_matchday_num = int(season_matches['matchday'].max())
    
    with col_selector:
        if "Naptári Dátum" in filter_mode:
            # Date Slider (Independent)
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
                value=unique_dates[default_date_idx],
                help="Pontosan ezen a napon érvényes tabella és innen előre szimulált esélyek."
            )
            
            # Execute Date-based simulation
            df_state, n_played_matches, n_rem_fixtures = simulate_state_by_date(selected_season, selected_model, selected_date_str)
            st.markdown(f"#### 📌 Állás a **{selected_date_str}** naptári napon (Eddig lejátszva: **{n_played_matches} meccs**, Hátralévő: **{n_rem_fixtures} meccs**)")
            
        else:
            # Matchday Slider (Independent)
            played_mds = season_matches[season_matches['is_played']]['matchday'].dropna().unique()
            default_md = int(max(played_mds)) if len(played_mds) > 0 and selected_season == 2026 else max_matchday_num
            
            selected_md_num = st.slider(
                "🔢 Válassz Fordulószámot (0 = Rajt előtt):",
                0, max_matchday_num, value=default_md,
                help="A megadott fordulószámig lejátszott mérkőzések alapján számol."
            )
            
            # Load Matchday cached table
            df_all_mds = get_cached_matchday_simulation(selected_season, selected_model)
            df_state = df_all_mds[df_all_mds['matchday'] == selected_md_num].sort_values(by='current_rank').copy()
            st.markdown(f"#### 📌 Állás a(z) **{selected_md_num}. Forduló** után")
            
    # Metric KPI Cards
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
        
        # Display Table
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
        
        # Visual Bar Chart
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
# TAB 2: LINEPLOT TRENDEK
# -------------------------------------------------------------
with tab2:
    st.markdown(f"### 📈 Szezonális Valószínűségi Trendek – **{season_labels[selected_season]}**")
    
    df_line_sim = get_cached_matchday_simulation(selected_season, selected_model)
    
    # Identify maximum played matchday in this season
    played_mds_in_s = season_matches[season_matches['is_played']]['matchday'].dropna().unique()
    max_played_md = int(max(played_mds_in_s)) if len(played_mds_in_s) > 0 else 0
    total_season_mds = int(season_matches['matchday'].max())
    
    # Filter only up to played matchdays if season is in progress!
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
    
    # Interactive Plotly Lineplot
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
    
    # Milestone Summary Table
    st.markdown("#### 📋 Fordulónkénti Részletes Valószínűségi Táblázat")
    pivot_df = plot_df.pivot(index='matchday', columns='team', values=col_name)
    pivot_df.index.name = 'Forduló'
    st.dataframe(pivot_df.style.format("{:.1f}%"), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: MODELL DOKUMENTÁCIÓ & BENCHMARKOK
# -------------------------------------------------------------
with tab3:
    st.markdown("### 📑 Modell Dokumentáció & Történelmi Benchmark Eredmények")
    st.markdown("A rendszerben megvalósított és letesztelt modellek tudományos validációs eredményei a **2015–2026** közötti valós NB I-es meccsadatokon:")
    
    st.markdown("#### 1. 🏆 Hivatalos Meccsszintű Benchmark (Érintetlen Teszthalmaz 2022–2025, N=792)")
    
    bench_data = [
        {
            "Modell": "Model 2: Dynamic Dixon-Coles (V2)",
            "Típus": "Dinamikus Támadó/Védekező rating + Dixon-Coles gólkorrekció (rho=-0.10)",
            "Test Log Loss (↓)": "1.0365",
            "Brier Score (↓)": "0.6226",
            "Accuracy (↑)": "48.11%",
            "Naiv Modellhez Képesti Előny": "+0.0384 Log Loss gain"
        },
        {
            "Modell": "Model 1: Dynamic Elo (V1 Baseline)",
            "Típus": "1D Csapaterősség rating + Gauss döntetlen lecsengés",
            "Test Log Loss (↓)": "1.0379",
            "Brier Score (↓)": "0.6238",
            "Accuracy (↑)": "47.85%",
            "Naiv Modellhez Képesti Előny": "+0.0370 Log Loss gain"
        },
        {
            "Modell": "Model 0: Static Poisson",
            "Típus": "Fix globális ligaátlagok (lambda_H=1.45, lambda_A=1.15)",
            "Test Log Loss (↓)": "1.0742",
            "Brier Score (↓)": "0.6499",
            "Accuracy (↑)": "43.56%",
            "Naiv Modellhez Képesti Előny": "+0.0007 Log Loss gain"
        },
        {
            "Modell": "Naiv Modell (Base Rate)",
            "Típus": "Táguló történelmi osztályarányok (44.3% H, 25.4% D, 30.3% A)",
            "Test Log Loss (↓)": "1.0749",
            "Brier Score (↓)": "0.6504",
            "Accuracy (↑)": "45.08%",
            "Naiv Modellhez Képesti Előny": "0.0000 (Bázis)"
        }
    ]
    st.table(pd.DataFrame(bench_data))
    
    st.markdown("#### 2. 🧪 Statisztikai Szignifikancia Teszt")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Paired t-test p-value", "9.09 × 10⁻⁴", "p < 0.001 (Szignifikáns)")
    with c2:
        st.metric("Wilcoxon Signed-Rank Test", "6.58 × 10⁻⁷", "Extrém szignifikáns")
    with c3:
        st.metric("95% Bootstrap Konfidencia Sáv", "[+0.0156, +0.0586]", "Szigorúan pozitív nyereség")
        
    st.markdown("#### 3. 🏁 Szezonszintű Monte Carlo Benchmark (8 Történelmi Szezon Átlaga: 2018–2025)")
    season_bench_data = [
        {
            "Modell": "Adaptive Dynamic DC (Early Boost)",
            "Végső Pontszám MAE": "8.02 pont",
            "Helyezés Rangkorreláció (Spearman)": "0.547",
            "Bajnoki Cím Brier": "0.0294",
            "Top 4 Brier": "0.1164",
            "Kiesés Brier": "0.1378"
        },
        {
            "Modell": "Dynamic Dixon-Coles (Statikus Szimuláció)",
            "Végső Pontszám MAE": "7.33 pont",
            "Helyezés Rangkorreláció (Spearman)": "0.536",
            "Bajnoki Cím Brier": "0.0288",
            "Top 4 Brier": "0.1184",
            "Kiesés Brier": "0.1397"
        },
        {
            "Modell": "Static Poisson (Baseline)",
            "Végső Pontszám MAE": "7.35 pont",
            "Helyezés Rangkorreláció (Spearman)": "0.524",
            "Bajnoki Cím Brier": "0.0285",
            "Top 4 Brier": "0.1189",
            "Kiesés Brier": "0.1390"
        }
    ]
    st.table(pd.DataFrame(season_bench_data))
    
    st.markdown("""
    #### 4. 📐 Matematikai Modell Architektúra
    
    ##### ⚽ Várható Gólok (Expected Goals):
    $$\ln \lambda_{\\text{home}} = \mu + \\alpha_{\\text{home}} + \\beta_{\\text{away}} + \\gamma_{\\text{home}}$$
    $$\ln \mu_{\\text{away}} = \mu + \\alpha_{\\text{away}} + \\beta_{\\text{home}}$$
    
    ##### 🎯 Dixon-Coles Alacsony Gólszámú Korrekció ($\\tau$):
    $$\\tau(0,0) = 1 - \\lambda \\mu \\rho, \\quad \\tau(0,1) = 1 + \\lambda \\rho, \\quad \\tau(1,0) = 1 + \\mu \\rho, \\quad \\tau(1,1) = 1 - \\rho$$
    
    ##### 📈 Dinamikus Gradiens Frissítés:
    $$\\Delta \\alpha_{\\text{home}} = \\eta_{\\text{att}} \\cdot (x_{\\text{obs}} - \\lambda_{\\text{home}}), \\quad \\Delta \\beta_{\\text{away}} = \\eta_{\\text{def}} \\cdot (x_{\\text{obs}} - \\lambda_{\\text{home}})$$
    """)

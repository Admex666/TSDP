import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.calibration import calibration_curve
import plotly.express as px
import plotly.graph_objects as go

# Set page config for a premium look
st.set_page_config(
    page_title="NB1 2024-25 | Odds vs Teljesítmény",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium feel
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .plot-container {
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚽ NB1 2024-25: Odds vs Valóság")
st.markdown("""
### Mennyire látták előre a fogadóirodák a szezont?
Ez az elemzés a **SofaScore** adatait használja, összevetve a mérkőzések előtti záró odds-okat a tényleges eredményekkel.
""")

@st.cache_data(ttl=60) # Cache for 1 minute to allow updates
def load_data():
    try:
        with open("nb1_2024_25_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return None
        df = pd.DataFrame(data)
        return df
    except (FileNotFoundError, json.JSONDecodeError):
        return None

df_raw = load_data()

if df_raw is None or df_raw.empty:
    st.warning("⚠️ Adatok betöltése folyamatban... Kérlek várj, amíg a `data_fetcher.py` begyűjti az első meccseket!")
    if st.button("Frissítés"):
        st.rerun()
    st.stop()

# --- Preprocessing ---
team_data = []
for idx, row in df_raw.iterrows():
    # Home team
    team_data.append({
        "round": row["round"],
        "team": row["home_team"],
        "pts": row["home_pts"],
        "e_pts": row["e_home_pts"],
        "opponent": row["away_team"],
        "is_home": True,
        "prob_win": row["prob_home"],
        "result": row["winner"] == "Home"
    })
    # Away team
    team_data.append({
        "round": row["round"],
        "team": row["away_team"],
        "pts": row["away_pts"],
        "e_pts": row["e_away_pts"],
        "opponent": row["home_team"],
        "is_home": False,
        "prob_win": row["prob_away"],
        "result": row["winner"] == "Away"
    })

df_teams = pd.DataFrame(team_data)
df_teams = df_teams.sort_values(["team", "round"])

# Cumulative calculations
df_teams["cum_pts"] = df_teams.groupby("team")["pts"].cumsum()
df_teams["cum_e_pts"] = df_teams.groupby("team")["e_pts"].cumsum()
df_teams["diff"] = df_teams["pts"] - df_teams["e_pts"]
df_teams["cum_diff"] = df_teams.groupby("team")["diff"].cumsum()

# --- Sidebar ---
st.sidebar.header("Beállítások")
teams = sorted(df_teams["team"].unique())
selected_teams = st.sidebar.multiselect("Csapatok összehasonlítása:", teams, default=teams[:4])

# --- Metrics Overview ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

total_games = len(df_raw)
col_m1.metric("Feldolgozott meccsek", total_games)

# Favorite accuracy
df_raw["favorite_prob"] = df_raw[["prob_home", "prob_draw", "prob_away"]].max(axis=1)
def is_fav_win(row):
    if row["prob_home"] == row["favorite_prob"] and row["winner"] == "Home": return True
    if row["prob_away"] == row["favorite_prob"] and row["winner"] == "Away": return True
    if row["prob_draw"] == row["favorite_prob"] and row["winner"] == "Draw": return True
    return False
df_raw["favorite_win"] = df_raw.apply(is_fav_win, axis=1)
fav_acc = df_raw["favorite_win"].mean()
col_m2.metric("Favorit bejött", f"{fav_acc:.1%}")

# Biggest overperformer
summary = df_teams.groupby("team")["diff"].sum().sort_values(ascending=False)
top_team = summary.index[0]
top_val = summary.iloc[0]
col_m3.metric("Legnagyobb túlteljesítő", top_team, f"+{top_val:.1f} pts")

# Consistency (Brier Score equivalent for points)
mse_total = (df_teams["pts"] - df_teams["e_pts"]).pow(2).mean()
col_m4.metric("Kiszámíthatatlanság", f"{mse_total:.2f}", help="Átlagos négyzetes hiba a pontok és várható pontok között.")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Idősoros Elemzés", "🎯 Fogadóiroda Precízió", "🏆 Teljesítmény Rangsor", "🔍 Csapat Profil"])

with tab1:
    st.subheader("Kumulatív Pontszámok: Valóság vs Odds-ok")
    
    # Plotly for interactivity
    fig_cum = go.Figure()
    for team in selected_teams:
        team_df = df_teams[df_teams["team"] == team]
        # Real points
        fig_cum.add_trace(go.Scatter(x=team_df["round"], y=team_df["cum_pts"], name=f"{team} (Real)", mode='lines+markers'))
        # Expected points
        fig_cum.add_trace(go.Scatter(x=team_df["round"], y=team_df["cum_e_pts"], name=f"{team} (Exp)", mode='lines', line=dict(dash='dash')))
    
    fig_cum.update_layout(height=600, xaxis_title="Forduló", yaxis_title="Pont", hovermode="x unified")
    st.plotly_chart(fig_cum, use_container_width=True)

    st.subheader("Túl- és Alulteljesítés az időben")
    fig_diff = px.line(df_teams[df_teams["team"].isin(selected_teams)], 
                       x="round", y="cum_diff", color="team", markers=True,
                       labels={"cum_diff": "Kumulatív Különbség (Real - Exp)", "round": "Forduló"},
                       title="Ha a görbe felfelé megy, a csapat jobban játszik, mint amit az odds sugall")
    fig_diff.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_diff, use_container_width=True)

with tab2:
    st.subheader("A 'Valószínűség' tényleg annyit jelent?")
    
    # Calibration data
    all_outcomes = []
    all_probs = []
    for _, row in df_raw.iterrows():
        all_outcomes.extend([1 if row["winner"] == "Home" else 0, 
                             1 if row["winner"] == "Draw" else 0, 
                             1 if row["winner"] == "Away" else 0])
        all_probs.extend([row["prob_home"], row["prob_draw"], row["prob_away"]])
    
    prob_true, prob_pred = calibration_curve(all_outcomes, all_probs, n_bins=8)
    
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Tökéletes', line=dict(color='gray', dash='dash')))
    fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Fogadóiroda'))
    fig_cal.update_layout(title="Kalibrációs görbe", xaxis_title="Jósolt valószínűség", yaxis_title="Tényleges gyakoriság", width=700, height=500)
    st.plotly_chart(fig_cal)
    
    st.info("""
    **Hogyan olvasd?** Ha a görbe a szaggatott vonal felett van, az iroda alulbecsülte az eseményt. Ha alatta, akkor túlbecsülte.
    """)

with tab3:
    st.subheader("NB1 'Szerencse' és Teljesítmény Táblázat")
    
    final_summary = df_teams.groupby("team").agg({
        "pts": "sum",
        "e_pts": "sum",
        "diff": "sum"
    }).reset_index()
    
    final_summary.columns = ["Csapat", "Pont", "Várható Pont", "Különbség"]
    final_summary = final_summary.sort_values("Különbség", ascending=False)
    
    st.dataframe(
        final_summary.style.background_gradient(subset=["Különbség"], cmap="RdYlGn")
        .format({"Várható Pont": "{:.2f}", "Különbség": "{:+.2f}"}),
        use_container_width=True
    )
    
    fig_bar = px.bar(final_summary, x="Különbség", y="Csapat", orientation='h',
                     color="Különbség", color_continuous_scale="RdYlGn",
                     title="Melyik csapat borítja leginkább a papírformát?")
    st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    selected_profile = st.selectbox("Válassz egy csapatot a részletes elemzéshez:", teams)
    
    prof_df = df_teams[df_teams["team"] == selected_profile]
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.write(f"**{selected_profile}** mérkőzései és elvárások:")
        display_df = prof_df[["round", "opponent", "is_home", "pts", "e_pts", "diff"]].copy()
        display_df["is_home"] = display_df["is_home"].map({True: "Otthon", False: "Idegenben"})
        st.table(display_df.style.format({"e_pts": "{:.2f}", "diff": "{:+.2f}"}))
        
    with col_p2:
        # Mini trend
        st.write("Teljesítmény trend (utolsó 5 meccs)")
        last_5 = prof_df.tail(5)
        fig_trend = px.bar(last_5, x="round", y="diff", color="diff", 
                           color_continuous_scale="RdYlGn", labels={"diff": "Különbség"})
        st.plotly_chart(fig_trend, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Készítette: Antigravity AI")

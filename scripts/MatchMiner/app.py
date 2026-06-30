import streamlit as st
import os
import pandas as pd
import numpy as np
import textwrap
from data_loader import load_team_data, load_player_data
from insight_engine import analyze_game
from narrative_formatter import format_team_insight, format_player_insight

# Set page configuration
st.set_page_config(
    page_title="MatchMiner | Live Story Detection Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to render HTML safely in Streamlit without triggering markdown formatting issues
def render_html(html_str):
    # Remove all newlines and multiple spaces to make it a single-line string
    single_line = " ".join([line.strip() for line in html_str.split('\n')])
    st.markdown(single_line, unsafe_allow_html=True)

# Custom Premium CSS Styling with responsive card support
render_html("""
    <style>
        /* Import modern Google font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        /* Box sizing reset to prevent layout overflow */
        *, *:before, *:after {
            box-sizing: border-box;
        }
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Global Background Accent */
        .stApp {
            background-color: #0f172a;
            color: #f8fafc;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1e293b;
            border-right: 1px solid #334155;
        }
        
        /* Header Gradient & Typography */
        .main-title {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-align: center;
            letter-spacing: -0.05em;
        }
        
        .subtitle {
            font-size: 1.25rem;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Match Card details */
        .match-header-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
            text-align: center;
        }
        
        .match-teams {
            font-size: 2rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 0.5rem;
        }
        
        .match-meta {
            font-size: 0.95rem;
            color: #38bdf8;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
        }
        
        /* Story Card styling - responsive and strict widths */
        .story-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: transform 0.2s, border-color 0.2s;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 100%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        
        .story-card:hover {
            transform: translateY(-2px);
            border-color: #6366f1;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        }
        
        /* Score/Z-score pill badges */
        .badge-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
            align-items: center;
        }
        
        .badge-score {
            background: linear-gradient(90deg, #6366f1, #4f46e5);
            color: white;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .badge-z {
            background-color: #0284c7;
            color: white;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .badge-z-extreme {
            background: linear-gradient(90deg, #ef4444, #dc2626);
            color: white;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .badge-record {
            background: linear-gradient(90deg, #eab308, #ca8a04);
            color: #0f172a;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
        }
        
        .story-text {
            font-size: 1.05rem;
            line-height: 1.5;
            color: #e2e8f0;
        }
        
        /* Gradient Dividers */
        .gradient-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, transparent);
            margin: 2rem 0;
        }
    </style>
""")

# Main Title Header
st.markdown("<div class='main-title'>MatchMiner</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Live Story & Anomaly Detection Engine for Football Analytics</div>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/football.png", width=60)
st.sidebar.markdown("### Control Center")

# File Uploaders or Fallbacks
st.sidebar.markdown("#### 1. Data Source")
team_file_upload = st.sidebar.file_uploader("Upload Team Match CSV", type=["csv"])
player_file_upload = st.sidebar.file_uploader("Upload Player Match CSV", type=["csv"])

# Define default fallback paths
DEFAULT_TEAM_PATH = "data/team0630_1752.csv"
DEFAULT_PLAYER_PATH = "data/player0630_1752.csv"

# Load Team data
team_df = None
if team_file_upload is not None:
    team_df = load_team_data(team_file_upload)
    st.sidebar.success("Loaded custom team data.")
elif os.path.exists(DEFAULT_TEAM_PATH):
    team_df = load_team_data(DEFAULT_TEAM_PATH)
    st.sidebar.info("Using default team data.")
else:
    st.error("Please upload a Team CSV file or check that data/team0630_1752.csv exists.")

# Load Player data
player_df = None
if player_file_upload is not None:
    player_df = load_player_data(player_file_upload)
    st.sidebar.success("Loaded custom player data.")
elif os.path.exists(DEFAULT_PLAYER_PATH):
    player_df = load_player_data(DEFAULT_PLAYER_PATH)
    st.sidebar.info("Using default player data.")
else:
    st.error("Please upload a Player CSV file or check that data/player0630_1752.csv exists.")

if team_df is not None and player_df is not None:
    # 2. Select Game
    st.sidebar.markdown("#### 2. Game Selection")
    
    # Generate match list sorted by date (newest first)
    unique_matches = team_df.drop_duplicates(subset=['gameId']).sort_values(by='Date', ascending=False)
    
    match_options = {}
    for _, row in unique_matches.iterrows():
        match_label = f"{row['Date'].strftime('%Y-%m-%d')} - {row['game']} ({row.get('leagueName', 'League')})"
        match_options[row['gameId']] = match_label
        
    selected_game_id = st.sidebar.selectbox(
        "Choose Match to Analyze",
        options=list(match_options.keys()),
        format_func=lambda x: match_options[x]
    )
    
    # 3. Parameters
    st.sidebar.markdown("#### 3. Parameters")
    min_z = st.sidebar.slider(
        "Min Z-Score (Anomaly Threshold)",
        min_value=0.5,
        max_value=4.0,
        value=1.5,
        step=0.25,
        help="Higher values show only extremely unusual deviations from league standards."
    )
    
    min_mins = st.sidebar.slider(
        "Min Minutes Played",
        min_value=1,
        max_value=90,
        value=15,
        step=5,
        help="Exclude players with fewer minutes to filter out small-sample noise."
    )
    
    top_n = st.sidebar.slider(
        "Show Top N Stories",
        min_value=1,
        max_value=15,
        value=5,
        step=1
    )
    
    # 4. per90 Correction Switch
    st.sidebar.markdown("#### 4. Stat Correction")
    use_per90 = st.sidebar.checkbox(
        "Scale player stats to per 90 mins",
        value=False,
        help="When enabled, player volume metrics are scaled to a 90-minute equivalent (Stat * 90 / Min). Baselines are also computed on per90 values."
    )
    
    # 5. Perform Analysis
    game_info = team_df[team_df['gameId'] == selected_game_id].iloc[0]
    opposing_info = team_df[(team_df['gameId'] == selected_game_id) & (team_df['Team'] != game_info['Team'])]
    
    opp_team_name = opposing_info.iloc[0]['Team'] if not opposing_info.empty else "Opponent"
    score_str = f"{int(game_info['score'])} - {int(game_info['finalScoreOpponent'])}" if game_info['Home'] else f"{int(game_info['finalScoreOpponent'])} - {int(game_info['score'])}"
    home_team = game_info['Team'] if game_info['Home'] else opp_team_name
    away_team = opp_team_name if game_info['Home'] else game_info['Team']
    
    # Render Match Header Card (single line string)
    render_html(f"""
        <div class='match-header-card'>
            <div class='match-meta'>{game_info.get('leagueName', 'Tournament')} • Matchday / Week {game_info.get('Week', 1)} • {game_info['Date'].strftime('%B %d, %Y')}</div>
            <div class='match-teams'>{home_team} {score_str} {away_team}</div>
        </div>
    """)
    
    # Run Engine
    with st.spinner("Analyzing statistics & searching historical records..."):
        analysis = analyze_game(
            team_df=team_df,
            player_df=player_df,
            target_game_id=selected_game_id,
            min_player_minutes=min_mins,
            min_z_score=min_z,
            per90=use_per90
        )
        
    if "error" in analysis:
        st.error(analysis["error"])
    else:
        # Columns for side-by-side display
        col_team, col_player = st.columns(2)
        
        with col_team:
            st.markdown("### 🛡️ Csapat Érdekességek (Team Stories)")
            st.markdown("<div class='gradient-divider' style='margin: 0.5rem 0 1rem 0;'></div>", unsafe_allow_html=True)
            
            top_teams = analysis["team_insights"][:top_n]
            
            if not top_teams:
                st.info("Nem találtunk a megadott küszöbérték feletti csapatszintű anomáliát.")
            else:
                for ins in top_teams:
                    z_class = "badge-z-extreme" if abs(ins['z_score']) >= 2.5 else "badge-z"
                    record_badge = "<span class='badge-record'>🏆 REKORD</span>" if ins['is_record'] else ""
                    
                    card_html = f"""
                        <div class='story-card'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <strong style='font-size: 1.1rem; color: #38bdf8;'>{ins['team']}</strong>
                                <span style='color: #94a3b8; font-size: 0.85rem;'>Pont: {ins['score']}</span>
                            </div>
                            <div class='badge-container'>
                                <span class='badge-score'>Sztori Érték</span>
                                <span class='{z_class}'>Z-Score: {ins['z_score']:.2f}</span>
                                {record_badge}
                            </div>
                            <div class='story-text'>{format_team_insight(ins)}</div>
                        </div>
                    """
                    render_html(card_html)
                    
        with col_player:
            st.markdown("### 🏃‍♂️ Játékos Érdekességek (Player Stories)")
            st.markdown("<div class='gradient-divider' style='margin: 0.5rem 0 1rem 0;'></div>", unsafe_allow_html=True)
            
            top_players = analysis["player_insights"][:top_n]
            
            if not top_players:
                st.info("Nem találtunk a megadott küszöbérték feletti egyéni játékos anomáliát.")
            else:
                for ins in top_players:
                    z_class = "badge-z-extreme" if abs(ins['z_score_pos']) >= 2.5 else "badge-z"
                    record_badge = "<span class='badge-record'>🏆 REKORD</span>" if ins['is_record'] else ""
                    
                    card_html = f"""
                        <div class='story-card'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <strong style='font-size: 1.1rem; color: #818cf8;'>{ins['player']}</strong>
                                <span style='color: #94a3b8; font-size: 0.85rem;'>Pont: {ins['score']}</span>
                            </div>
                            <div class='badge-container'>
                                <span class='badge-score'>Sztori Érték</span>
                                <span class='{z_class}'>Z-Score (Poz.): {ins['z_score_pos']:.2f}</span>
                                {record_badge}
                            </div>
                            <div class='story-text'>{format_player_insight(ins)}</div>
                        </div>
                    """
                    render_html(card_html)

else:
    st.info("Kérjük töltsd fel a csapat és játékos fájlokat a bal oldali menüben az elemzés elindításához.")

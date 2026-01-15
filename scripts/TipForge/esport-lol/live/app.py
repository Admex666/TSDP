
import streamlit as st
import pandas as pd
import time
import json
import os
import sys
import logging
import joblib
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from riot_api import RiotEsportsAPI
from discovery import TippmixDiscovery
from scrapers import OddsScraper
from value_betting import ValueBettingEngine

# Page config
st.set_page_config(
    page_title="TipForge Live Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'riot_api' not in st.session_state:
    st.session_state.riot_api = RiotEsportsAPI()

if 'discovery' not in st.session_state:
    st.session_state.discovery = TippmixDiscovery(headless=True)

if 'odds_scraper' not in st.session_state:
    st.session_state.odds_scraper = OddsScraper(headless=True)

if 'tracked_matches' not in st.session_state:
    st.session_state.tracked_matches = {} # {riot_game_id: {info...}}

if 'tippmix_matches' not in st.session_state:
    st.session_state.tippmix_matches = {}

if 'riot_live_games' not in st.session_state:
    st.session_state.riot_live_games = []

if 'models_loaded' not in st.session_state:
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(root_dir, "models")
        
        gb_model = joblib.load(os.path.join(models_dir, "live_gb_model_20251031.joblib"))
        rf_model = joblib.load(os.path.join(models_dir, "live_rf_model_20251031.joblib"))
        scaler = joblib.load(os.path.join(models_dir, "live_scaler_20251031.joblib"))
        
        st.session_state.engine = ValueBettingEngine(
            gb_model, rf_model, scaler,
            min_edge=0.03,
            min_confidence=0.4
        )
        st.session_state.models_loaded = True
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.session_state.models_loaded = False

# Sidebar
with st.sidebar:
    st.header("⚡ Live Scanner Control")
    
    if st.session_state.models_loaded:
        st.success("✅ Models Loaded")
    else:
        st.error("❌ Models Failed")
        
    st.divider()
    
    if st.button("🔄 Refresh Riot Live Games"):
        with st.spinner("Fetching live games from Riot..."):
            api = st.session_state.riot_api
            live_events = api.get_live()
            
            # Fallback to schedule if no live events returned directly
            if not live_events:
                schedule = api.get_schedule()
                live_events = [e for e in schedule if e.get('state') == 'inProgress']
            
            games_found = []
            for event in live_events:
                event_id = event.get('id')
                league_name = event.get('league', {}).get('name', 'Unknown')
                
                # Fetch details
                details = api.get_event_details(event_id)
                if not details: continue
                
                match_data = details.get('match', {})
                teams = match_data.get('teams', [])
                if len(teams) < 2: continue
                
                blue_name = teams[0].get('name', teams[0].get('code', 'Unknown'))
                red_name = teams[1].get('name', teams[1].get('code', 'Unknown'))
                
                games = match_data.get('games', [])
                for g in games:
                    if g.get('state') == 'inProgress':
                        games_found.append({
                            'id': g['id'],
                            'event_id': event_id,
                            'league': league_name,
                            'blue': blue_name,
                            'red': red_name,
                            'label': f"[{league_name}] {blue_name} vs {red_name}"
                        })
            
            st.session_state.riot_live_games = games_found
            st.success(f"Found {len(games_found)} live games.")

    if st.button("🌐 Discover Tippmix Matches"):
        with st.spinner("Scraping Tippmix..."):
            discovery = st.session_state.discovery
            matches = discovery.discover_lol_matches()
            st.session_state.tippmix_matches = matches
            st.success(f"Found {len(matches)} Tippmix matches.")
            
    st.divider()
    st.subheader("Active Tracking")
    if st.button("🗑️ Clear All Tracked"):
        st.session_state.tracked_matches = {}
        st.rerun()

# Main Content
tab_mapping, tab_dashboard, tab_debug = st.tabs(["🗺️ Match Mapping", "📊 Live Dashboard", "🐞 Debug"])

with tab_mapping:
    st.subheader("Map Live Games")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 1. Select Riot Game")
        riot_options = {g['id']: g['label'] for g in st.session_state.riot_live_games}
        selected_riot_id = st.selectbox(
            "Live Games", 
            options=list(riot_options.keys()),
            format_func=lambda x: riot_options[x],
            key="map_riot_select"
        )
    
    with col2:
        st.write("### 2. Select Tippmix Match")
        tippmix_options = st.session_state.tippmix_matches
        # Add a custom option for manual URL
        match_keys = list(tippmix_options.keys())
        selected_tippmix_key = st.selectbox(
            "Discovered Matches",
            options=["Manual URL"] + match_keys,
            key="map_tippmix_select"
        )
        
        tippmix_url = ""
        if selected_tippmix_key == "Manual URL":
            tippmix_url = st.text_input("Enter Tippmix URL")
        else:
            tippmix_url = tippmix_options[selected_tippmix_key]
            st.info(f"URL: {tippmix_url}")
            
    st.write("### 3. Team Mapping")
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        st.info("How does Tippmix 'Hazai' (Home) map to Riot Side?")
        mapping_choice = st.radio(
            "Mapping",
            ["Hazai = BLUE Side", "Hazai = RED Side"],
            horizontal=True
        )
        home_is_blue = (mapping_choice == "Hazai = BLUE Side")

    with col_map2:
        if st.button("➕ Start Tracking Match", use_container_width=True):
            if selected_riot_id and tippmix_url:
                riot_game_info = next((g for g in st.session_state.riot_live_games if g['id'] == selected_riot_id), None)
                if riot_game_info:
                    st.session_state.tracked_matches[selected_riot_id] = {
                        'info': riot_game_info,
                        'tippmix_url': tippmix_url,
                        'home_is_blue': home_is_blue,
                        'last_update': None,
                        'history': []
                    }
                    st.success(f"Tracking started for {riot_game_info['label']}")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("Please select both a Riot game and a Tippmix match/URL.")

with tab_dashboard:
    st.subheader("Live Value Betting Dashboard")
    
    # Auto-refresh mechanism
    auto_refresh = st.toggle("Auto-Refresh (30s)", value=False)
    
    if st.button("🔄 Refresh Stats Now"):
        st.rerun()
        
    if not st.session_state.tracked_matches:
        st.warning("No matches currently being tracked. Go to 'Match Mapping' to add one.")
    
    for game_id, match_data in st.session_state.tracked_matches.items():
        info = match_data['info']
        url = match_data['tippmix_url']
        home_is_blue = match_data['home_is_blue']
        
        with st.container(border=True):
            # Header
            st.markdown(f"### {info['label']}")
            st.caption(f"ID: {game_id} | Tippmix: {url} | Mapping: {'Hazai=BLUE' if home_is_blue else 'Hazai=RED'}")
            
            # Fetch Live Data
            api = st.session_state.riot_api
            match_state = api.get_latest_match_state(game_id)
            
            if not match_state:
                st.error("Could not fetch live stats from Riot API.")
                continue
                
            # Display Game Stats
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            
            blue = match_state['blue_team']
            red = match_state['red_team']
            
            with col_stats1:
                st.metric("Game Time", match_state['game_time'])
            
            with col_stats2:
                st.metric("Kills (Blue vs Red)", f"{blue['kills']} - {red['kills']}")
            
            with col_stats3:
                gold_diff = blue['gold'] - red['gold']
                st.metric("Gold Diff", f"{gold_diff:+,}", delta_color="normal")
            
            with col_stats4:
                # Show dragon types, not just count
                blue_drakes = blue['dragons']
                red_drakes = red['dragons']
                drake_text = f"{len(blue_drakes)} - {len(red_drakes)}"
                
                # Show types in tooltip/caption
                if blue_drakes or red_drakes:
                    blue_types = ", ".join([d.capitalize() for d in blue_drakes]) if blue_drakes else "None"
                    red_types = ", ".join([d.capitalize() for d in red_drakes]) if red_drakes else "None"
                    st.metric("Dragons", drake_text, help=f"Blue: {blue_types} | Red: {red_types}")
                else:
                    st.metric("Dragons", drake_text)
                
            # Predictions
            engine = st.session_state.engine
            features = engine.calculate_features(match_state)
            prob_blue, prob_red = engine.predict_win_probability(features)
            
            st.progress(prob_blue, text=f"**BLUE Win Probability: {prob_blue:.1%}** (Red: {prob_red:.1%})")
            
            # Scrape Odds & Calculate Value
            odds_scraper = st.session_state.odds_scraper
            odds_data = odds_scraper.scrape(url)
            
            if odds_data and odds_data.get('markets'):
                st.write("#### 💰 Live Odds & Value")
                
                value_bets = engine.find_value_bets(
                    match_state, odds_data, 
                    home_is_blue=home_is_blue
                )
                
                if value_bets:
                    for vb in value_bets:
                        color = "green" if vb['confidence'] == 'HIGH' else "orange" if vb['confidence'] == 'MEDIUM' else "gray"
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; border: 1px solid {color}; background-color: rgba(0,255,0,0.1);">
                            <strong>🎯 {vb['team_name']} ({vb['team']})</strong> @ {vb['odds']:.2f}<br>
                            Edge: <strong>{vb['edge']:.1f}%</strong> | Confidence: {vb['confidence']} | Kelly: {vb['kelly_fraction']:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No value bets found currently.")
                    
                # Store history
                timestamp = datetime.now().isoformat()
                history_entry = {
                    'time': timestamp,
                    'game_time': match_state['game_time'],
                    'prob_blue': prob_blue,
                    'gold_diff': gold_diff
                }
                match_data['history'].append(history_entry)
                
                # Show raw market data (simplified)
                with st.expander("Live Markets"):
                    for m in odds_data['markets']:
                        st.write(f"**{m['name']}**")
                        cols = st.columns(len(m['options']))
                        for i, opt in enumerate(m['options']):
                             cols[i].metric(opt['name'], f"{opt['odds']:.2f}")

            else:
                st.warning("Could not scrape odds. Check if market is suspended.")

            # Store match state for debug
            st.session_state.last_match_state = match_state
            if odds_data:
                st.session_state.last_odds_data = odds_data

    if auto_refresh:
        time.sleep(30)
        st.rerun()

with tab_debug:
    st.subheader("Debug Information")
    
    col_dbg1, col_dbg2 = st.columns(2)
    
    with col_dbg1:
        st.write("#### Last Riot Match State")
        if 'last_match_state' in st.session_state:
            st.json(st.session_state.last_match_state)
        else:
            st.info("No match state available yet.")
            
    with col_dbg2:
        st.write("#### Last Odds Data")
        if 'last_odds_data' in st.session_state:
            st.json(st.session_state.last_odds_data)
        else:
            st.info("No odds data available yet.")
            
    st.divider()
    st.write("#### Log Tail")
    try:
        with open("live_scanner.log", "r") as f:
            lines = f.readlines()
            st.text_area("Log Output", "".join(lines[-20:]), height=300)
    except:
        st.info("No log file found.")
"""
Advanced Live Value Betting Dashboard
Real-time match tracking with automated value bet detection
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import joblib
import os
from typing import Dict, List

# Import our custom modules (adjust imports based on your file structure)
# from scrapers import MatchStatsScraper, OddsScraper
# from value_betting import ValueBettingEngine
# from live_tracker import LiveMatchTracker

# Page config
st.set_page_config(
    page_title="🎯 Live Value Betting Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .value-bet-card {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00ff00;
        background-color: #1e1e1e;
        margin: 10px 0;
    }
    .high-confidence {
        border-left-color: #00ff00 !important;
    }
    .medium-confidence {
        border-left-color: #ffaa00 !important;
    }
    .low-confidence {
        border-left-color: #ff6600 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tracking_active' not in st.session_state:
    st.session_state.tracking_active = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'value_bets' not in st.session_state:
    st.session_state.value_bets = []
if 'match_history' not in st.session_state:
    st.session_state.match_history = []
if 'probability_history' not in st.session_state:
    st.session_state.probability_history = []

# Load models
@st.cache_resource
def load_models():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))

        gb_model = joblib.load(os.path.join(BASE_DIR, "models", "live_gb_model_20251031.joblib"))
        rf_model = joblib.load(os.path.join(BASE_DIR, "models", "live_rf_model_20251031.joblib"))
        scaler = joblib.load(os.path.join(BASE_DIR, "models", "live_scaler_20251031.joblib"))
        return gb_model, rf_model, scaler
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

gb_model, rf_model, scaler = load_models()

# Sidebar - Configuration
st.sidebar.title("⚙️ Configuration")

# Match URL inputs
st.sidebar.subheader("📊 Data Sources")
match_url = st.sidebar.text_input(
    "AndyDanger Match URL",
    value="https://andydanger.github.io/live-lol-esports/#/live/113475871523985235/game-index/3",
    help="URL to live match stats"
)

odds_url = st.sidebar.text_input(
    "Tippmix Odds URL",
    value="https://www.tippmixpro.hu/hu/elo/i/elo-esemenyek/100/league-of-legends-lol/...",
    help="URL to live betting odds"
)

# Tracking settings
st.sidebar.subheader("🔄 Tracking Settings")
update_interval = st.sidebar.slider(
    "Update Interval (seconds)",
    min_value=30,
    max_value=300,
    value=60,
    step=30,
    help="How often to scrape new data"
)

# Value bet criteria
st.sidebar.subheader("🎯 Value Bet Criteria")
min_edge = st.sidebar.slider(
    "Minimum Edge (%)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.5,
    help="Minimum expected value to flag as value bet"
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.50,
    max_value=0.75,
    value=0.55,
    step=0.01,
    format="%.2f",
    help="Minimum win probability to consider betting"
)

use_ensemble = st.sidebar.checkbox(
    "Use Ensemble Prediction",
    value=True,
    help="Average GB and RF model predictions"
)

# Auto-track toggle
st.sidebar.markdown("---")
auto_track_col1, auto_track_col2 = st.sidebar.columns([3, 1])
with auto_track_col1:
    st.write("**Auto-Tracking:**")
with auto_track_col2:
    if st.button("▶️" if not st.session_state.tracking_active else "⏸️"):
        st.session_state.tracking_active = not st.session_state.tracking_active
        st.rerun()

# Main content
st.title("🎯 LIVE VALUE BETTING DASHBOARD")
st.markdown("Real-time match analysis and value bet detection")

# Status bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    status = "🟢 TRACKING" if st.session_state.tracking_active else "🔴 PAUSED"
    st.metric("Status", status)
with col2:
    last_update = st.session_state.last_update or "Never"
    if isinstance(last_update, str) and last_update != "Never":
        last_update = datetime.fromisoformat(last_update).strftime("%H:%M:%S")
    st.metric("Last Update", last_update)
with col3:
    st.metric("Value Bets Found", len(st.session_state.value_bets))
with col4:
    st.metric("Update Interval", f"{update_interval}s")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Match", 
    "🎯 Value Bets", 
    "📈 Performance", 
    "🔍 History"
])

# Tab 1: Live Match
with tab1:
    st.subheader("🎮 Current Match State")
    
    # Manual refresh button
    col1, col2 = st.columns([1, 5])
    with col1:
        manual_refresh = st.button("🔄 Refresh Now", key="manual_refresh")
    
    # Placeholder for match data
    if 'current_match_data' in st.session_state and st.session_state.current_match_data:
        match_data = st.session_state.current_match_data
        
        # Game time and score
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            st.markdown(f"### ⏱️ {match_data.get('game_time', 'Unknown')}")
        with col2:
            blue_kills = match_data['blue_team'].get('kills', 0)
            red_kills = match_data['red_team'].get('kills', 0)
            st.markdown(f"### 🔵 {blue_kills} - {red_kills} 🔴")
        with col3:
            if 'predicted_probs' in st.session_state:
                prob_blue, prob_red = st.session_state.predicted_probs
                st.markdown(f"### {prob_blue:.1%} - {prob_red:.1%}")
        
        st.markdown("---")
        
        # Team stats comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔵 BLUE TEAM")
            blue = match_data['blue_team']
            st.metric("Kills", blue.get('kills', 0))
            st.metric("Towers", blue.get('towers', 0))
            st.metric("Gold", f"{blue.get('gold', 0):,}")
            st.metric("Dragons", len(blue.get('dragons', [])))
            st.metric("Barons", blue.get('barons', 0))
        
        with col2:
            st.markdown("#### 🔴 RED TEAM")
            red = match_data['red_team']
            st.metric("Kills", red.get('kills', 0))
            st.metric("Towers", red.get('towers', 0))
            st.metric("Gold", f"{red.get('gold', 0):,}")
            st.metric("Dragons", len(red.get('dragons', [])))
            st.metric("Barons", red.get('barons', 0))
        
        # Win probability chart
        if 'predicted_probs' in st.session_state:
            prob_blue, prob_red = st.session_state.predicted_probs
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Blue Team', 'Red Team'],
                    y=[prob_blue * 100, prob_red * 100],
                    marker_color=['#3b82f6', '#ef4444'],
                    text=[f"{prob_blue:.1%}", f"{prob_red:.1%}"],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Win Probability",
                yaxis_title="Probability (%)",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ====== ÚJ RÉSZ: Probability Over Time ======
        st.markdown("---")
        st.markdown("#### 📈 Win Probability Timeline")
        
        if st.session_state.probability_history and len(st.session_state.probability_history) > 1:
            # Convert to DataFrame
            history_df = pd.DataFrame(st.session_state.probability_history)
            
            # Create time series chart
            fig_timeline = go.Figure()
            
            fig_timeline.add_trace(go.Scatter(
                x=history_df['game_time'],
                y=history_df['blue_prob'] * 100,
                name='🔵 Blue Team',
                mode='lines+markers',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=6),
                fill='tonexty',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            
            fig_timeline.add_trace(go.Scatter(
                x=history_df['game_time'],
                y=history_df['red_prob'] * 100,
                name='🔴 Red Team',
                mode='lines+markers',
                line=dict(color='#ef4444', width=3),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            
            # Add 50% reference line
            fig_timeline.add_hline(
                y=50, 
                line_dash="dash", 
                line_color="gray",
                annotation_text="50% (Even)",
                annotation_position="right"
            )
            
            fig_timeline.update_layout(
                title="Win Probability Evolution",
                xaxis_title="Game Time",
                yaxis_title="Win Probability (%)",
                height=400,
                hovermode='x unified',
                yaxis=dict(range=[0, 100]),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Data table with key metrics
            st.markdown("##### 📊 Detailed Timeline")
            
            # Format the dataframe for display
            display_df = history_df[['game_time', 'blue_prob', 'red_prob', 'blue_kills', 'red_kills', 'gold_diff']].copy()
            display_df['blue_prob'] = display_df['blue_prob'].apply(lambda x: f"{x:.1%}")
            display_df['red_prob'] = display_df['red_prob'].apply(lambda x: f"{x:.1%}")
            display_df['gold_diff'] = display_df['gold_diff'].apply(lambda x: f"{x:+,}")
            display_df.columns = ['Game Time', 'Blue Win %', 'Red Win %', 'Blue Kills', 'Red Kills', 'Gold Diff']
            
            st.dataframe(
                display_df.iloc[::-1],  # Reverse to show newest first
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("📊 Timeline data will appear after a few updates...")
    
    else:
        st.info("👆 Click 'Refresh Now' or enable auto-tracking to load match data")

# Tab 2: Value Bets
with tab2:
    st.subheader("🎯 Identified Value Bets")
    
    if st.session_state.value_bets:
        for i, bet in enumerate(st.session_state.value_bets):
            confidence_class = f"{bet['confidence'].lower()}-confidence"
            
            with st.container():
                st.markdown(f"""
                <div class="value-bet-card {confidence_class}">
                    <h3>🎲 {bet['team_name']} @ {bet['odds']:.2f}</h3>
                    <p><strong>Market:</strong> {bet['market_name']}</p>
                    <p><strong>Game Time:</strong> {bet['game_time']}</p>
                    <p><strong>Edge:</strong> {bet['edge']:.1f}% | 
                       <strong>Confidence:</strong> {bet['confidence']}</p>
                    <p><strong>Predicted Probability:</strong> {bet['predicted_prob']:.1%} | 
                       <strong>Implied Probability:</strong> {bet['implied_prob']:.1%}</p>
                    <p><strong>Kelly Stake:</strong> {bet['kelly_fraction']:.1%} of bankroll</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("✅ Place Bet", key=f"place_{i}"):
                        st.success("Bet placed! (Simulation)")
                with col2:
                    if st.button("❌ Dismiss", key=f"dismiss_{i}"):
                        st.session_state.value_bets.pop(i)
                        st.rerun()
        
    else:
        st.info("No value bets identified yet. Enable tracking to start monitoring.")

# Tab 3: Performance
with tab3:
    st.subheader("📈 Betting Performance")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Total Bets</h4>
            <h2>0</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Win Rate</h4>
            <h2>0%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Total ROI</h4>
            <h2>0%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>Avg Edge</h4>
            <h2>0%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Simulated performance chart
    st.markdown("#### 📊 Bankroll Over Time")
    
    # Placeholder chart
    sample_data = pd.DataFrame({
        'Time': pd.date_range(start='2024-01-01', periods=20, freq='H'),
        'Bankroll': [1000 + i * 50 + (i % 3 - 1) * 30 for i in range(20)]
    })
    
    fig = px.line(sample_data, x='Time', y='Bankroll', 
                  title='Simulated Bankroll Growth')
    fig.add_hline(y=1000, line_dash="dash", line_color="white", 
                  annotation_text="Starting Bankroll")
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: History
# Tab 4: History
with tab4:
    st.subheader("🔍 Match History & Analytics")
    
    if st.session_state.probability_history:
        df = pd.DataFrame(st.session_state.probability_history)
        
        # Create tabs within tab for different views
        history_tab1, history_tab2, history_tab3 = st.tabs([
            "📈 Probability Timeline", 
            "⚔️ Kill Progression",
            "💰 Gold Difference"
        ])
        
        with history_tab1:
            # Main probability chart (same as in Tab 1 but larger)
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['game_time'],
                y=df['blue_prob'] * 100,
                name='🔵 Blue Team',
                mode='lines+markers',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=df['game_time'],
                y=df['red_prob'] * 100,
                name='🔴 Red Team',
                mode='lines+markers',
                line=dict(color='#ef4444', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_hline(y=50, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Win Probability Over Time (Full History)",
                xaxis_title="Game Time",
                yaxis_title="Win Probability (%)",
                height=500,
                hovermode='x unified',
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Current Blue Win %",
                    f"{df['blue_prob'].iloc[-1]:.1%}",
                    delta=f"{(df['blue_prob'].iloc[-1] - df['blue_prob'].iloc[0]):.1%}"
                )
            with col2:
                st.metric(
                    "Max Blue Advantage",
                    f"{df['blue_prob'].max():.1%}"
                )
            with col3:
                st.metric(
                    "Min Blue Advantage",
                    f"{df['blue_prob'].min():.1%}"
                )
        
        with history_tab2:
            # Kill progression
            fig_kills = go.Figure()
            
            fig_kills.add_trace(go.Scatter(
                x=df['game_time'],
                y=df['blue_kills'],
                name='🔵 Blue Kills',
                mode='lines+markers',
                line=dict(color='#3b82f6', width=2)
            ))
            
            fig_kills.add_trace(go.Scatter(
                x=df['game_time'],
                y=df['red_kills'],
                name='🔴 Red Kills',
                mode='lines+markers',
                line=dict(color='#ef4444', width=2)
            ))
            
            fig_kills.update_layout(
                title="Kill Count Over Time",
                xaxis_title="Game Time",
                yaxis_title="Kills",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_kills, use_container_width=True)
        
        with history_tab3:
            # Gold difference
            fig_gold = go.Figure()
            
            # Color based on positive/negative
            colors = ['green' if x > 0 else 'red' for x in df['gold_diff']]
            
            fig_gold.add_trace(go.Bar(
                x=df['game_time'],
                y=df['gold_diff'],
                name='Gold Difference',
                marker_color=colors,
                text=df['gold_diff'].apply(lambda x: f"{x:+,}"),
                textposition='outside'
            ))
            
            fig_gold.add_hline(y=0, line_dash="dash", line_color="white")
            
            fig_gold.update_layout(
                title="Gold Difference Over Time (Blue - Red)",
                xaxis_title="Game Time",
                yaxis_title="Gold Difference",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_gold, use_container_width=True)
        
        # Full data table
        st.markdown("---")
        st.markdown("#### 📋 Complete Data Table")
        
        display_df = df[['game_time', 'blue_prob', 'red_prob', 'blue_kills', 'red_kills', 'gold_diff']].copy()
        display_df['blue_prob'] = display_df['blue_prob'].apply(lambda x: f"{x:.1%}")
        display_df['red_prob'] = display_df['red_prob'].apply(lambda x: f"{x:.1%}")
        display_df['gold_diff'] = display_df['gold_diff'].apply(lambda x: f"{x:+,}")
        display_df.columns = ['Game Time', 'Blue Win %', 'Red Win %', 'Blue Kills', 'Red Kills', 'Gold Diff']
        
        st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)
    
    else:
        st.info("📊 No historical data yet. Start tracking to build history.")
        
        # Show example
        st.markdown("#### Example visualization:")
        sample_data = pd.DataFrame({
            'game_time': ['10:00', '12:00', '14:00', '16:00', '18:00', '20:00'],
            'blue_prob': [0.52, 0.55, 0.60, 0.58, 0.65, 0.70],
            'red_prob': [0.48, 0.45, 0.40, 0.42, 0.35, 0.30]
        })
        
        fig = px.line(sample_data, x='game_time', y=['blue_prob', 'red_prob'],
                     title='Sample: Win Probability Evolution',
                     labels={'value': 'Win Probability', 'game_time': 'Game Time'})
        st.plotly_chart(fig, use_container_width=True)
# Auto-refresh logic
if st.session_state.tracking_active:
    with st.spinner("🔄 Fetching latest data..."):
        try:
            # ÉLES HASZNÁLATHOZ: Uncommenteld ezeket
            # from scrapers import scrape_match_stats, scrape_odds
            # from value_betting import ValueBettingEngine
            
            # match_data = scrape_match_stats(match_url)
            # odds_data = scrape_odds(odds_url)
            
            # if match_data and gb_model and rf_model and scaler:
            #     engine = ValueBettingEngine(gb_model, rf_model, scaler, 
            #                                 min_edge=min_edge/100, 
            #                                 min_confidence=min_confidence)
            #     features = engine.calculate_features(match_data)
            #     prob_blue, prob_red = engine.predict_win_probability(features, use_ensemble)
            
            # DEMO MODE: Szimulált adatok
            import random
            current_time = datetime.now()
            game_minute = len(st.session_state.probability_history) + 10
            
            # Szimuláld a valószínűségek változását
            if not st.session_state.probability_history:
                prob_blue = 0.50 + random.uniform(-0.05, 0.05)
            else:
                last_prob = st.session_state.probability_history[-1]['blue_prob']
                prob_blue = last_prob + random.uniform(-0.08, 0.08)
                prob_blue = max(0.1, min(0.9, prob_blue))  # Keep in 10-90% range
            
            prob_red = 1 - prob_blue
            
            # Szimulált match data
            match_data = {
                'timestamp': current_time.isoformat(),
                'game_time': f"{game_minute}:00",
                'blue_team': {
                    'kills': int(10 + game_minute * 0.5),
                    'towers': min(11, int(game_minute / 5)),
                    'gold': int(35000 + game_minute * 1000),
                    'dragons': ['Ocean'] * min(4, int(game_minute / 8)),
                    'barons': 1 if game_minute > 25 else 0,
                    'inhibitors': 1 if game_minute > 28 else 0
                },
                'red_team': {
                    'kills': int(8 + game_minute * 0.4),
                    'towers': min(11, int(game_minute / 6)),
                    'gold': int(32000 + game_minute * 950),
                    'dragons': [] if game_minute < 15 else ['Infernal'],
                    'barons': 0,
                    'inhibitors': 0
                },
                'players': [{'cs': 150 + game_minute * 5} for _ in range(10)]
            }
            
            st.session_state.current_match_data = match_data
            st.session_state.predicted_probs = (prob_blue, prob_red)
            st.session_state.last_update = current_time.isoformat()
            
            # Mentsd el a probability historyt
            st.session_state.probability_history.append({
                'timestamp': current_time,
                'game_time': match_data['game_time'],
                'blue_prob': prob_blue,
                'red_prob': prob_red,
                'blue_kills': match_data['blue_team']['kills'],
                'red_kills': match_data['red_team']['kills'],
                'gold_diff': match_data['blue_team']['gold'] - match_data['red_team']['gold']
            })
            
            # Limit history to last 50 points
            if len(st.session_state.probability_history) > 50:
                st.session_state.probability_history = st.session_state.probability_history[-50:]
            
            # Value bet detection (opcionális demo)
            if prob_blue > 0.60 and random.random() > 0.7:
                new_value_bet = {
                    'timestamp': current_time.isoformat(),
                    'game_time': match_data['game_time'],
                    'team': 'BLUE',
                    'team_name': 'Blue Team',
                    'market_name': 'Match Winner',
                    'odds': 1.45,
                    'predicted_prob': prob_blue,
                    'implied_prob': 1/1.45,
                    'edge': (prob_blue * 1.45 - 1) * 100,
                    'kelly_fraction': 0.12,
                    'confidence': 'MEDIUM'
                }
                
                # Check if not duplicate
                if not any(vb['game_time'] == new_value_bet['game_time'] for vb in st.session_state.value_bets):
                    st.session_state.value_bets.append(new_value_bet)
                    if len(st.session_state.value_bets) > 10:
                        st.session_state.value_bets = st.session_state.value_bets[-10:]
        
        except Exception as e:
            st.error(f"Error during update: {e}")
    
    # Schedule next update
    time.sleep(update_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 Live Value Betting Dashboard | Built with Streamlit</p>
    <p>⚠️ For educational purposes only. Bet responsibly.</p>
</div>
""", unsafe_allow_html=True)
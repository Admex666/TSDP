"""
Match Report Generator - Streamlit Application
Pre-match analysis tool using SofaScore data
"""
import streamlit as st
import sys
import os
from datetime import datetime

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from data_fetcher import DataFetcher
from metrics_calculator import MetricsCalculator
from visualizations import Visualizations
from leagues import LEAGUES

# Page configuration
st.set_page_config(
    page_title="Match Report Generator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #062b5c;
        margin-bottom: 2rem;
    }
    .team-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .home-team {
        background-color: #3498db;
        color: white;
    }
    .away-team {
        background-color: #e74c3c;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #062b5c;
        margin: 1rem 0;
    }
    .stTab {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'selected_match' not in st.session_state:
    st.session_state.selected_match = None

# Sidebar - Match Selection
st.sidebar.markdown("## ⚽ Match Report Generator")
st.sidebar.markdown("---")

# League selection
league_name = st.sidebar.selectbox(
    "Select League",
    options=list(LEAGUES.keys()),
    index=0
)

league_config = LEAGUES[league_name]

# Round selection
round_num = st.sidebar.number_input(
    "Select Round",
    min_value=1,
    max_value=league_config['total_rounds'],
    value=league_config['current_round']
)

# Fetch matches for selected round
with st.spinner("Loading matches..."):
    matches = st.session_state.data_fetcher.get_round_matches(
        league_config['tournament_id'],
        league_config['season_id'],
        round_num
    )

if not matches:
    st.sidebar.error("No matches found for this round")
    st.stop()

# Match selection
match_options = [
    f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}"
    for m in matches
]

selected_match_idx = st.sidebar.selectbox(
    "Select Match",
    options=range(len(match_options)),
    format_func=lambda x: match_options[x]
)

selected_match = matches[selected_match_idx]

# Generate Report Button
if st.sidebar.button("🔍 Generate Report", type="primary", use_container_width=True):
    st.session_state.selected_match = selected_match
    st.rerun()

# Display match info in sidebar
if st.session_state.selected_match:
    match = st.session_state.selected_match
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Selected Match")
    st.sidebar.markdown(f"**{match['homeTeam']['name']}**")
    st.sidebar.markdown("vs")
    st.sidebar.markdown(f"**{match['awayTeam']['name']}**")
    
    # Match time
    timestamp = match.get('startTimestamp', 0)
    if timestamp:
        match_time = datetime.fromtimestamp(timestamp)
        st.sidebar.markdown(f"📅 {match_time.strftime('%Y-%m-%d %H:%M')}")

# Main content
if not st.session_state.selected_match:
    st.markdown('<div class="main-header">⚽ Match Report Generator</div>', unsafe_allow_html=True)
    st.info("👈 Select a match from the sidebar to generate a detailed pre-match report")
    
    st.markdown("### Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Team Analysis")
        st.write("- Playing styles")
        st.write("- Attack & defense metrics")
        st.write("- Shot quality analysis")
    
    with col2:
        st.markdown("#### ⭐ Key Players")
        st.write("- Top scorers")
        st.write("- Playmakers")
        st.write("- Defensive leaders")
    
    with col3:
        st.markdown("#### 🎯 Tactical Insights")
        st.write("- Shot maps")
        st.write("- Set piece analysis")
        st.write("- Match predictions")
    
    st.stop()

# Generate full report
match = st.session_state.selected_match
home_team = match['homeTeam']
away_team = match['awayTeam']
tournament_id = league_config['tournament_id']
season_id = league_config['season_id']

# Header
st.markdown('<div class="main-header">Match Report</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown(f'<div class="team-header home-team">🏠 {home_team["name"]}</div>', unsafe_allow_html=True)

with col2:
    st.markdown("<h2 style='text-align: center; padding-top: 1.5rem;'>VS</h2>", unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="team-header away-team">{away_team["name"]} 🚌</div>', unsafe_allow_html=True)

# Match details
timestamp = match.get('startTimestamp', 0)
if timestamp:
    match_time = datetime.fromtimestamp(timestamp)
    st.markdown(f"<h3 style='text-align: center;'>📅 {match_time.strftime('%A, %B %d, %Y at %H:%M')}</h3>", unsafe_allow_html=True)

st.markdown("---")

# Fetch all data with progress
progress_bar = st.progress(0)
status_text = st.empty()

try:
    # Fetch home team data
    status_text.text("Fetching home team statistics...")
    progress_bar.progress(10)
    home_stats = st.session_state.data_fetcher.get_team_statistics(
        home_team['id'], tournament_id, season_id
    )
    
    status_text.text("Fetching home team form...")
    progress_bar.progress(20)
    home_form = st.session_state.data_fetcher.get_team_form(home_team['id'], limit=10)
    
    status_text.text("Fetching home team players...")
    progress_bar.progress(30)
    home_players = st.session_state.data_fetcher.get_top_players(
        home_team['id'], tournament_id, season_id
    )
    
    status_text.text("Fetching home team shot maps...")
    progress_bar.progress(40)
    home_shotmap = st.session_state.data_fetcher.get_team_shotmaps(
        home_team['id'], tournament_id, season_id, max_matches=10
    )
    
    # Fetch away team data
    status_text.text("Fetching away team statistics...")
    progress_bar.progress(50)
    away_stats = st.session_state.data_fetcher.get_team_statistics(
        away_team['id'], tournament_id, season_id
    )
    
    status_text.text("Fetching away team form...")
    progress_bar.progress(60)
    away_form = st.session_state.data_fetcher.get_team_form(away_team['id'], limit=10)
    
    status_text.text("Fetching away team players...")
    progress_bar.progress(70)
    away_players = st.session_state.data_fetcher.get_top_players(
        away_team['id'], tournament_id, season_id
    )
    
    status_text.text("Fetching away team shot maps...")
    progress_bar.progress(80)
    away_shotmap = st.session_state.data_fetcher.get_team_shotmaps(
        away_team['id'], tournament_id, season_id, max_matches=10
    )
    
    status_text.text("Calculating metrics...")
    progress_bar.progress(90)
    
    # Calculate metrics
    calc = MetricsCalculator()
    
    # Home team metrics
    home_efficiency = calc.calculate_efficiency_metrics(home_stats)
    home_xg = calc.calculate_xg_metrics(home_stats, home_shotmap)
    home_shot_locations = calc.calculate_shot_location_metrics(home_shotmap)
    home_style = calc.calculate_style_indicators(home_stats, home_shotmap)
    home_defensive = calc.calculate_defensive_metrics(home_stats)
    home_form_metrics = calc.calculate_form_metrics(home_form, home_team['id'], tournament_id)
    home_set_pieces = calc.calculate_set_piece_metrics(home_stats)
    
    # Away team metrics
    away_efficiency = calc.calculate_efficiency_metrics(away_stats)
    away_xg = calc.calculate_xg_metrics(away_stats, away_shotmap)
    away_shot_locations = calc.calculate_shot_location_metrics(away_shotmap)
    away_style = calc.calculate_style_indicators(away_stats, away_shotmap)
    away_defensive = calc.calculate_defensive_metrics(away_stats)
    away_form_metrics = calc.calculate_form_metrics(away_form, away_team['id'], tournament_id)
    away_set_pieces = calc.calculate_set_piece_metrics(away_stats)
    
    # Combine metrics for visualization
    home_combined = {**home_efficiency, **home_xg, **home_shot_locations, 
                     **home_style, **home_defensive, **home_set_pieces}
    away_combined = {**away_efficiency, **away_xg, **away_shot_locations,
                     **away_style, **away_defensive, **away_set_pieces}
    
    progress_bar.progress(100)
    status_text.text("Report generated successfully!")
    progress_bar.empty()
    status_text.empty()
    
except Exception as e:
    st.error(f"Error generating report: {str(e)}")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "🎯 Team Analysis", 
    "⚽ Set Pieces",
    "⭐ Key Players",
    "📈 Form & Trends",
    "🔮 Prediction",
    "🗺️ Tactical Analysis"
])

# Tab 1: Overview
with tab1:
    st.header("Match Overview")
    
    # Form display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_team['name']}")
        st.metric("Last 5 Matches", home_form_metrics['form_string'])
        st.metric("Points", home_form_metrics['points'])
        st.metric("Goals For/Against", f"{home_form_metrics['goals_for']}/{home_form_metrics['goals_against']}")
    
    with col2:
        st.subheader(f"🚌 {away_team['name']}")
        st.metric("Last 5 Matches", away_form_metrics['form_string'])
        st.metric("Points", away_form_metrics['points'])
        st.metric("Goals For/Against", f"{away_form_metrics['goals_for']}/{away_form_metrics['goals_against']}")
    
    st.markdown("---")
    
    # Radar chart
    st.subheader("Team Comparison")
    viz = Visualizations()
    radar_fig = viz.create_radar_chart(
        home_combined, away_combined,
        home_team['name'], away_team['name']
    )
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # Comparison table
    st.subheader("Key Statistics")
    comparison_df = viz.create_comparison_table(
        home_combined, away_combined,
        home_team['name'], away_team['name']
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Tab 2: Team Analysis
with tab2:
    st.header("Team Style Profiles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_team['name']}")
        
        st.markdown("#### 🎯 Playing Style")
        st.write(f"**Possession:** {home_style['possession_style']} ({home_style['possession_pct']}%)")
        st.write(f"**Direct Play:** {'Yes' if home_style['direct_play'] else 'No'} ({home_style['long_ball_pct']}% long balls)")
        st.write(f"**Wing Oriented:** {'Yes' if home_style['wing_oriented'] else 'No'} ({home_style['cross_pct']}% crosses)")
        
        st.markdown("#### ⚔️ Attacking")
        st.write(f"**Goals/90:** {home_combined['goals_per_90']}")
        st.write(f"**xG/90:** {home_combined['xg_per_90']}")
        st.write(f"**Shots/90:** {home_combined['shots_per_90']}")
        st.write(f"**Shot Accuracy:** {home_combined['shooting_accuracy']}%")
        st.write(f"**Big Chances/90:** {MetricsCalculator.calculate_per_90(home_stats.get('bigChances', 0), home_stats.get('matches', 1))}")
        
        st.markdown("#### 🎯 Shot Quality")
        st.write(f"**xG/Shot:** {home_combined['xg_per_shot']}")
        st.write(f"**Quality:** {home_combined['shot_quality']}")
        st.write(f"**Finishing:** {home_combined['finishing_performance']} ({home_combined['xg_overperformance']:+.1f}%)")
        
        st.markdown("#### 📍 Shot Locations")
        st.write(f"**Inside Box:** {home_shot_locations['inside_box_pct']}%")
        st.write(f"**Outside Box:** {home_shot_locations['outside_box_pct']}%")
        st.write(f"**6-Yard Box:** {home_shot_locations['six_yard_pct']}%")
        
        st.markdown("#### 🦶 Shot Method")
        st.write(f"**Right Foot:** {home_shot_locations['right_foot_pct']}%")
        st.write(f"**Left Foot:** {home_shot_locations['left_foot_pct']}%")
        st.write(f"**Headers:** {home_shot_locations['header_pct']}%")
        
        st.markdown("#### 🛡️ Defending")
        st.write(f"**Goals Conceded/90:** {home_defensive['goals_conceded_per_90']}")
        st.write(f"**Clean Sheets:** {home_defensive['clean_sheet_pct']}%")
        st.write(f"**Tackles/90:** {home_defensive['tackles_per_90']}")
        st.write(f"**Interceptions/90:** {home_defensive['interceptions_per_90']}")
        st.write(f"**Duels Won:** {home_defensive['duels_won_pct']}%")
        st.write(f"**Aerial Duels Won:** {home_defensive['aerial_duels_won_pct']}%")
    
    with col2:
        st.subheader(f"🚌 {away_team['name']}")
        
        st.markdown("#### 🎯 Playing Style")
        st.write(f"**Possession:** {away_style['possession_style']} ({away_style['possession_pct']}%)")
        st.write(f"**Direct Play:** {'Yes' if away_style['direct_play'] else 'No'} ({away_style['long_ball_pct']}% long balls)")
        st.write(f"**Wing Oriented:** {'Yes' if away_style['wing_oriented'] else 'No'} ({away_style['cross_pct']}% crosses)")
        
        st.markdown("#### ⚔️ Attacking")
        st.write(f"**Goals/90:** {away_combined['goals_per_90']}")
        st.write(f"**xG/90:** {away_combined['xg_per_90']}")
        st.write(f"**Shots/90:** {away_combined['shots_per_90']}")
        st.write(f"**Shot Accuracy:** {away_combined['shooting_accuracy']}%")
        st.write(f"**Big Chances/90:** {MetricsCalculator.calculate_per_90(away_stats.get('bigChances', 0), away_stats.get('matches', 1))}")
        
        st.markdown("#### 🎯 Shot Quality")
        st.write(f"**xG/Shot:** {away_combined['xg_per_shot']}")
        st.write(f"**Quality:** {away_combined['shot_quality']}")
        st.write(f"**Finishing:** {away_combined['finishing_performance']} ({away_combined['xg_overperformance']:+.1f}%)")
        
        st.markdown("#### 📍 Shot Locations")
        st.write(f"**Inside Box:** {away_shot_locations['inside_box_pct']}%")
        st.write(f"**Outside Box:** {away_shot_locations['outside_box_pct']}%")
        st.write(f"**6-Yard Box:** {away_shot_locations['six_yard_pct']}%")
        
        st.markdown("#### 🦶 Shot Method")
        st.write(f"**Right Foot:** {away_shot_locations['right_foot_pct']}%")
        st.write(f"**Left Foot:** {away_shot_locations['left_foot_pct']}%")
        st.write(f"**Headers:** {away_shot_locations['header_pct']}%")
        
        st.markdown("#### 🛡️ Defending")
        st.write(f"**Goals Conceded/90:** {away_defensive['goals_conceded_per_90']}")
        st.write(f"**Clean Sheets:** {away_defensive['clean_sheet_pct']}%")
        st.write(f"**Tackles/90:** {away_defensive['tackles_per_90']}")
        st.write(f"**Interceptions/90:** {away_defensive['interceptions_per_90']}")
        st.write(f"**Duels Won:** {away_defensive['duels_won_pct']}%")
        st.write(f"**Aerial Duels Won:** {away_defensive['aerial_duels_won_pct']}%")

# Tab 3: Set Pieces
with tab3:
    st.header("Set Piece Analysis")
    
    # Set piece comparison chart
    sp_fig = viz.create_set_piece_comparison(
        home_set_pieces, away_set_pieces,
        home_team['name'], away_team['name']
    )
    st.plotly_chart(sp_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_team['name']}")
        
        st.markdown("#### 🚩 Corners")
        st.write(f"**Won/90:** {home_set_pieces['corners_per_90']}")
        st.write(f"**Against/90:** {home_set_pieces['corners_against_per_90']}")
        
        st.markdown("#### 🎯 Free Kicks")
        st.write(f"**Goals:** {home_set_pieces['free_kick_goals']}")
        st.write(f"**Shots:** {home_set_pieces['free_kick_shots']}")
        st.write(f"**Conversion:** {home_set_pieces['fk_conversion']}%")
        
        st.markdown("#### ⚡ Penalties")
        st.write(f"**Scored:** {home_set_pieces['penalty_goals']}")
        st.write(f"**Taken:** {home_set_pieces['penalties_taken']}")
        st.write(f"**Conversion:** {home_set_pieces['penalty_conversion']}%")
        st.write(f"**Conceded:** {home_set_pieces['penalties_conceded']}")
    
    with col2:
        st.subheader(f"🚌 {away_team['name']}")
        
        st.markdown("#### 🚩 Corners")
        st.write(f"**Won/90:** {away_set_pieces['corners_per_90']}")
        st.write(f"**Against/90:** {away_set_pieces['corners_against_per_90']}")
        
        st.markdown("#### 🎯 Free Kicks")
        st.write(f"**Goals:** {away_set_pieces['free_kick_goals']}")
        st.write(f"**Shots:** {away_set_pieces['free_kick_shots']}")
        st.write(f"**Conversion:** {away_set_pieces['fk_conversion']}%")
        
        st.markdown("#### ⚡ Penalties")
        st.write(f"**Scored:** {away_set_pieces['penalty_goals']}")
        st.write(f"**Taken:** {away_set_pieces['penalties_taken']}")
        st.write(f"**Conversion:** {away_set_pieces['penalty_conversion']}%")
        st.write(f"**Conceded:** {away_set_pieces['penalties_conceded']}")

# Tab 4: Key Players
with tab4:
    st.header("Key Players")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_team['name']}")
        
        # Top scorers
        if 'goals' in home_players and home_players['goals']:
            st.markdown("#### ⚽ Top Scorers")
            for i, player in enumerate(home_players['goals'][:3], 1):
                st.write(f"{i}. **{player['player']['name']}** - {player.get('goals', 0)} goals")
        
        # Top assisters
        if 'assists' in home_players and home_players['assists']:
            st.markdown("#### 🎨 Top Assisters")
            for i, player in enumerate(home_players['assists'][:3], 1):
                st.write(f"{i}. **{player['player']['name']}** - {player.get('assists', 0)} assists")
        
        # Best rated
        if 'rating' in home_players and home_players['rating']:
            st.markdown("#### ⭐ Best Rated")
            for i, player in enumerate(home_players['rating'][:3], 1):
                st.write(f"{i}. **{player['player']['name']}** - {player.get('rating', 0):.2f} avg rating")
    
    with col2:
        st.subheader(f"🚌 {away_team['name']}")
        
        # Top scorers
        if 'goals' in away_players and away_players['goals']:
            st.markdown("#### ⚽ Top Scorers")
            for i, player in enumerate(away_players['goals'][:3], 1):
                st.write(f"{i}. **{player['player']['name']}** - {player.get('goals', 0)} goals")
        
        # Top assisters
        if 'assists' in away_players and away_players['assists']:
            st.markdown("#### 🎨 Top Assisters")
            for i, player in enumerate(away_players['assists'][:3], 1):
                st.write(f"{i}. **{player['player']['name']}** - {player.get('assists', 0)} assists")
        
        # Best rated
        if 'rating' in away_players and away_players['rating']:
            st.markdown("#### ⭐ Best Rated")
            for i, player in enumerate(away_players['rating'][:3], 1):
                st.write(f"{i}. **{player['player']['name']}** - {player.get('rating', 0):.2f} avg rating")

# Tab 5: Form & Trends
with tab5:
    st.header("Recent Form Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        form_fig_home = viz.create_form_chart(home_form, home_team['id'], home_team['name'])
        st.plotly_chart(form_fig_home, use_container_width=True)
    
    with col2:
        form_fig_away = viz.create_form_chart(away_form, away_team['id'], away_team['name'])
        st.plotly_chart(form_fig_away, use_container_width=True)

# Tab 6: Prediction
with tab6:
    st.header("Match Insights & Prediction")
    
    st.markdown("### 🎯 Key Battles")
    
    # Determine advantages
    attacking_advantage = home_team['name'] if home_combined['xg_per_90'] > away_combined['xg_per_90'] else away_team['name']
    defensive_advantage = home_team['name'] if home_defensive['goals_conceded_per_90'] < away_defensive['goals_conceded_per_90'] else away_team['name']
    set_piece_advantage = home_team['name'] if home_set_pieces['corners_per_90'] > away_set_pieces['corners_per_90'] else away_team['name']
    
    st.write(f"**1️⃣ Attacking Prowess:** {attacking_advantage} has the edge in attack")
    st.write(f"**2️⃣ Defensive Solidity:** {defensive_advantage} has the stronger defense")
    st.write(f"**3️⃣ Set Pieces:** {set_piece_advantage} more dangerous from set pieces")
    
    st.markdown("### ⚠️ What to Watch")
    
    watch_points = []
    
    if home_combined['xg_per_shot'] > 0.12:
        watch_points.append(f"✓ {home_team['name']}'s high-quality shot selection")
    if away_combined['xg_per_shot'] > 0.12:
        watch_points.append(f"✓ {away_team['name']}'s high-quality shot selection")
    
    if home_set_pieces['penalties_conceded'] > 3:
        watch_points.append(f"✓ {home_team['name']} prone to conceding penalties")
    if away_set_pieces['penalties_conceded'] > 3:
        watch_points.append(f"✓ {away_team['name']} prone to conceding penalties")
    
    for point in watch_points:
        st.write(point)
    
    st.markdown("### 📊 Statistical Prediction")
    
    # Simple prediction based on xG
    home_xg_pred = home_combined['xg_per_90']
    away_xg_pred = away_combined['xg_per_90']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expected Goals", f"{home_team['name']}", f"{home_xg_pred:.2f}")
    
    with col2:
        st.metric("vs", "", "")
    
    with col3:
        st.metric("Expected Goals", f"{away_team['name']}", f"{away_xg_pred:.2f}")
    
    # Win probability (simplified)
    total_xg = home_xg_pred + away_xg_pred
    if total_xg > 0:
        home_win_prob = (home_xg_pred / total_xg) * 100
        away_win_prob = (away_xg_pred / total_xg) * 100
        draw_prob = 100 - home_win_prob - away_win_prob
        
        st.markdown("### 🎲 Win Probability")
        st.write(f"**{home_team['name']}:** {home_win_prob:.1f}%")
        st.write(f"**Draw:** {abs(draw_prob):.1f}%")
        st.write(f"**{away_team['name']}:** {away_win_prob:.1f}%")

# Tab 7: Tactical Analysis
with tab7:
    st.header("Tactical Visualization")
    
    st.subheader("🎯 Shot Maps (Season Aggregate)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {home_team['name']}")
        home_shot_fig = viz.create_shot_map(home_shotmap, home_team['name'], '#3498db')
        st.pyplot(home_shot_fig)
    
    with col2:
        st.markdown(f"#### {away_team['name']}")
        away_shot_fig = viz.create_shot_map(away_shotmap, away_team['name'], '#e74c3c')
        st.pyplot(away_shot_fig)
    
    st.markdown("---")
    st.markdown("### 📊 Shot Map Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{home_team['name']}:**")
        st.write(f"- Shots from outside box: {home_shot_locations['outside_box_pct']}%")
        st.write(f"- Central shots: {home_shot_locations['central_pct']}%")
        st.write(f"- xG per shot: {home_combined['xg_per_shot']}")
    
    with col2:
        st.write(f"**{away_team['name']}:**")
        st.write(f"- Shots from outside box: {away_shot_locations['outside_box_pct']}%")
        st.write(f"- Central shots: {away_shot_locations['central_pct']}%")
        st.write(f"- xG per shot: {away_combined['xg_per_shot']}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Data source: SofaScore | Generated with ⚽ Match Report Generator</p>", unsafe_allow_html=True)

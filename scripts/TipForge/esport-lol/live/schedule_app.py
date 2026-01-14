import streamlit as st
import pandas as pd
from datetime import datetime
from riot_api import RiotEsportsAPI
import pytz

# Page config
st.set_page_config(page_title="📅 LoL Esports Schedule", page_icon="📅", layout="wide")

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .match-card {
        background-color: #1e2128;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #2d3139;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .match-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        border-color: #00cfb3;
    }
    .league-badge {
        background-color: #00cfb3;
        color: #000;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .time-badge {
        color: #8b949e;
        font-size: 0.9rem;
    }
    .team-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
    }
    .vs-text {
        font-size: 1rem;
        color: #8b949e;
        font-weight: bold;
        margin: 0 10px;
    }
    .team-logo {
        width: 40px;
        height: 40px;
        object-fit: contain;
        margin-right: 15px;
    }
    .state-badge {
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        font-weight: bold;
    }
    .state-unstarted { background-color: #2d3139; color: #8b949e; }
    .state-inprogress { background-color: #f85149; color: white; animation: blink 1.5s infinite; }
    .state-completed { background-color: #238636; color: white; }
    
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.title("📅 UPCOMING LoL MATCHES")
st.markdown("Real-time schedule from the official Riot Esports API")

# Initialize API
api = RiotEsportsAPI()

@st.cache_data(ttl=300)
def load_schedule():
    return api.get_schedule()

with st.spinner("Fetching matches..."):
    events = load_schedule()

if not events:
    st.error("No matches found or API error.")
else:
    # Sidebar filters
    st.sidebar.title("🔍 Filters")
    leagues = sorted(list(set([e['league']['name'] for e in events])))
    selected_league = st.sidebar.multiselect("Select Leagues", leagues, default=[])
    
    show_completed = st.sidebar.checkbox("Show Completed", value=False)
    
    # Filter logistics
    filtered_events = events
    if selected_league:
        filtered_events = [e for e in filtered_events if e['league']['name'] in selected_league]
    
    if not show_completed:
        filtered_events = [e for e in filtered_events if e['state'] != 'completed']

    st.sidebar.markdown("---")
    st.sidebar.info(f"Showing {len(filtered_events)} matches")

    if not filtered_events:
        st.warning("No matches match your filters.")
    else:
        # Sort by time
        filtered_events.sort(key=lambda x: x['startTime'])
        
        # Display in rows
        for event in filtered_events:
            start_time = datetime.fromisoformat(event['startTime'].replace('Z', '+00:00'))
            local_time = start_time.astimezone(pytz.timezone('Europe/Budapest'))
            time_str = local_time.strftime("%Y-%m-%d %H:%M")
            
            league = event['league']['name']
            match_data = event.get('match', {})
            teams = match_data.get('teams', [])
            state = event['state']
            
            state_class = f"state-{state.lower()}"
            
            with st.container():
                cols = st.columns([1])
                with cols[0]:
                    # Build HTML for card
                    t1_name = teams[0]['name'] if len(teams) > 0 else "TBD"
                    t2_name = teams[1]['name'] if len(teams) > 1 else "TBD"
                    t1_logo = teams[0].get('image', 'https://via.placeholder.com/40') if len(teams) > 0 else 'https://via.placeholder.com/40'
                    t2_logo = teams[1].get('image', 'https://via.placeholder.com/40') if len(teams) > 1 else 'https://via.placeholder.com/40'
                    
                    st.markdown(f"""
                    <div class="match-card">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div>
                                <span class="league-badge">{league}</span>
                                <span class="time-badge"> | {time_str}</span>
                            </div>
                            <span class="state-badge {state_class}">{state}</span>
                        </div>
                        <div style="display: flex; align-items: center; margin-top: 15px;">
                            <img src="{t1_logo}" class="team-logo">
                            <span class="team-name">{t1_name}</span>
                            <span class="vs-text">VS</span>
                            <span class="team-name">{t2_name}</span>
                            <img src="{t2_logo}" class="team-logo" style="margin-left: 15px; margin-right: 0;">
                        </div>
                        <div style="margin-top: 15px; display: flex; gap: 10px;">
                            <p style="color: #8b949e; margin: 0;">Match ID: {match_data.get('id', 'N/A')}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add Predict Button as a real Streamlit button for functionality
                    btn_id = match_data.get('id') or f"evt_{filtered_events.index(event)}"
                    if state == 'inProgress':
                        if st.button(f"🚀 Open Live Predictor for {t1_name} vs {t2_name}", key=f"btn_{btn_id}"):
                            # In a real app we could redirect or update session state
                            st.info(f"Opening predictor for Match ID: {btn_id}")
                    elif state == 'unstarted':
                         st.button(f"📊 View Team Stats", key=f"btn_stats_{btn_id}", disabled=True)

st.markdown("---")
st.caption("Powered by TipForge & Riot Esports API")

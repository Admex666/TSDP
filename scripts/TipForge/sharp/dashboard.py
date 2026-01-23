import streamlit as st
import pandas as pd
import logging
from pinnacle_scraper import PinnacleScraper
from tippmix_scraper import TippmixScraper
from engine import ValueBetEngine
import main  # Import config

# Configure logging to show in console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="TipForge Sharp", page_icon="💰", layout="wide")

st.title("💰 Sharp Value Bet Finder")
st.markdown("Scrapes Pinnacle (Sharp) and Tippmix (Rec) to find +EV opportunities.")

# Sidebar Config
st.sidebar.header("Configuration")
min_ev = st.sidebar.slider("Minimum EV (%)", 0.0, 10.0, 1.0, 0.1) / 100
match_threshold = st.sidebar.slider("Fuzzy Match Threshold", 60, 100, 80)

if st.button("Start Scan", type="primary"):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    pinnacle_urls = main.PINNACLE_URLS
    tippmix_urls = main.TIPPMIX_URLS
    total_steps = len(pinnacle_urls) + len(tippmix_urls) + 1 # +1 for processing
    current_step = 0
    
    # --- 1. PINNACLE ---
    p_scraper = PinnacleScraper(headless=True)
    p_matches = []
    
    for url in pinnacle_urls:
        status_text.text(f"Scraping Pinnacle: {url.split('/esports/')[-1] if 'esports' in url else 'Tennis'}...")
        try:
            matches = p_scraper.scrape_matches(url)
            p_matches.extend(matches)
        except Exception as e:
            st.error(f"Error scraping {url}: {e}")
            
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
    st.success(f"Retrieved {len(p_matches)} matches from Pinnacle")
    
    # --- 2. TIPPMIX ---
    t_scraper = TippmixScraper(headless=True)
    t_matches = []
    
    for url in tippmix_urls:
        sport_name = "Tennis" if "tenisz" in url else url.split('/')[-3]
        status_text.text(f"Scraping Tippmix: {sport_name}...")
        try:
            matches = t_scraper.scrape_matches(url)
            t_matches.extend(matches)
        except Exception as e:
            st.error(f"Error scraping {url}: {e}")
            
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
    st.success(f"Retrieved {len(t_matches)} matches from Tippmix")
    
    # --- 3. PROCESSING ---
    status_text.text("Processing matches and calculating EV...")
    engine = ValueBetEngine(min_ev=0.0, match_threshold=match_threshold) # Get all matches first
    all_bets = engine.find_value_bets(p_matches, t_matches)
    
    progress_bar.progress(1.0)
    status_text.text("Done!")
    
    # --- 4. DISPLAY ---
    if all_bets:
        df = pd.DataFrame(all_bets)
        
        # Format Data
        df['ev_pct'] = (df['ev'] * 100).round(2)
        df['tippmix_odds'] = df['tippmix_odds'].round(3)
        df['pinnacle_no_vig'] = df['pinnacle_no_vig'].round(3)
        df['fair_prob'] = (df['fair_prob'] * 100).round(1).astype(str) + '%'
        
        # Filter for Value Bets tab
        value_bets_df = df[df['ev'] >= min_ev].copy()
        
        # TABS
        tab1, tab2 = st.tabs([f"🔥 Value Bets ({len(value_bets_df)})", f"📋 All Matches ({len(df)})"])
        
        with tab1:
            if not value_bets_df.empty:
                st.dataframe(
                    value_bets_df[['match', 'outcome', 'tippmix_odds', 'pinnacle_no_vig', 'ev_pct', 'fair_prob']].style.applymap(
                        lambda x: 'background-color: #d4edda; color: #155724' if x > 0 else '', subset=['ev_pct']
                    ),
                    use_container_width=True
                )
            else:
                st.info("No value bets found with current settings.")
                
        with tab2:
            st.dataframe(
                df[['match', 'outcome', 'tippmix_odds', 'pinnacle_no_vig', 'ev_pct', 'fair_prob']].style.applymap(
                    lambda x: 'background-color: #f8d7da; color: #721c24' if x < 0 else 'background-color: #d4edda; color: #155724', subset=['ev_pct']
                ),
                use_container_width=True
            )
            
    else:
        st.warning("No matches paired. Check debug logs if this persists.")

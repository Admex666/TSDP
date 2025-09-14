# mlb_dashboard.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from telegram import send_to_telegram
import streamlit as st
import pandas as pd
from mlb_main import main, format_telegram_message_group, format_telegram_message_owner, filter_mlb_predictions
from mlb_sheets_integration import MLBSheetsIntegration
from datetime import datetime

def run_mlb_app():
    st.set_page_config(page_title="MLB Predictor", layout="wide")
    
    st.title("⚾ MLB Match Predictor")
    st.write("Predikciók és érték fogadások elemzése MLB mérkőzésekre")
    
    # Initialize Google Sheets integration
    if 'sheets_integration' not in st.session_state:
        st.session_state.sheets_integration = MLBSheetsIntegration()
        if st.session_state.sheets_integration.client:
            st.session_state.sheets_integration.ensure_worksheets()
    
    # Sidebar for Google Sheets stats
    with st.sidebar:
        st.header("📊 Fogadási Statisztikák")
        
        if st.session_state.sheets_integration.client:
            stats = st.session_state.sheets_integration.get_stats()
            if stats:
                st.metric("Total Bets", stats.get('Total Bets', 0))
                st.metric("Win Rate", stats.get('Win Rate (%)', '0%'))
                st.metric("Total Profit", stats.get('Total Profit', '$0'))
                st.metric("ROI", stats.get('ROI (%)', '0%'))
            
            # Pending bets
            st.subheader("⏳ Függő Fogadások")
            pending_bets = st.session_state.sheets_integration.get_pending_bets()
            if pending_bets:
                for bet in pending_bets[-5:]:  # Show last 5
                    st.write(f"**{bet['Home_Team']} vs {bet['Away_Team']}**")
                    st.write(f"Bet: {bet['Bet_Type']} @ {bet['Odds']}")
                    st.write("---")
            else:
                st.write("Nincsenek függő fogadások")
        else:
            st.warning("Google Sheets kapcsolat nincs beállítva")
    
    # Main content
    if st.button("Predikciók frissítése"):
        with st.spinner("MLB adatok betöltése és predikciók készítése..."):
            predictions = main()
            st.session_state.mlb_predictions = predictions
            st.success(f"{len(predictions)} predikció elkészült!")
    
    if 'mlb_predictions' not in st.session_state:
        st.info("Kattints a 'Predikciók frissítése' gombra az induláshoz")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        min_value = st.slider("Minimum érték fogadások", 1, 2, 1)
    
    with col2:
        show_explanations = st.checkbox("Magyarázatok mutatása", value=True)
    
    # Filter predictions
    filtered = filter_mlb_predictions(st.session_state.mlb_predictions, min_value)
    
    # Selection tracking
    selected_matches = []
    
    for i, pred in enumerate(filtered):
        col1, col2 = st.columns([0.1, 0.9])
        
        with col1:
            selected = st.checkbox("Kijelöl", key=f"mlb_select_{i}", label_visibility="collapsed")
            if selected:
                selected_matches.append(i)
        
        with col2:
            with st.expander(f"⚾ {pred['home_team']} vs {pred['away_team']} - {pred['date']}", expanded=True):
                
                # Game info
                st.markdown(f"**📍 Venue:** {pred['venue']}")
                st.markdown(f"**🎯 Odds:** Home: {pred['odds']['Home_odds']:.2f} | Away: {pred['odds']['Away_odds']:.2f}")
                
                # Value betting indicators
                value_count = sum(1 for v in pred['value_bets'].values() if v)
                st.markdown(f"**💰 Érték fogadások:** {value_count}")
                
                # Predictions display
                col_home, col_away = st.columns(2)
                
                with col_home:
                    st.markdown(f"**🏠 {pred['home_team']}**")
                    st.progress(pred['predictions']['home_prob'])
                    st.markdown(f"**{pred['predictions']['home_prob']*100:.1f}%**")
                    st.markdown(f"*Fair Odds: {1/pred['predictions']['home_prob']:.2f}* {'✅' if pred['value_bets']['home_value'] else '❌'}")
                
                with col_away:
                    st.markdown(f"**✈️ {pred['away_team']}**")
                    st.progress(pred['predictions']['away_prob'])
                    st.markdown(f"**{pred['predictions']['away_prob']*100:.1f}%**")
                    st.markdown(f"*Fair Odds: {1/pred['predictions']['away_prob']:.2f}* {'✅' if pred['value_bets']['away_value'] else '❌'}")
                
                # Team stats
                if show_explanations:
                    st.markdown("---")
                    st.subheader("📊 Csapat Statisztikák")
                    
                    col_home_stats, col_away_stats = st.columns(2)
                    
                    with col_home_stats:
                        st.markdown(f"**{pred['home_team']} (Home)**")
                        st.write(f"Batting Average: {pred['home_stats']['avg']:.3f}")
                        st.write(f"Slugging: {pred['home_stats']['slg']:.3f}")
                        st.write(f"On-Base %: {pred['home_stats']['obp']:.3f}")
                    
                    with col_away_stats:
                        st.markdown(f"**{pred['away_team']} (Away)**")
                        st.write(f"Batting Average: {pred['away_stats']['avg']:.3f}")
                        st.write(f"Slugging: {pred['away_stats']['slg']:.3f}")
                        st.write(f"On-Base %: {pred['away_stats']['obp']:.3f}")
                
                # Betting section
                st.markdown("---")
                st.subheader("🎲 Fogadás Rögzítése")
                
                col_bet_type, col_odds_select, col_stake = st.columns(3)
                
                with col_bet_type:
                    bet_options = []
                    if pred['value_bets']['home_value']:
                        bet_options.append(f"Home ({pred['home_team']})")
                    if pred['value_bets']['away_value']:
                        bet_options.append(f"Away ({pred['away_team']})")
                    
                    if bet_options:
                        bet_selection = st.selectbox("Fogadás típusa", bet_options, key=f"bet_type_{i}")
                    else:
                        st.write("Nincs érték fogadás")
                        bet_selection = None
                
                with col_odds_select:
                    if bet_selection:
                        if "Home" in bet_selection:
                            selected_odds = pred['odds']['Home_odds']
                        else:
                            selected_odds = pred['odds']['Away_odds']
                        st.number_input("Odds", value=selected_odds, key=f"odds_{i}", disabled=True)
                
                with col_stake:
                    stake = st.number_input("Tét ($)", min_value=1.0, value=10.0, step=1.0, key=f"stake_{i}")
                
                # Add bet button
                if bet_selection and st.button(f"Fogadás rögzítése", key=f"add_bet_{i}"):
                    if st.session_state.sheets_integration.client:
                        bet_type = "Home" if "Home" in bet_selection else "Away"
                        success = st.session_state.sheets_integration.add_bet(
                            pred, bet_type, selected_odds, stake, pred['predictions']
                        )
                        if success:
                            st.success("Fogadás rögzítve!")
                        else:
                            st.error("Hiba a rögzítés során")
                    else:
                        st.warning("Google Sheets kapcsolat szükséges")
                
                # Telegram sending buttons
                col_group, col_owner = st.columns(2)
                
                with col_group:
                    if st.button("📤 Küldés Felhasználóknak", key=f"group_{i}"):
                        message = format_telegram_message_group(
                            pred['home_team'], pred['away_team'], 
                            pred['predictions'], pred['odds'], pred['value_bets']
                        )
                        if message:
                            send_to_telegram(message, to="group", topic_id='6')
                            st.success("Elküldve felhasználóknak!")
                        else:
                            st.warning("Nincs érték fogadás!")
                
                with col_owner:
                    if st.button("📝 Küldés Magamnak", key=f"owner_{i}"):
                        message = format_telegram_message_owner(
                            pred['home_team'], pred['away_team'], 
                            pred['predictions'], pred['odds'], pred['value_bets'],
                            pred['explanation']
                        )
                        send_to_telegram(message, to="owner")
                        st.success("Elküldve!")
    
    # Bulk operations
    if selected_matches:
        st.markdown("---")
        st.subheader("📤 Tömeges Műveletek")
        
        col_bulk_group, col_bulk_owner = st.columns(2)
        
        with col_bulk_group:
            if st.button("📤 Kijelöltek → Felhasználók", key="send_selected_group"):
                sent_count = 0
                for match_idx in selected_matches:
                    pred = filtered[match_idx]
                    message = format_telegram_message_group(
                        pred['home_team'], pred['away_team'], 
                        pred['predictions'], pred['odds'], pred['value_bets']
                    )
                    if message:
                        send_to_telegram(message, to="group", topic_id='6')
                        sent_count += 1
                st.success(f"{sent_count} mérkőzés elküldve felhasználóknak!")
        
        with col_bulk_owner:
            if st.button("📝 Kijelöltek → Magam", key="send_selected_owner"):
                for match_idx in selected_matches:
                    pred = filtered[match_idx]
                    message = format_telegram_message_owner(
                        pred['home_team'], pred['away_team'], 
                        pred['predictions'], pred['odds'], pred['value_bets'],
                        pred['explanation']
                    )
                    send_to_telegram(message, to="owner")
                st.success(f"{len(selected_matches)} mérkőzés elküldve!")
    
    # Manual result update section
    st.markdown("---")
    st.subheader("🏆 Eredmények Frissítése")
    
    if st.session_state.sheets_integration.client:
        col_game_id, col_winner, col_update = st.columns(3)
        
        with col_game_id:
            game_id_input = st.text_input("Game ID")
        
        with col_winner:
            winner_input = st.selectbox("Győztes", ["Home", "Away"])
        
        with col_update:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("Eredmény Frissítése"):
                if game_id_input:
                    success = st.session_state.sheets_integration.update_bet_result(
                        game_id_input, winner_input
                    )
                    if success:
                        st.success("Eredmény frissítve!")
                        st.rerun()
                    else:
                        st.error("Hiba történt")
                else:
                    st.warning("Game ID szükséges")

if __name__ == "__main__":
    run_mlb_app()
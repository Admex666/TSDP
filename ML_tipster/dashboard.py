# streamlit_app.py
import streamlit as st
from main import main, format_telegram_message, filter_predictions
from telegram import send_to_telegram
from config import LEAGUES

def run_app():
    st.set_page_config(page_title="Football Predictor", layout="wide")
    
    st.title("⚽ Football Match Predictor")
    st.write("Predikciók és érték fogadások elemzése")
    
    if st.button("Predikciók frissítése"):
        with st.spinner("Adatok betöltése és predikciók készítése..."):
            predictions = main()
            st.session_state.predictions = predictions
            st.success("Predikciók elkészültek!")
    
    if 'predictions' not in st.session_state:
        st.info("Kattints a 'Predikciók frissítése' gombra az induláshoz")
        return
    
    # Szűrési opciók
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_value = st.slider("Minimum érték fogadások", 1, 5, 1)
    
    with col2:
        selected_league = st.selectbox("Liga", ["Összes"] + list(LEAGUES.keys()))
    
    with col3:
        st.write("")  # Üres hely
        send_all = st.button("Összes kijelölt küldése")
    
    # Szűrés
    filtered = filter_predictions(st.session_state.predictions, min_value)
    if selected_league != "Összes":
        filtered = [p for p in filtered if p['league'] == selected_league]
    
    # Kijelölés
    selected_matches = []
    
    for i, pred in enumerate(filtered):
        col1, col2 = st.columns([0.1, 0.9])
        
        with col1:
            selected = st.checkbox("", key=f"select_{i}")
            if selected:
                selected_matches.append(i)
        
        with col2:
            with st.expander(f"{pred['home_team']} vs {pred['away_team']} ({pred['league']})"):
                for model_name, prob in pred['probs'].items():
                    if model_name == 'LogisticRegression':
                        continue
                    
                    value_marks = pred['value_bets'][model_name]
                    st.write(f"**{model_name}:**")
                    cols = st.columns(3)
                    cols[0].write(f"🏠 {prob[0]*100:.1f}%")
                    cols[1].write(f"⚖️ {prob[1]*100:.1f}%")
                    cols[2].write(f"✈️ {prob[2]*100:.1f}%")
                    
                    cols = st.columns(3)
                    cols[0].write(f"({1/prob[0]:.2f}) {'✅' if value_marks['home'] else ''}")
                    cols[1].write(f"({1/prob[1]:.2f}) {'✅' if value_marks['draw'] else ''}")
                    cols[2].write(f"({1/prob[2]:.2f}) {'✅' if value_marks['away'] else ''}")
                
                st.write(f"**Tippmix odds:** {pred['odds']['home']:.2f} | {pred['odds']['draw']:.2f} | {pred['odds']['away']:.2f}")
                
                if st.button("Küldés Telegramra", key=f"single_{i}"):
                    message = format_telegram_message(
                        pred['home_team'], 
                        pred['away_team'], 
                        pred['probs'], 
                        pred['odds'], 
                        pred['value_bets']
                    )
                    send_to_telegram(message, to="owner")
                    st.success("Elküldve!")
    
    # Tömeges küldés
    if send_all and selected_matches:
        for match_idx in selected_matches:
            pred = filtered[match_idx]
            message = format_telegram_message(
                pred['home_team'], 
                pred['away_team'], 
                pred['probs'], 
                pred['odds'], 
                pred['value_bets']
            )
            send_to_telegram(message, to="owner")
        
        st.success(f"{len(selected_matches)} mérkőzés elküldve!")

if __name__ == "__main__":
    run_app()
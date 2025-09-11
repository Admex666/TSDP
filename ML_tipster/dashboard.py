# dashboard.py - BŐVÍTETT VERZIÓ MAGYARÁZATTAL
import streamlit as st
from main import main, format_telegram_message_group, format_telegram_message_owner, filter_predictions
from telegram import send_to_telegram
from config import LEAGUES
import pandas as pd
import numpy as np

def format_detailed_explanation(home_team, away_team, impacts, prediction, odds):
    """Részletes magyarázat generálása"""
    
    message = f"🔍 {home_team} vs {away_team} - PREDIKCIÓ MAGYARÁZAT\n\n"
    message += f"🏠 Otthoni győzelem: {prediction[0]*100:.1f}% (odds: {1/prediction[0]:.2f})\n"
    message += f"⚖️ Döntetlen: {prediction[1]*100:.1f}% (odds: {1/prediction[1]:.2f})\n" 
    message += f"✈️ Vendég győzelem: {prediction[2]*100:.1f}% (odds: {1/prediction[2]:.2f})\n\n"
    
    message += "🎯 TIPPMIX ODDS:\n"
    message += f"H: {odds['home']:.2f} | D: {odds['draw']:.2f} | A: {odds['away']:.2f}\n\n"
    
    message += "📊 PREDIKCIÓT BEFOLYÁSOLÓ TÉNYEZŐK:\n"
    message += "-" * 50 + "\n"
    
    for i, impact in enumerate(impacts[:7]):  # Top 7 tényező
        sign = "+" if impact['impact'] > 0 else ""
        
        # Értelmezhető leírások
        feature_descriptions = {
            'Home_Points_Last_5': f"Otthon pontátlag ({impact['original_value']:.2f})",
            'Away_Points_Last_5': f"Vendég pontátlag ({impact['original_value']:.2f})",
            'Home_Goals_For_Last_5': f"Otthon lőtt gólátlag ({impact['original_value']:.2f})",
            'Away_Goals_For_Last_5': f"Vendég lőtt gólátlag ({impact['original_value']:.2f})",
            'Home_Goals_Against_Last_5': f"Otthon kapott gólátlag ({impact['original_value']:.2f})",
            'Away_Goals_Against_Last_5': f"Vendég kapott gólátlag ({impact['original_value']:.2f})",
            'Home_Implied_Prob': f"Otthon odds alapú esély ({impact['original_value']:.2%})",
            'Away_Implied_Prob': f"Vendég odds alapú esély ({impact['original_value']:.2%})",
            'Draw_Implied_Prob': f"Döntetlen odds alapú esély ({impact['original_value']:.2%})"
        }
        
        feat_desc = feature_descriptions.get(impact['feature'], impact['feature'])
        message += f"{i+1}. {feat_desc} → {sign}{impact['impact']:+.1f}%\n"
    
    return message

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
        show_explanations = st.checkbox("Magyarázatok mutatása", value=True)
    
    # Szűrés
    filtered = filter_predictions(st.session_state.predictions, min_value)
    if selected_league != "Összes":
        filtered = [p for p in filtered if p['league'] == selected_league]
    
    # Kijelölés
    selected_matches = []
    
    for i, pred in enumerate(filtered):
        col1, col2 = st.columns([0.1, 0.9])
        
        with col1:
            selected = st.checkbox("Kijelöl", key=f"select_{i}", label_visibility="collapsed")
            if selected:
                selected_matches.append(i)
        
        with col2:
            with st.expander(f"{pred['home_team']} vs {pred['away_team']} ({pred['league']})", expanded=True):
                
                # Odds és value badges
                col_odds, col_value = st.columns(2)
                
                with col_odds:
                    st.markdown(f"**🎯 Odds:** {pred['odds']['home']:.2f} | {pred['odds']['draw']:.2f} | {pred['odds']['away']:.2f}")
                
                with col_value:
                    value_count = sum(1 for model_bets in pred['value_bets'].values() 
                                    for bet in model_bets.values() if bet)
                    st.markdown(f"**💰 Érték fogadások:** {value_count}")
                
                # Model predikciók
                for model_name, prob in pred['probs'].items():
                    if model_name == 'LogisticRegression':
                        continue
                    
                    value_marks = pred['value_bets'][model_name]
                    
                    st.subheader(f"{model_name}")
                    
                    # Progress bar-ok a valószínűségekhez
                    col_h, col_d, col_a = st.columns(3)
                    
                    with col_h:
                        st.markdown(f"**🏠 Otthon**")
                        st.progress(prob[0])
                        st.markdown(f"**{prob[0]*100:.1f}%**")
                        st.markdown(f"*({1/prob[0]:.2f})* {'✅' if value_marks['home'] else '❌'}")
                    
                    with col_d:
                        st.markdown(f"**⚖️ Döntetlen**")
                        st.progress(prob[1])
                        st.markdown(f"**{prob[1]*100:.1f}%**")
                        st.markdown(f"*({1/prob[1]:.2f})* {'✅' if value_marks['draw'] else '❌'}")
                    
                    with col_a:
                        st.markdown(f"**✈️ Vendég**")
                        st.progress(prob[2])
                        st.markdown(f"**{prob[2]*100:.1f}%**")
                        st.markdown(f"*({1/prob[2]:.2f})* {'✅' if value_marks['away'] else '❌'}")
                
                # Részletes magyarázat
                if show_explanations and 'explanations' in pred and pred['explanations']:
                    st.markdown("---")
                    st.subheader("📊 Predikció Magyarázata")
                    
                    impacts = pred['explanations'].get('GradientBoosting', [])
                    if impacts:
                        explanation_df = pd.DataFrame(impacts[:10])  # Top 10
                        
                        # Formázás
                        explanation_df['impact_formatted'] = explanation_df['impact'].apply(lambda x: f"{x:+.1f}%")
                        explanation_df['value_formatted'] = explanation_df['original_value'].round(2)
                        
                        # Megjelenítés
                        st.dataframe(
                            explanation_df[['feature', 'value_formatted', 'impact_formatted']],
                            column_config={
                                "feature": "Tényező",
                                "value_formatted": "Érték", 
                                "impact_formatted": "Hatás"
                            },
                            hide_index=True
                        )
                        
                        # Értelmezés
                        st.markdown("**🎯 Értelmezés:**")
                        for _, row in explanation_df.head(3).iterrows():
                            if 'Home' in row['feature'] and row['impact'] > 0:
                                st.write(f"✅ {row['feature']}: erős otthon forma")
                            elif 'Away' in row['feature'] and row['impact'] < 0:
                                st.write(f"✅ {row['feature']}: gyenge vendég forma")
                    
                # Egyedüli küldés gombjai helyett:
                col_group, col_owner = st.columns(2)

                with col_group:
                    if st.button("📤 Küldés Felhasználóknak", key=f"group_{i}"):
                        message = format_telegram_message_group(
                            pred['home_team'], 
                            pred['away_team'], 
                            pred['probs'], 
                            pred['odds'], 
                            pred['value_bets']
                        )
                        if message:
                            send_to_telegram(message, to="group", topic_id='12')
                            st.success("Elküldve felhasználóknak!")
                        else:
                            st.warning("Nincs érték fogadás!")

                with col_owner:
                    if st.button("🔍 Küldés Magamnak", key=f"owner_{i}"):
                        message = format_telegram_message_owner(
                            pred['home_team'], 
                            pred['away_team'], 
                            pred['probs'], 
                            pred['odds'], 
                            pred['value_bets'],
                            pred.get('explanations')
                        )
                        send_to_telegram(message, to="owner")
                        st.success("Elküldve!")
    
    # Tömeges küldés
    if selected_matches:
        col_bulk_group, col_bulk_owner = st.columns(2)
        
        with col_bulk_group:
            if st.button("📤 Kijelöltek → Felhasználók", key="send_selected_group"):
                sent_count = 0
                for match_idx in selected_matches:
                    pred = filtered[match_idx]
                    message = format_telegram_message_group(
                        pred['home_team'], pred['away_team'], pred['probs'], 
                        pred['odds'], pred['value_bets']
                    )
                    if message:
                        send_to_telegram(message, to="group", topic_id='12')
                        sent_count += 1
                st.success(f"{sent_count} mérkőzés elküldve felhasználóknak!")
        
        with col_bulk_owner:
            if st.button("🔍 Kijelöltek → Magam", key="send_selected_owner"):
                for match_idx in selected_matches:
                    pred = filtered[match_idx]
                    message = format_telegram_message_owner(
                        pred['home_team'], pred['away_team'], pred['probs'], 
                        pred['odds'], pred['value_bets'], pred.get('explanations')
                    )
                    send_to_telegram(message, to="owner")
                st.success(f"{len(selected_matches)} mérkőzés elküldve!")

if __name__ == "__main__":
    run_app()
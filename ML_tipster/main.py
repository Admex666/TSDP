# main.py
import pandas as pd
import streamlit as st
from data_loader import load_models, load_league_data, load_fuzz_data
from stats_calculator import get_last5_stats
from predictor import MatchPredictor
from tippmix_api import get_tippmix_data
from telegram import send_to_telegram
from config import LEAGUES

def format_telegram_message(home, away, probs, odds, value_bets):
    """Telegram üzenet formázása"""
    message = f"⚽ {home} vs {away}\n\n"
    
    for model_name, prob in probs.items():
        if model_name == 'LogisticRegression':
            continue
        else:
            message += f"{model_name}:\n"
            message += f"🏠 {prob[0]*100:.1f}% ({1/prob[0]:.2f}) {'✅' if value_bets[model_name]['home'] else ''}\n"
            message += f"⚖️ {prob[1]*100:.1f}% ({1/prob[1]:.2f}) {'✅' if value_bets[model_name]['draw'] else ''}\n"
            message += f"✈️ {prob[2]*100:.1f}% ({1/prob[2]:.2f}) {'✅' if value_bets[model_name]['away'] else ''}\n\n"
        
    message += f"Tippmix odds (1x2): {odds['home']:.2f} | {odds['draw']:.2f} | {odds['away']:.2f}"
    return message

def main():
    # Adatok betöltése
    models, scaler, feature_columns, loaded = load_models()
    if not loaded:
        print("Modellek betöltése sikertelen")
        return []  # Visszaadunk egy üres listát ahelyett, hogy None-t adnánk
    
    fuzz_data = load_fuzz_data()
    if fuzz_data is None:
        print("Fuzz adatok betöltése sikertelen")
        return []  # Visszaadunk egy üres listát
    
    predictor = MatchPredictor(models, scaler, feature_columns)
    
    # Liga végigiterálása
    all_predictions = []
    
    for league_name, league_config in LEAGUES.items():
        print(f"Processing {league_name} league...")
        
        # Ligaadatok betöltése
        df_league = load_league_data(league_config['season'], league_config['league_code'])
        if df_league is None:
            continue
        
        # Tippmix adatok
        tippmix_data = get_tippmix_data(10)
        if tippmix_data is None:
            continue
        
        # Mérkőzések feldolgozása
        for _, match in tippmix_data.iterrows():
            try:
                home_team = match['Home']
                away_team = match['Away']
                
                # Csapatnév mapping
                home_matches = fuzz_data[fuzz_data['Team_tippmix'] == home_team]
                away_matches = fuzz_data[fuzz_data['Team_tippmix'] == away_team]
                
                if len(home_matches) == 0 or len(away_matches) == 0:
                    continue
                
                home_fd = home_matches['Team_fdcouk'].iloc[0]
                away_fd = away_matches['Team_fdcouk'].iloc[0]
                
                # Statisztikák számítása
                home_points, home_gf, home_ga, home_days = get_last5_stats(home_fd, df_league)
                away_points, away_gf, away_ga, away_days = get_last5_stats(away_fd, df_league)
                
                home_stats = {'points': home_points, 'goals_for': home_gf, 'goals_against': home_ga, 'days_since': home_days}
                away_stats = {'points': away_points, 'goals_for': away_gf, 'goals_against': away_ga, 'days_since': away_days}
                
                odds = {'home': match['H_odds'], 'draw': match['D_odds'], 'away': match['A_odds']}
                
                # Predikció
                probs = predictor.predict(home_stats, away_stats, odds)
                value_bets = predictor.analyze_value(probs, odds)
                
                # Eredmények mentése
                prediction_data = {
                    'league': league_name,
                    'home_team': home_team,
                    'away_team': away_team,
                    'probs': probs,
                    'odds': odds,
                    'value_bets': value_bets
                }
                all_predictions.append(prediction_data)
                
            except Exception as e:
                print(f"Hiba a mérkőzés feldolgozásánál: {e}")
                continue
    
    return all_predictions

def filter_predictions(predictions, min_value_bets=1):
    """Szűrés érték fogadások alapján"""
    if predictions is None:
        return []  # Ha None, visszaadunk üres listát
    
    filtered = []
    for pred in predictions:
        value_count = sum(1 for model_bets in pred['value_bets'].values() 
                         for bet in model_bets.values() if bet)
        if value_count >= min_value_bets:
            filtered.append(pred)
    return filtered

if __name__ == "__main__":
    predictions = main()
    
    # Streamlit UI
    st.title("🏈 Mérkőzés Predikciók")
    
    # Szűrési opciók
    min_value = st.slider("Minimum érték fogadások száma", 1, 5, 1)
    selected_league = st.selectbox("Liga", ["Összes"] + list(LEAGUES.keys()))
    
    # Szűrés
    filtered = filter_predictions(predictions, min_value)
    if selected_league != "Összes":
        filtered = [p for p in filtered if p['league'] == selected_league]
    
    # Mérkőzések listázása
    for i, pred in enumerate(filtered):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.subheader(f"{pred['home_team']} vs {pred['away_team']} ({pred['league']})")
            
            for model_name, prob in pred['probs'].items():
                if model_name == 'LogisticRegression':
                    continue
                
                value_marks = pred['value_bets'][model_name]
                st.write(f"**{model_name}:**")
                st.write(f"🏠 {prob[0]*100:.1f}% ({1/prob[0]:.2f}) {'✅' if value_marks['home'] else ''}")
                st.write(f"⚖️ {prob[1]*100:.1f}% ({1/prob[1]:.2f}) {'✅' if value_marks['draw'] else ''}")
                st.write(f"✈️ {prob[2]*100:.1f}% ({1/prob[2]:.2f}) {'✅' if value_marks['away'] else ''}")
        
        with col2:
            if st.button("Küldés", key=f"send_{i}"):
                message = format_telegram_message(
                    pred['home_team'], 
                    pred['away_team'], 
                    pred['probs'], 
                    pred['odds'], 
                    pred['value_bets']
                )
                send_to_telegram(message, to="owner")
                st.success("Elküldve!")
        
        st.divider()

if __name__ == "__main__":
    main()
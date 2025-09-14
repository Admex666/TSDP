# main.py
import pandas as pd
import streamlit as st
from data_loader import load_models, load_league_data, load_fuzz_data
from stats_calculator import get_last5_stats
from predictor import MatchPredictor
from tippmix_api import get_tippmix_data
from telegram import send_to_telegram
from config import LEAGUES

def format_telegram_message_group(home, away, probs, odds, value_bets):
    """Egyszerűsített üzenet felhasználóknak"""
    # Legjobb value bet keresése
    best_bet = None
    best_strength = 0
    
    for model_name, prob in probs.items():
        if model_name == 'LogisticRegression':
            continue
        
        value_marks = value_bets[model_name]
        for outcome, has_value in value_marks.items():
            if has_value:
                if outcome == 'home':
                    strength = (1/odds['home']) / prob[0] * 10
                    if strength > best_strength:
                        best_strength = strength
                        best_bet = f"{home} győzelem"
                elif outcome == 'draw':
                    strength = (1/odds['draw']) / prob[1] * 10
                    if strength > best_strength:
                        best_strength = strength
                        best_bet = "Döntetlen"
                elif outcome == 'away':
                    strength = (1/odds['away']) / prob[2] * 10
                    if strength > best_strength:
                        best_strength = strength
                        best_bet = f"{away} győzelem"
    
    if not best_bet:
        return None
    
    # Erősség 1-10 skálán
    strength_score = min(10, max(1, int(best_strength)))
    
    message = f"🚨 ÉRTÉK FOGADÁS (Erősség: {strength_score}/10)\n\n"
    message += f"⚽ {home} vs {away}\n\n"
    message += f"🎯 AJÁNLAT: {best_bet}\n\n"
    message += "Kellemes szórakozást! 🍀"
    
    return message

def format_telegram_message_owner(home, away, probs, odds, value_bets, explanations=None):
    """Részletes üzenet tulajdonosnak"""
    message = f"🔍 SAJÁT ELEMZÉS: {home} vs {away}\n\n"
    
    # Modell predikciók
    message += "🤖 MODELL PREDIKCIÓK:\n"
    for model_name, prob in probs.items():
        if model_name == 'LogisticRegression':
            continue
        
        value_marks = value_bets[model_name]
        message += f"{model_name}:\n"
        message += f"H: {prob[0]*100:.0f}% ({1/prob[0]:.2f}) {'✅' if value_marks['home'] else '❌'} | "
        message += f"D: {prob[1]*100:.0f}% ({1/prob[1]:.2f}) {'✅' if value_marks['draw'] else '❌'} | "
        message += f"A: {prob[2]*100:.0f}% ({1/prob[2]:.2f}) {'✅' if value_marks['away'] else '❌'}\n"
    
    message += f"\n🎯 TIPPMIX ODDS: {odds['home']:.2f} | {odds['draw']:.2f} | {odds['away']:.2f}\n\n"
    
    # Magyarázat (ha van)
    if explanations and 'GradientBoosting' in explanations:
        impacts = explanations['GradientBoosting'][:5]  # Top 5
        message += "📊 MAGYARÁZAT (Top 5):\n"
        for i, impact in enumerate(impacts):
            sign = "+" if impact['impact'] > 0 else ""
            message += f"{i+1}. {impact['feature']} ({impact['original_value']:.1f}) → {sign}{impact['impact']:.1f}%\n"
    
    # Value bets összesítése
    value_count = sum(1 for model_bets in value_bets.values() 
                     for bet in model_bets.values() if bet)
    message += f"\n💰 ÉRTÉK FOGADÁSOK SZÁMA: {value_count}\n"
    
    return message

def format_detailed_explanation(home_team, away_team, impacts, prediction):
    """Részletes magyarázat generálása"""
    
    message = f"🔍 {home_team} vs {away_team} - PREDIKCIÓ MAGYARÁZAT\n\n"
    message += f"🏠 Otthoni győzelem: {prediction[0]*100:.1f}%\n"
    message += f"⚖️ Döntetlen: {prediction[1]*100:.1f}%\n" 
    message += f"✈️ Vendég győzelem: {prediction[2]*100:.1f}%\n\n"
    
    message += "📊 LEGFONTOSABB TÉNYEZŐK:\n"
    message += "-" * 40 + "\n"
    
    for i, impact in enumerate(impacts[:5]):  # Top 5 tényező
        sign = "+" if impact['impact'] > 0 else ""
        message += f"{i+1}. {impact['feature']}: {impact['original_value']:.2f} → {sign}{impact['impact']:+.1f}%\n"
    
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
    
    # Tippmix adatok
    tippmix_data = get_tippmix_data(10)
    if tippmix_data is None:
        print("tippmix_data not found")

    for league_name, league_config in LEAGUES.items():
        print(f"Processing {league_name} league...")
        
        # Ligaadatok betöltése
        df_league = load_league_data(league_config['season'], league_config['league_code'])
        if df_league is None:
            continue
        
        # Mérkőzések feldolgozása
        for _, match in tippmix_data.iterrows():
            if match['league_name'] != f"{league_name}1":
                continue
            try:
                home_team = match['Home']
                away_team = match['Away']
                
                # Csapatnév mapping - case insensitive és strip
                home_matches = fuzz_data[fuzz_data['Team_tippmix'].str.strip().str.lower() == home_team.strip().lower()]
                away_matches = fuzz_data[fuzz_data['Team_tippmix'].str.strip().str.lower() == away_team.strip().lower()]
                
                if len(home_matches) == 0:
                    print(f"WARNING: No matching found for home team: {home_team}")
                    # Próbáljunk fuzzy matching-et
                    from fuzzywuzzy import fuzz
                    best_match = None
                    best_score = 0
                    for _, row in fuzz_data.iterrows():
                        score = fuzz.ratio(home_team.lower(), row['Team_tippmix'].lower())
                        if score > best_score and score > 80:  # 80% threshold
                            best_score = score
                            best_match = row
                    
                    if best_match is not None:
                        print(f"Found fuzzy match: {home_team} -> {best_match['Team_tippmix']}")
                        home_fd = best_match['Team_fdcouk']
                    else:
                        continue
                else:
                    home_fd = home_matches['Team_fdcouk'].iloc[0]
                    
                # Ugyanez az away team-re
                if len(away_matches) == 0:
                    print(f"WARNING: No matching found for away team: {away_team}")
                    from fuzzywuzzy import fuzz
                    best_match = None
                    best_score = 0
                    for _, row in fuzz_data.iterrows():
                        score = fuzz.ratio(away_team.lower(), row['Team_tippmix'].lower())
                        if score > best_score and score > 80:
                            best_score = score
                            best_match = row
                    
                    if best_match is not None:
                        print(f"Found fuzzy match: {away_team} -> {best_match['Team_tippmix']}")
                        away_fd = best_match['Team_fdcouk']
                    else:
                        continue
                else:
                    away_fd = away_matches['Team_fdcouk'].iloc[0]
                
                # Statisztikák számítása
                home_points, home_gf, home_ga, home_days = get_last5_stats(home_fd, df_league)
                away_points, away_gf, away_ga, away_days = get_last5_stats(away_fd, df_league)
                print(home_team)
                print(home_fd, home_points)
                home_stats = {'points': home_points, 'goals_for': home_gf, 'goals_against': home_ga, 'days_since': home_days}
                away_stats = {'points': away_points, 'goals_for': away_gf, 'goals_against': away_ga, 'days_since': away_days}
                
                odds = {'home': match['H_odds'], 'draw': match['D_odds'], 'away': match['A_odds']}
                
                print(f"DEBUG: {home_team} vs {away_team}")
                print(f"DEBUG: Home stats: {home_stats}")
                print(f"DEBUG: Away stats: {away_stats}") 
                print(f"DEBUG: Odds: {odds}")

                # Predikció - most már 3 értéket ad vissza
                probs, X_original, X_scaled = predictor.predict(home_stats, away_stats, odds)
                print(f"DEBUG: Prepared features: {X_original.iloc[0].to_dict()}")

                # Ellenőrizd, hogy minden feature megvan-e
                if X_original.empty:
                    print(f"HIBA: Nincsenek feature-ök ehhez a mérkőzéshez: {home_team} vs {away_team}")
                    continue
                value_bets = predictor.analyze_value(probs, odds)
                
                # Magyarázat generálása
                explanations = {}
                for model_name, prob in probs.items():
                    impacts = predictor.explain_prediction(model_name, X_original, X_scaled, prob)
                    explanations[model_name] = impacts
                
                # Eredmények mentése
                prediction_data = {
                    'league': league_name,
                    'home_team': home_team,
                    'away_team': away_team,
                    'probs': probs,
                    'odds': odds,
                    'value_bets': value_bets,
                    'explanations': explanations,  # magyarázatok
                    'X_original': X_original,     # original adatok
                    'X_scaled': X_scaled          # scaled adatok
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
    main()
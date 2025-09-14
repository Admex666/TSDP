# mlb_main.py

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import streamlit as st
from mlb_data_loader import load_mlb_models, get_upcoming_mlb_games, get_team_recent_stats
from mlb_tippmix_api import get_mlb_tippmix_data
from mlb_predictor import MLBPredictor
from mlb_config import API_TO_TIPPMIX
from telegram import send_to_telegram
from fuzzywuzzy import fuzz

def format_telegram_message_group(home, away, predictions, odds, value_bets):
    """Egyszerű üzenet felhasználóknak"""
    if not any(value_bets.values()):
        return None
    
    # Legjobb value bet keresése
    best_bet = None
    best_strength = 0
    
    if value_bets['home_value']:
        strength = (1/odds['Home_odds']) / predictions['home_prob'] * 10
        if strength > best_strength:
            best_strength = strength
            best_bet = f"{home} győzelem"
    
    if value_bets['away_value']:
        strength = (1/odds['Away_odds']) / predictions['away_prob'] * 10
        if strength > best_strength:
            best_strength = strength
            best_bet = f"{away} győzelem"
    
    if not best_bet:
        return None
    
    # Erősség 1-10 skálán
    strength_score = min(10, max(1, int(best_strength)))
    
    message = f"🚨 MLB ÉRTÉK FOGADÁS (Erősség: {strength_score}/10)\n\n"
    message += f"⚾ {home} vs {away}\n\n"
    message += f"🎯 AJÁNLAT: {best_bet}\n\n"
    message += "Kellemes szórakozást! 🤞"
    
    return message

def format_telegram_message_owner(home, away, predictions, odds, value_bets, explanation=None):
    """Részletes üzenet tulajdonosnak"""
    message = f"📊 MLB ELEMZÉS: {home} vs {away}\n\n"
    
    # Predikciók
    message += "🤖 MODEL PREDIKCIÓ:\n"
    message += f"Home: {predictions['home_prob']*100:.0f}% ({1/predictions['home_prob']:.2f}) {'✅' if value_bets['home_value'] else '❌'}\n"
    message += f"Away: {predictions['away_prob']*100:.0f}% ({1/predictions['away_prob']:.2f}) {'✅' if value_bets['away_value'] else '❌'}\n"
    
    message += f"\n🎯 TIPPMIX ODDS: {odds['Home_odds']:.2f} | {odds['Away_odds']:.2f}\n"
    
    # Explanation
    if explanation:
        message += f"\n{explanation}\n"
    
    # Value bets összesítése
    value_count = sum(1 for v in value_bets.values() if v)
    message += f"\n💰 ÉRTÉK FOGADÁSOK SZÁMA: {value_count}\n"
    
    return message

def main():
    """Fő MLB predikciós logika"""
    # Modellek betöltése
    model, scaler, features, loaded = load_mlb_models()
    if not loaded:
        print("MLB modellek betöltése sikertelen")
        return []
    
    predictor = MLBPredictor(model, scaler, features)
    
    # Közelgő mérkőzések
    upcoming_games = get_upcoming_mlb_games(1)
    if upcoming_games.empty:
        print("Nincsenek közelgő MLB mérkőzések")
        return []
    
    # Tippmix odds
    tippmix_data = get_mlb_tippmix_data(1)
    if tippmix_data.empty:
        print("Tippmix adatok nem elérhetőek")
        return []
    
    all_predictions = []
    
    for _, game in upcoming_games.iterrows():
        try:
            home_team = game['home_team_name']
            away_team = game['away_team_name']
            
            # Csapat statisztikák
            home_stats = get_team_recent_stats(game['home_team_id'])
            away_stats = get_team_recent_stats(game['away_team_id'])
            
            # Predikció
            predictions, X_original, X_scaled = predictor.predict(home_stats, away_stats)
            if predictions is None:
                continue
            
            # Tippmix odds keresése - NOTEBOOK verzió alapján
            home_tippmix_name = API_TO_TIPPMIX.get(home_team)
            away_tippmix_name = API_TO_TIPPMIX.get(away_team)

            if home_tippmix_name is None or away_tippmix_name is None:
                print(f"Nincs mapping: {home_team} -> {home_tippmix_name}, {away_team} -> {away_tippmix_name}")
                continue

            try:
                # Keresés úgy mint a notebook-ban
                home_odds_value = tippmix_data.loc[(tippmix_data['Home'] == home_tippmix_name)]['H_odds'].iloc[0]
                away_odds_value = tippmix_data.loc[(tippmix_data['Away'] == away_tippmix_name)]['A_odds'].iloc[0]
                
                odds_dict = {
                    'Home_odds': home_odds_value,
                    'Away_odds': away_odds_value
                }
                
            except (IndexError, KeyError) as e:
                print(f"Odds nem található: {home_tippmix_name} vs {away_tippmix_name}")
                continue
            
            # Value betting elemzés
            value_bets = predictor.analyze_value(predictions, odds_dict)
            
            # Explanation
            explanation = predictor.get_prediction_explanation(
                home_team, away_team, home_stats, away_stats, predictions
            )
            
            prediction_data = {
                'game_id': game['game_id'],
                'date': game['date'],
                'home_team': home_team,
                'away_team': away_team,
                'home_stats': home_stats,
                'away_stats': away_stats,
                'predictions': predictions,
                'odds': odds_dict,
                'value_bets': value_bets,
                'explanation': explanation,
                'venue': game.get('venue', '')
            }
            
            all_predictions.append(prediction_data)
            
        except Exception as e:
            print(f"Hiba a mérkőzés feldolgozásánál {home_team} vs {away_team}: {e}")
            continue
    
    return all_predictions

def filter_mlb_predictions(predictions, min_value_bets=1):
    """Szűrés érték fogadások alapján"""
    if not predictions:
        return []
    
    filtered = []
    for pred in predictions:
        value_count = sum(1 for v in pred['value_bets'].values() if v)
        if value_count >= min_value_bets:
            filtered.append(pred)
    
    return filtered

if __name__ == "__main__":
    predictions = main()
    print(f"Generated {len(predictions)} MLB predictions")
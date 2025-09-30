import pandas as pd
from datetime import datetime
from scrape import get_date_matches
from predictor import predict_tennis_match_simple
import os

def update_tennis_paper(match_ids, csv_path='Tennis/tennis_paper.csv'):
    """
    Predikciók futtatása és dataframe frissítése - JAVÍTOTT
    """
    predictions = []
    
    for match_id in match_ids:
        print(f"🔮 Predikció futtatása: {match_id}")
        result = predict_tennis_match_simple(match_id)
        
        if result:
            # Kelly számítás biztonságosabb verziója
            try:
                odds1 = result['match_info']['player1_odds']
                odds2 = result['match_info']['player2_odds']
                p1_prob = result['prediction']['player1_prob']
                p2_prob = result['prediction']['player2_prob']
                
                # Kelly formula biztonságos változata
                kelly_p1 = max(0, (p1_prob * (odds1 - 1) - p2_prob) / (odds1 - 1)) if odds1 > 1 else 0
                kelly_p2 = max(0, (p2_prob * (odds2 - 1) - p1_prob) / (odds2 - 1)) if odds2 > 1 else 0
            except:
                kelly_p1 = 0
                kelly_p2 = 0
            
            # DataFrame sor létrehozása
            row = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'event_id': match_id,
                'player1_name': result['match_info']['player1_name'],
                'player2_name': result['match_info']['player2_name'],
                'player1_rank': result['match_info']['player1_rank'],
                'player2_rank': result['match_info']['player2_rank'],
                'surface': result['match_info']['surface'],
                'tournament': result['match_info']['tournament'],
                'player1_odds': result['match_info']['player1_odds'],
                'player2_odds': result['match_info']['player2_odds'],
                'h2h_matches': result['match_info']['h2h_matches'],
                'h2h_p1_wins': result['match_info']['h2h_p1_wins'],
                'h2h_last_winner': result['match_info']['h2h_last_winner'],
                'player1_pred_prob': result['prediction']['player1_prob'],
                'player2_pred_prob': result['prediction']['player2_prob'],
                'player1_fair_odds': result['prediction']['player1_fair_odds'],
                'player2_fair_odds': result['prediction']['player2_fair_odds'],
                'predicted_winner': result['prediction']['predicted_winner'],
                'confidence': result['prediction']['confidence'],
                'player1_value': result['value_analysis']['player1_value'],
                'player2_value': result['value_analysis']['player2_value'],
                'best_value': result['value_analysis']['best_value'],
                'kelly_p1': kelly_p1,
                'kelly_p2': kelly_p2,
                'bet_recommendation': result.get('bet_recommendation', 'NO_BET'),
                'bet_stake_percent': result.get('bet_stake_percent', 0),
                'bet_placed_on': result.get('bet_placed_on', ''),
                'actual_winner': '',
                'result': '',
                'profit': 0,
                'bet_settled': False
            }
            predictions.append(row)
    
    # DataFrame létrehozása
    new_df = pd.DataFrame(predictions)
    
    # Meglévő CSV betöltése vagy új létrehozása - JAVÍTOTT
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            # Duplikátumok eltávolítása (ugyanaz event_id)
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=['event_id'], keep='last'
            ).reset_index(drop=True)
        except Exception as e:
            print(f"⚠️  CSV fájl üres vagy hibás, új létrehozása: {e}")
            combined_df = new_df
    else:
        combined_df = new_df
    
    # CSV mentése
    combined_df.to_csv(csv_path, index=False)
    print(f"✅ CSV frissítve: {csv_path} ({len(combined_df)} sor)")
    
    return combined_df
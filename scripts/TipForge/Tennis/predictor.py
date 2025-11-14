from scrape import get_tennis_match_data, create_prediction_input
from loader import load_latest_model

def predict_tennis_match_simple(event_id):
    """
    Egyszerű predikció scrapelt tenisz meccs adatok alapján - BŐVÍTETT
    """
    # Adatok scrapelése
    match_info = get_tennis_match_data(event_id)
    
    if not match_info:
        print(f"❌ Nem sikerült lekérni a meccs adatait: {event_id}")
        return None
    
    try:
        # Predikciós input előkészítése
        prediction_input = create_prediction_input(match_info)
        
        # Predikció végrehajtása
        model_data = load_latest_model()
        model = model_data['model']
        win_probability = model.predict_proba(prediction_input)[0, 1]
        
        # Győzelmi valószínűségek
        p1_prob = win_probability
        p2_prob = 1 - win_probability
        
        # Fair odds számítás
        p1_fair_odds = 1 / p1_prob
        p2_fair_odds = 1 / p2_prob
        
        # Value számítás
        p1_value = (match_info['player1_odds'] - p1_fair_odds) / p1_fair_odds * 100
        p2_value = (match_info['player2_odds'] - p2_fair_odds) / p2_fair_odds * 100
        
        # Kelly kritérium
        p1_kelly = max(0, (p1_prob * (match_info['player1_odds'] - 1) - (1 - p1_prob)) / (match_info['player1_odds'] - 1))
        p2_kelly = max(0, (p2_prob * (match_info['player2_odds'] - 1) - (1 - p2_prob)) / (match_info['player2_odds'] - 1))
        
        # Prediktált győztes
        predicted_winner = match_info['player1_name'] if p1_prob > p2_prob else match_info['player2_name']
        confidence = max(p1_prob, p2_prob)
        
        # Betting ajánlás
        best_value = max(p1_value, p2_value)
        if best_value > 5:
            if p1_value > p2_value:
                bet_recommendation = "STRONG_BET_P1"
                bet_placed_on = match_info['player1_name']
                bet_stake_percent = min(p1_kelly * 100, 5)
            else:
                bet_recommendation = "STRONG_BET_P2"
                bet_placed_on = match_info['player2_name']
                bet_stake_percent = min(p2_kelly * 100, 5)
        elif best_value > 2:
            if p1_value > p2_value:
                bet_recommendation = "WEAK_BET_P1"
                bet_placed_on = match_info['player1_name']
                bet_stake_percent = min(p1_kelly * 100, 2)
            else:
                bet_recommendation = "WEAK_BET_P2"
                bet_placed_on = match_info['player2_name']
                bet_stake_percent = min(p2_kelly * 100, 2)
        else:
            bet_recommendation = "NO_BET"
            bet_placed_on = ""
            bet_stake_percent = 0
        
        # Eredmény kiírása
        print(f"🎾 {match_info['player1_name']} (fair: {p1_fair_odds:.2f}, prob: {p1_prob:.1%}) vs {match_info['player2_name']} (fair: {p2_fair_odds:.2f}, prob: {p2_prob:.1%})")
        
        return {
            'match_info': match_info,
            'prediction': {
                'player1_prob': p1_prob,
                'player2_prob': p2_prob,
                'player1_fair_odds': p1_fair_odds,
                'player2_fair_odds': p2_fair_odds,
                'predicted_winner': predicted_winner,
                'confidence': confidence
            },
            'value_analysis': {
                'player1_value': p1_value,
                'player2_value': p2_value,
                'best_value': best_value
            },
            'kelly_p1': p1_kelly,
            'kelly_p2': p2_kelly,
            'bet_recommendation': bet_recommendation,
            'bet_stake_percent': bet_stake_percent,
            'bet_placed_on': bet_placed_on
        }
        
    except Exception as e:
        print(f"❌ Hiba a predikció során: {str(e)}")
        return None
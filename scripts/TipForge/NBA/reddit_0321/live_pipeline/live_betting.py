import pandas as pd
import os
import datetime

def calculate_kelly(odds, prob, fraction=0.25):
    """Calculates Fractional Kelly Criterion."""
    b = odds - 1
    p = prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    return max(0, kelly) * fraction

def generate_betting_card():
    pipeline_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\live_pipeline"
    inference_path = os.path.join(pipeline_dir, "staging_3_inference.csv")
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Lépés 4: Value Calculation & Bet Sizing (1/4 Kelly)")
    
    if not os.path.exists(inference_path):
        print("❌ Hiba: Nincs Stage 3 fájl (staging_3_inference.csv).")
        return
        
    df = pd.read_csv(inference_path)
    print(f"Bemenet: {len(df)} mérkőzés predikciókkal és oddsokkal.")
    
    # 1. Edge Calculation
    df['implied_prob_home'] = 1 / df['home_odds']
    df['implied_prob_away'] = 1 / df['away_odds']
    
    df['edge_home'] = df['fair_prob_home'] - df['implied_prob_home']
    df['edge_away'] = df['fair_prob_away'] - df['implied_prob_away']
    
    # Required edge threshold (as per backtest optimization)
    MIN_EDGE = 0.02
    
    betting_card = []
    
    for idx, row in df.iterrows():
        matchup = f"{row['Away_Abbr']} @ {row['Home_Abbr']}"
        bet_target = None
        odds = 0
        prob = 0
        edge = 0
        unit_size = 0
        
        # Check Home Edge
        if row['edge_home'] > MIN_EDGE:
            bet_target = row['Home_Abbr']
            odds = row['home_odds']
            prob = row['fair_prob_home']
            edge = row['edge_home']
            unit_size = calculate_kelly(odds, prob, fraction=0.25)
            
        # Check Away Edge
        elif row['edge_away'] > MIN_EDGE:
            bet_target = row['Away_Abbr']
            odds = row['away_odds']
            prob = row['fair_prob_away']
            edge = row['edge_away']
            unit_size = calculate_kelly(odds, prob, fraction=0.25)
            
        if bet_target is not None:
            betting_card.append({
                'Matchup': matchup,
                'Bet': bet_target,
                'Odds': odds,
                'Model_Prob': prob,
                'Edge': edge,
                'Rec_Units': unit_size * 100  # Storing as percentage of bankroll
            })
            
    # Output to File
    card_df = pd.DataFrame(betting_card)
    out_path = os.path.join(pipeline_dir, "staging_4_betting_card.csv")
    card_df.to_csv(out_path, index=False)
    
    print("\n=======================================================")
    print("--- STAGE 4 OUTPUT: FINAL BETTING CARD (EYE TEST) ---")
    print("=======================================================")
    
    if len(betting_card) == 0:
        print("Mai meccsekre nincs Value Bet! (Nincs >2% edge)")
    else:
        print(f"Talált Value Betek száma: {len(betting_card)}")
        print(f"{'Meccs':<20} | {'Tipp':<5} | {'Odds':<6} | {'Valószínűség':<13} | {'Edge':<8} | {'Tét javaslat (Bankroll %)'}")
        print("-" * 88)
        
        for bet in betting_card:
            print(f"{bet['Matchup']:<20} | {bet['Bet']:<5} | {bet['Odds']:<6.2f} | {bet['Model_Prob']*100:>6.1f}%       | {bet['Edge']*100:>5.1f}%   | {bet['Rec_Units']:.2f}% (1/4 Kelly)")

    print("\n=======================================================")
    print(f"Betting Card elmentve: {out_path}")

if __name__ == "__main__":
    generate_betting_card()

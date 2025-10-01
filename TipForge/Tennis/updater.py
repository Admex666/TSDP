import pandas as pd
from datetime import datetime
from scrape import scrape_tennis_match
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


def update_betting_results(csv_path='Tennis/tennis_paper.csv'):
    """
    Fogadási eredmények frissítése a befejezett meccsekre
    """
    if not os.path.exists(csv_path):
        print("❌ CSV fájl nem található!")
        return None
    
    # CSV betöltése
    df = pd.read_csv(csv_path)
    
    # Csak a nem lezárt fogadások
    unsettled_bets = df[df['bet_settled'] == False]
    
    if unsettled_bets.empty:
        print("✅ Nincs függőben lévő fogadás")
        return df
    
    print(f"🔍 {len(unsettled_bets)} függőben lévő fogadás ellenőrzése...")
    
    updated_count = 0
    total_profit = 0
    results_summary = {
        'total_checked': 0,
        'settled_bets': 0,
        'won_bets': 0,
        'lost_bets': 0,
        'total_profit': 0,
        'total_stake': 0
    }
    
    for index, row in unsettled_bets.iterrows():
        event_id = row['event_id']
        
        try:
            # Meccs adatok lekérése
            match_data = scrape_tennis_match(event_id)
            
            if not match_data:
                continue
                
            event = match_data['event']
            status_code = event.get('status', {}).get('code')
            
            # Csak befejezett meccseket nézünk
            if status_code == 100:  # Ended
                results_summary['total_checked'] += 1
                winner_code = event.get('winnerCode')
                actual_winner = ""
                
                # Győztes meghatározása
                if winner_code == 1:
                    actual_winner = row['player1_name']
                elif winner_code == 2:
                    actual_winner = row['player2_name']
                else:
                    # Döntetlen vagy ismeretlen eredmény
                    actual_winner = "DRAW"
                
                # Eredmény és profit számítás
                result = ""
                profit = 0
                stake_percent = row['bet_stake_percent']
                
                if row['bet_recommendation'] != 'NO_BET' and actual_winner != "DRAW":
                    if actual_winner == row['bet_placed_on']:
                        # NYERESÉG
                        result = "WON"
                        if row['bet_placed_on'] == row['player1_name']:
                            profit = stake_percent * (row['player1_odds'] - 1)
                        else:
                            profit = stake_percent * (row['player2_odds'] - 1)
                        results_summary['won_bets'] += 1
                    else:
                        # VESZTESÉG
                        result = "LOST"
                        profit = -stake_percent
                        results_summary['lost_bets'] += 1
                    
                    results_summary['settled_bets'] += 1
                    results_summary['total_profit'] += profit
                    results_summary['total_stake'] += stake_percent
                
                # DataFrame frissítése
                df.at[index, 'actual_winner'] = actual_winner
                df.at[index, 'result'] = result
                df.at[index, 'profit'] = profit
                df.at[index, 'bet_settled'] = True
                
                updated_count += 1
                
                print(f"✅ {row['player1_name']} vs {row['player2_name']}: {result} ({profit:+.1f}%)")
                
        except Exception as e:
            print(f"❌ Hiba a meccs {event_id} feldolgozásakor: {str(e)}")
            continue
    
    # CSV mentése
    if updated_count > 0:
        df.to_csv(csv_path, index=False)
        print(f"✅ {updated_count} fogadás eredménye frissítve")
        
        # Összegzés megjelenítése
        print_summary(results_summary)
    else:
        print("ℹ️  Nincs új eredmény a függőben lévő fogadásokhoz")
    
    return df

def print_summary(results_summary):
    """
    Eredmények összegzésének megjelenítése
    """
    print("\n📊 FOGADÁSI EREDMÉNYEK ÖSSZEGZÉSE")
    print("=" * 50)
    
    total_checked = results_summary['total_checked']
    settled_bets = results_summary['settled_bets']
    won_bets = results_summary['won_bets']
    lost_bets = results_summary['lost_bets']
    total_profit = results_summary['total_profit']
    total_stake = results_summary['total_stake']
    
    print(f"🔍 Átfésült meccsek: {total_checked}")
    print(f"🎯 Lezárt fogadások: {settled_bets}")
    
    if settled_bets > 0:
        hit_rate = (won_bets / settled_bets) * 100
        roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
        
        print(f"✅ Nyert fogadások: {won_bets}")
        print(f"❌ Vesztett fogadások: {lost_bets}")
        print(f"🎯 Találati arány: {hit_rate:.1f}%")
        print(f"💰 Összes profit: {total_profit:+.1f}%")
        print(f"📈 ROI (Return on Investment): {roi:+.1f}%")
        
        # Teljesítmény értékelés
        if roi > 0:
            performance = "KIVÁLÓ 🔥" if roi > 10 else "JÓ ✅"
        else:
            performance = "GYENGE 📉" if roi > -5 else "ROSSZ ❌"
            
        print(f"🏆 Teljesítmény: {performance}")
    else:
        print("ℹ️  Nincs lezárt fogadás az átfésült meccsek között")
    
    print("=" * 50)
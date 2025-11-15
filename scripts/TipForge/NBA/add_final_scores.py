# add_final_scores.py

"""
Lépés 4: Végeredmények hozzáadása (target változók)
"""

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3
import time

def add_final_scores(ml_ready_csv, output_csv='ml_ready_with_targets.csv'):
    """Végeredmények hozzáadása minden snapshot-hoz"""
    
    df = pd.read_csv(ml_ready_csv)
    
    # Cache a végeredményekhez
    final_scores_cache = {}
    
    for idx, row in df.iterrows():
        game_id = f"00{row['game_id']}"
        
        # Ha még nincs cache-elve, lekérjük
        if game_id not in final_scores_cache:
            print(f"[{idx+1}/{len(df)}] Végeredmény lekérése: {game_id}")
            
            try:
                bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
                data = bs.get_dict()
                
                home_stats = data['boxScoreTraditional']['homeTeam']['statistics']
                away_stats = data['boxScoreTraditional']['awayTeam']['statistics']
                
                final_scores_cache[game_id] = {
                    'final_home_score': home_stats['points'],
                    'final_away_score': away_stats['points']
                }
                
                time.sleep(0.6)
                
            except Exception as e:
                print(f"Hiba {game_id}: {e}")
                final_scores_cache[game_id] = {
                    'final_home_score': None,
                    'final_away_score': None
                }
                time.sleep(2)
        
        # Végeredmények beállítása
        final = final_scores_cache[game_id]
        df.at[idx, 'final_home_score'] = final['final_home_score']
        df.at[idx, 'final_away_score'] = final['final_away_score']
        
        if final['final_home_score'] and final['final_away_score']:
            df.at[idx, 'home_won'] = 1 if final['final_home_score'] > final['final_away_score'] else 0
            df.at[idx, 'final_score_diff'] = final['final_home_score'] - final['final_away_score']
        
        # Mentés minden 20. sor után
        if (idx + 1) % 20 == 0:
            df.to_csv(output_csv, index=False)
            print(f"Mentve: {idx+1} sor")
    
    # Végső mentés
    df.to_csv(output_csv, index=False)
    print(f"Kész! Target változókkal: {output_csv}")
    
    # Statisztika
    print(f"\nÖsszesen: {len(df)} snapshot")
    t = df.groupby("game_id")['home_won'].mean()
    print(f"Home wins: {len(t[t > 0])}")
    print(f"Away wins: {len(df.game_id.unique()) - len(t[t > 0])}")

if __name__ == "__main__":
    add_final_scores('data/ml_ready_2024_25.csv', 'data/ml_ready_targets_2024_25.csv')


# data_collect_pipe.py

"""
Teljes pipeline futtatása: 2023-24 és 2024-25 szezonok
"""

import time
from collect_games import collect_games
from collect_snapshots import process_games
from enrich_features import enrich_snapshots
from add_final_scores import add_final_scores

def run_pipeline(season):
    """Egy szezon teljes feldolgozása"""
    print(f"\n{'='*50}")
    print(f"SZEZON: {season}")
    print(f"{'='*50}\n")
    
    # 1. Meccsek összegyűjtése
    print("1. Meccsek gyűjtése...")
    games_file = f'data/games_{season.replace("-", "_")}.csv'
    collect_games(season, games_file)
    time.sleep(1)
    
    # 2. Snapshots készítése
    print("\n2. Snapshots készítése...")
    snapshots_file = f'data/snapshots_{season.replace("-", "_")}.csv'
    process_games(games_file, snapshots_file)
    time.sleep(1)
    
    # 3. Feature enrichment
    print("\n3. Feature enrichment...")
    ml_ready_file = f'data/ml_ready_{season.replace("-", "_")}.csv'
    enrich_snapshots(snapshots_file, ml_ready_file, season)
    time.sleep(1)
    
    # 4. Végeredmények hozzáadása
    print("\n4. Végeredmények hozzáadása...")
    final_file = f'data/ml_ready_targets_{season.replace("-", "_")}.csv'
    add_final_scores(ml_ready_file, final_file)
    
    print(f"\n✅ {season} kész! Végső file: {final_file}")

if __name__ == "__main__":
    # 2023-24 szezon
    run_pipeline('2023-24')
    
    # 2024-25 szezon
    run_pipeline('2024-25')
    
    print("\n" + "="*50)
    print("🎉 MINDKÉT SZEZON KÉSZ!")
    print("="*50)
import os
import sys
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_schema import PlayerMatchStats, Player
from fantasy_calculator import calculate_fantasy_points

# Setup DB connection
DB_PATH = os.path.join(os.path.dirname(__file__), 'sofascore_fantasy.db')
engine = create_engine(f'sqlite:///{DB_PATH}')
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

def verify_stats_keys_and_calc():
    print("--- Recalculating All Fantasy Points in DB ---\n")
    
    # Query all player stats
    all_stats = session.query(PlayerMatchStats).join(Player).all()
    
    if not all_stats:
        print("No stats found.")
        return

    # First row special highlight
    first_row = all_stats[0]
    
    print(f"--- [FIRST ROW HIGHLIGHT] ---")
    print(f"Player: {first_row.player.name} ({first_row.player.position})")
    print(f"Old Points (DB): {first_row.total_points}")
    
    new_points_first = calculate_fantasy_points(first_row.detailed_stats, first_row.player.position)
    print(f"New Points (Calc): {new_points_first}")
    print(f"Detailed Stats:\n{json.dumps(first_row.detailed_stats, indent=2)}")
    print("-" * 30 + "\n")
    
    print("--- Sample Comparisons (Next 5) ---")
    for row in all_stats[1:6]:
        new_p = calculate_fantasy_points(row.detailed_stats, row.player.position)
        print(f"{row.player.name:<25} | {row.player.position:<3} | Old: {row.total_points:>4} -> New: {new_p:>4}")

if __name__ == "__main__":
    verify_stats_keys_and_calc()

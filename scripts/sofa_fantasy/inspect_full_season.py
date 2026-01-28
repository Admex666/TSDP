import os
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from db_schema import Match, PlayerMatchStats, Tournament, Team, Player

DB_PATH = os.path.join(os.path.dirname(__file__), 'sofascore_fantasy.db')
engine = create_engine(f'sqlite:///{DB_PATH}')
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

def inspect_db_coverage():
    print("--- Inspecting Database Coverage (Rounds 1-38) ---\n")
    
    # 1. Matches Per Round
    matches_per_round = session.query(Match.round, func.count(Match.id)).group_by(Match.round).order_by(Match.round).all()
    
    if not matches_per_round:
        print("No matches found in DB!")
        return

    print("Matches per Round:")
    missing_rounds = []
    incomplete_rounds = []
    
    rounds_found = {r for r, c in matches_per_round}
    for r in range(1, 39):
        if r not in rounds_found:
            missing_rounds.append(r)
    
    for r, count in matches_per_round:
        if count < 10: # Premier League has 10 matches per round
            incomplete_rounds.append(f"Round {r}: {count} matches")
        # print(f"  Round {r}: {count} matches") # Too verbose to print all 38

    if missing_rounds:
        print(f"❌ MISSING ROUNDS: {missing_rounds}")
    else:
        print("✅ All 38 rounds present.")

    if incomplete_rounds:
        print(f"⚠️ INCOMPLETE ROUNDS (<10 matches): {incomplete_rounds}")
    else:
        print("✅ All rounds have >= 10 matches coverage.")

    # 2. Total Counts
    total_matches = session.query(Match).count()
    total_stats = session.query(PlayerMatchStats).count()
    total_players = session.query(Player).count()
    
    print(f"\nTotal Matches: {total_matches} (Expected approx 380)")
    print(f"Total Player Stats Records: {total_stats}")
    print(f"Unique Players: {total_players}")
    
    # 3. Stats Quality Check
    # Check for stats with 0 points but played > 60 mins (possible calculation error or terrible performance)
    weird_stats = session.query(PlayerMatchStats).filter(PlayerMatchStats.minutes > 60, PlayerMatchStats.total_points == 0).count()
    print(f"\nStats with 0 points despite >60 mins played: {weird_stats} (inspect if high number)")
    
    # Check average points
    avg_points = session.query(func.avg(PlayerMatchStats.total_points)).scalar()
    print(f"Average Fantasy Points across all records: {avg_points:.2f}")

if __name__ == "__main__":
    inspect_db_coverage()

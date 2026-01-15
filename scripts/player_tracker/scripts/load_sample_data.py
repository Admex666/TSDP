"""
Sample data loader for testing
"""
import sys
from pathlib import Path
from datetime import date, timedelta
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.database import SessionLocal
from app.models import League, Team, Player, PlayerStats
from config import config


def create_sample_data():
    """
    Create sample data for testing
    """
    db = SessionLocal()
    
    try:
        print("Creating sample data...")
        
        # Create leagues
        nb1 = League(name="NB I", country="Hungary", tier=1)
        bundesliga = League(name="Bundesliga", country="Germany", tier=1)
        eredivisie = League(name="Eredivisie", country="Netherlands", tier=1)
        
        db.add_all([nb1, bundesliga, eredivisie])
        db.flush()
        
        print("✓ Created 3 leagues")
        
        # Create teams
        ftc = Team(name="Ferencváros", league_id=nb1.id, country="Hungary")
        paks = Team(name="Paks", league_id=nb1.id, country="Hungary")
        bayern = Team(name="Bayern Munich", league_id=bundesliga.id, country="Germany")
        ajax = Team(name="Ajax", league_id=eredivisie.id, country="Netherlands")
        
        db.add_all([ftc, paks, bayern, ajax])
        db.flush()
        
        print("✓ Created 4 teams")
        
        # Create sample players
        players_data = [
            {"name": "Szabó Dominik", "position": "MID", "detailed_position": "CM", "team": ftc, "tracked": True},
            {"name": "Nagy Péter", "position": "ATT", "detailed_position": "ST", "team": ftc, "tracked": True},
            {"name": "Kovács Márk", "position": "DEF", "detailed_position": "CB", "team": paks, "tracked": True},
            {"name": "Tóth András", "position": "GK", "detailed_position": "GK", "team": paks, "tracked": False},
            {"name": "Horváth Bence", "position": "MID", "detailed_position": "CAM", "team": bayern, "tracked": True},
            {"name": "Kiss Dániel", "position": "ATT", "detailed_position": "LW", "team": ajax, "tracked": True},
        ]
        
        players = []
        for p_data in players_data:
            player = Player(
                name=p_data["name"],
                position=p_data["position"],
                detailed_position=p_data["detailed_position"],
                current_team_id=p_data["team"].id,
                nationality="Hungary",
                tracked=p_data["tracked"]
            )
            players.append(player)
            db.add(player)
        
        db.flush()
        print(f"✓ Created {len(players)} players")
        
        # Create sample stats for each player
        stats_count = 0
        for player in players:
            # Create 10 random match stats
            for i in range(10):
                match_date = date.today() - timedelta(days=i*7)
                
                # Generate random stats based on position
                if player.position == "ATT":
                    goals = random.randint(0, 2)
                    assists = random.randint(0, 1)
                    shots = random.randint(2, 8)
                elif player.position == "MID":
                    goals = random.randint(0, 1)
                    assists = random.randint(0, 2)
                    shots = random.randint(1, 5)
                elif player.position == "DEF":
                    goals = 0
                    assists = 0
                    shots = random.randint(0, 2)
                else:  # GK
                    goals = 0
                    assists = 0
                    shots = 0
                
                stats = PlayerStats(
                    player_id=player.id,
                    date=match_date,
                    minutes_played=random.choice([90, 90, 90, 75, 60]),
                    goals=goals,
                    assists=assists,
                    shots=shots,
                    shots_on_target=random.randint(0, shots) if shots > 0 else 0,
                    pass_completion=random.uniform(75, 95),
                    passes_attempted=random.randint(30, 80),
                    passes_completed=random.randint(25, 75),
                    tackles=random.randint(0, 5) if player.position != "ATT" else 0,
                    interceptions=random.randint(0, 4) if player.position == "DEF" else 0,
                    source="sample"
                )
                db.add(stats)
                stats_count += 1
        
        print(f"✓ Created {stats_count} stat records")
        
        # Commit all changes
        db.commit()
        print("\n✅ Sample data created successfully!")
        print(f"\nYou can now:")
        print(f"1. Start the server: python -m app.main")
        print(f"2. Open browser: http://localhost:8003")
        print(f"3. View tracked players on the dashboard")
        
    except Exception as e:
        db.rollback()
        print(f"\n❌ Error creating sample data: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    create_sample_data()

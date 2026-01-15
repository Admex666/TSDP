"""
Database initialization script
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.database import engine, Base
from app.models import Player, Team, League, Match, PlayerStats, PlayerEvaluation, LeagueStrength
from config import config


def init_database():
    """
    Initialize database with tables
    """
    print("Creating database tables...")
    
    # Ensure data directory exists
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print(f"✓ Database created at: {config.DATABASE_URL}")
    print("✓ Tables created:")
    print("  - leagues")
    print("  - league_strengths")
    print("  - teams")
    print("  - players")
    print("  - matches")
    print("  - player_stats")
    print("  - player_evaluations")


if __name__ == "__main__":
    init_database()

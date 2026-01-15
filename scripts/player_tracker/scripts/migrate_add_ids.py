"""
Add fbref_id and sofascore_id columns to players table
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.database import SessionLocal, engine
from sqlalchemy import text


def migrate_database():
    """
    Add fbref_id and sofascore_id columns to players table
    """
    db = SessionLocal()
    
    try:
        print("Adding fbref_id and sofascore_id columns to players table...")
        
        # Check if columns already exist
        result = db.execute(text("PRAGMA table_info(players)"))
        columns = [row[1] for row in result.fetchall()]
        
        if 'fbref_id' not in columns:
            db.execute(text("ALTER TABLE players ADD COLUMN fbref_id VARCHAR"))
            db.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_players_fbref_id ON players(fbref_id)"))
            print("✓ Added fbref_id column")
        else:
            print("✓ fbref_id column already exists")
        
        if 'sofascore_id' not in columns:
            db.execute(text("ALTER TABLE players ADD COLUMN sofascore_id INTEGER"))
            db.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_players_sofascore_id ON players(sofascore_id)"))
            print("✓ Added sofascore_id column")
        else:
            print("✓ sofascore_id column already exists")
        
        db.commit()
        print("\n✅ Database migration completed successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"\n❌ Migration failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    migrate_database()

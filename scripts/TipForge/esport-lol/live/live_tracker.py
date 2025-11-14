"""
Live Match Tracker
Continuously monitors matches and identifies value betting opportunities
"""

import time
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveMatchTracker:
    """
    Tracks live matches and stores historical data
    """
    
    def __init__(self, db_path: str = "data/live_tracking.db"):
        """
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Match snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                game_time TEXT NOT NULL,
                match_url TEXT NOT NULL,
                blue_kills INTEGER,
                red_kills INTEGER,
                blue_towers INTEGER,
                red_towers INTEGER,
                blue_drakes INTEGER,
                red_drakes INTEGER,
                blue_barons INTEGER,
                red_barons INTEGER,
                blue_gold INTEGER,
                red_gold INTEGER,
                gold_diff INTEGER,
                predicted_blue_prob REAL,
                predicted_red_prob REAL,
                raw_data TEXT
            )
        """)
        
        # Odds snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                match_url TEXT NOT NULL,
                market_name TEXT NOT NULL,
                team_name TEXT NOT NULL,
                odds REAL NOT NULL,
                raw_data TEXT
            )
        """)
        
        # Value bets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS value_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                game_time TEXT NOT NULL,
                match_url TEXT NOT NULL,
                team TEXT NOT NULL,
                team_name TEXT NOT NULL,
                market_name TEXT NOT NULL,
                odds REAL NOT NULL,
                predicted_prob REAL NOT NULL,
                implied_prob REAL NOT NULL,
                edge REAL NOT NULL,
                kelly_fraction REAL NOT NULL,
                confidence TEXT NOT NULL,
                status TEXT DEFAULT 'OPEN',
                entry_time TEXT,
                exit_time TEXT,
                result TEXT
            )
        """)
        
        # Active positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value_bet_id INTEGER NOT NULL,
                entry_timestamp TEXT NOT NULL,
                entry_game_time TEXT NOT NULL,
                team TEXT NOT NULL,
                entry_odds REAL NOT NULL,
                entry_prob REAL NOT NULL,
                stake_amount REAL,
                current_prob REAL,
                last_update TEXT,
                FOREIGN KEY (value_bet_id) REFERENCES value_bets(id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized at {self.db_path}")
    
    def save_match_snapshot(self, match_stats: Dict, predicted_probs: tuple):
        """Save match statistics snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        blue = match_stats['blue_team']
        red = match_stats['red_team']
        prob_blue, prob_red = predicted_probs
        
        cursor.execute("""
            INSERT INTO match_snapshots (
                timestamp, game_time, match_url,
                blue_kills, red_kills, blue_towers, red_towers,
                blue_drakes, red_drakes, blue_barons, red_barons,
                blue_gold, red_gold, gold_diff,
                predicted_blue_prob, predicted_red_prob, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match_stats['timestamp'],
            match_stats['game_time'],
            match_stats.get('url', 'unknown'),
            blue['kills'], red['kills'],
            blue['towers'], red['towers'],
            len(blue['dragons']), len(red['dragons']),
            blue['barons'], red['barons'],
            blue['gold'], red['gold'],
            blue['gold'] - red['gold'],
            prob_blue, prob_red,
            json.dumps(match_stats)
        ))
        
        conn.commit()
        conn.close()
    
    def save_odds_snapshot(self, odds_data: Dict, match_url: str):
        """Save odds snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for market in odds_data['markets']:
            for option in market['options']:
                cursor.execute("""
                    INSERT INTO odds_snapshots (
                        timestamp, match_url, market_name, team_name, odds, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    odds_data['timestamp'],
                    match_url,
                    market['name'],
                    option['name'],
                    option['odds'],
                    json.dumps(option)
                ))
        
        conn.commit()
        conn.close()
    
    def save_value_bet(self, value_bet: Dict) -> int:
        """
        Save identified value bet
        
        Returns:
            ID of the saved value bet
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO value_bets (
                timestamp, game_time, match_url, team, team_name,
                market_name, odds, predicted_prob, implied_prob,
                edge, kelly_fraction, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            value_bet['timestamp'],
            value_bet['game_time'],
            value_bet.get('match_url', 'unknown'),
            value_bet['team'],
            value_bet['team_name'],
            value_bet['market_name'],
            value_bet['odds'],
            value_bet['predicted_prob'],
            value_bet['implied_prob'],
            value_bet['edge'],
            value_bet['kelly_fraction'],
            value_bet['confidence']
        ))
        
        bet_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return bet_id
    
    def open_position(self, value_bet_id: int, stake_amount: float = None):
        """Open a betting position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get value bet details
        cursor.execute("""
            SELECT timestamp, game_time, team, odds, predicted_prob
            FROM value_bets WHERE id = ?
        """, (value_bet_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Value bet ID {value_bet_id} not found")
        
        entry_time, game_time, team, entry_odds, entry_prob = row
        
        cursor.execute("""
            INSERT INTO active_positions (
                value_bet_id, entry_timestamp, entry_game_time,
                team, entry_odds, entry_prob, stake_amount, current_prob, last_update
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            value_bet_id, entry_time, game_time, team,
            entry_odds, entry_prob, stake_amount, entry_prob,
            datetime.now().isoformat()
        ))
        
        # Update value bet status
        cursor.execute("""
            UPDATE value_bets SET status = 'ACTIVE', entry_time = ?
            WHERE id = ?
        """, (entry_time, value_bet_id))
        
        conn.commit()
        conn.close()
        logger.info(f"📊 Opened position for value bet #{value_bet_id}")
    
    def update_active_positions(self, current_prob_blue: float, current_prob_red: float):
        """Update all active positions with current probabilities"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, team FROM active_positions")
        positions = cursor.fetchall()
        
        for pos_id, team in positions:
            current_prob = current_prob_blue if team == 'BLUE' else current_prob_red
            
            cursor.execute("""
                UPDATE active_positions 
                SET current_prob = ?, last_update = ?
                WHERE id = ?
            """, (current_prob, datetime.now().isoformat(), pos_id))
        
        conn.commit()
        conn.close()
    
    def close_position(self, value_bet_id: int, result: str):
        """Close a betting position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        exit_time = datetime.now().isoformat()
        
        cursor.execute("""
            UPDATE value_bets 
            SET status = 'CLOSED', exit_time = ?, result = ?
            WHERE id = ?
        """, (exit_time, result, value_bet_id))
        
        cursor.execute("""
            DELETE FROM active_positions WHERE value_bet_id = ?
        """, (value_bet_id,))
        
        conn.commit()
        conn.close()
        logger.info(f"🔒 Closed position #{value_bet_id} - {result}")
    
    def get_active_positions(self) -> List[Dict]:
        """Get all currently active positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                ap.id, ap.value_bet_id, ap.entry_timestamp, ap.entry_game_time,
                ap.team, ap.entry_odds, ap.entry_prob, ap.stake_amount,
                ap.current_prob, ap.last_update,
                vb.team_name, vb.edge, vb.confidence
            FROM active_positions ap
            JOIN value_bets vb ON ap.value_bet_id = vb.id
        """)
        
        positions = []
        for row in cursor.fetchall():
            positions.append({
                'position_id': row[0],
                'value_bet_id': row[1],
                'entry_timestamp': row[2],
                'entry_game_time': row[3],
                'team': row[4],
                'entry_odds': row[5],
                'entry_prob': row[6],
                'stake_amount': row[7],
                'current_prob': row[8],
                'last_update': row[9],
                'team_name': row[10],
                'edge': row[11],
                'confidence': row[12]
            })
        
        conn.close()
        return positions
    
    def get_match_history(self, match_url: str, limit: int = 100) -> List[Dict]:
        """Get historical snapshots for a match"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, game_time, blue_kills, red_kills,
                   blue_gold, red_gold, predicted_blue_prob, predicted_red_prob
            FROM match_snapshots
            WHERE match_url = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (match_url, limit))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'timestamp': row[0],
                'game_time': row[1],
                'blue_kills': row[2],
                'red_kills': row[3],
                'blue_gold': row[4],
                'red_gold': row[5],
                'predicted_blue_prob': row[6],
                'predicted_red_prob': row[7]
            })
        
        conn.close()
        return history
    
    def get_performance_summary(self) -> Dict:
        """Get summary of betting performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total value bets identified
        cursor.execute("SELECT COUNT(*) FROM value_bets")
        total_identified = cursor.fetchone()[0]
        
        # Active positions
        cursor.execute("SELECT COUNT(*) FROM active_positions")
        active_count = cursor.fetchone()[0]
        
        # Closed positions by result
        cursor.execute("""
            SELECT result, COUNT(*) 
            FROM value_bets 
            WHERE status = 'CLOSED' AND result IS NOT NULL
            GROUP BY result
        """)
        results = dict(cursor.fetchall())
        
        # Average edge of identified bets
        cursor.execute("SELECT AVG(edge) FROM value_bets")
        avg_edge = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_value_bets_identified': total_identified,
            'active_positions': active_count,
            'closed_results': results,
            'average_edge': avg_edge
        }


if __name__ == "__main__":
    # Test tracker
    tracker = LiveMatchTracker()
    
    # Sample data
    sample_stats = {
        'timestamp': datetime.now().isoformat(),
        'game_time': '15:00',
        'url': 'https://example.com/match/123',
        'blue_team': {'kills': 10, 'towers': 2, 'dragons': ['Ocean'], 
                      'barons': 0, 'gold': 35000, 'inhibitors': 0},
        'red_team': {'kills': 8, 'towers': 1, 'dragons': [], 
                     'barons': 0, 'gold': 32000, 'inhibitors': 0}
    }
    
    tracker.save_match_snapshot(sample_stats, (0.65, 0.35))
    print("✅ Match snapshot saved")
    
    summary = tracker.get_performance_summary()
    print(f"Performance summary: {summary}")
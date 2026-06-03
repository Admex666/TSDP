import sqlite3
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Kezeli az SQLite adatbázis kapcsolatot és sémát a historikus adatokhoz.
    """
    def __init__(self, db_path="sorare_historical.db"):
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Létrehozza a szükséges táblákat, ha még nem léteznek."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Játékosok alapadatainak táblája
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS players (
                        id TEXT PRIMARY KEY,
                        slug TEXT UNIQUE,
                        display_name TEXT,
                        age INTEGER,
                        position TEXT,
                        club_name TEXT,
                        average_score REAL,
                        is_injured INTEGER DEFAULT 0,
                        is_suspended INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Konkrét meccsteljesítmények táblája (historikus elemzésekhez)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS match_performances (
                        id TEXT PRIMARY KEY,
                        player_id TEXT,
                        match_date TIMESTAMP,
                        opponent TEXT,
                        is_home INTEGER,
                        decisive_score REAL,
                        all_around_score REAL,
                        total_score REAL,
                        FOREIGN KEY(player_id) REFERENCES players(id)
                    )
                ''')
                
                # Kártyák / Aukciók / Historikus eladások táblája
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS auctions (
                        id TEXT PRIMARY KEY,
                        player_id TEXT,
                        price_eur REAL,
                        price_eth REAL,
                        price_type TEXT, -- 'auction', 'direct_listing', 'recent_sale'
                        date TIMESTAMP,
                        FOREIGN KEY(player_id) REFERENCES players(id)
                    )
                ''')
                
                conn.commit()
                logger.info("Adatbázis séma inicializálva.")
        except Exception as e:
            logger.error(f"Hiba az adatbázis inicializálásakor: {e}")

    def save_match_performance(self, perf_data):
        """Elment egy meccsteljesítményt (INSERT OR REPLACE)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO match_performances 
                    (id, player_id, match_date, opponent, is_home, decisive_score, all_around_score, total_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    perf_data.get('id'),
                    perf_data.get('player_id'),
                    perf_data.get('match_date'),
                    perf_data.get('opponent'),
                    perf_data.get('is_home'),
                    perf_data.get('decisive_score'),
                    perf_data.get('all_around_score'),
                    perf_data.get('total_score')
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Hiba a meccsteljesítmény mentésekor ({perf_data.get('id')}): {e}")

    def save_auction(self, auction_data):
        """Elment egy aukciót vagy áradatot (INSERT OR REPLACE)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO auctions 
                    (id, player_id, price_eur, price_eth, price_type, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    auction_data.get('id'),
                    auction_data.get('player_id'),
                    auction_data.get('price_eur'),
                    auction_data.get('price_eth'),
                    auction_data.get('price_type'),
                    auction_data.get('date')
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Hiba az áradat mentésekor ({auction_data.get('id')}): {e}")

    def save_player(self, player_data):
        """Elment egy játékost (INSERT OR REPLACE)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Dinamikus oszlop hozzáadások a visszamenőleges kompatibilitásért
                try:
                    cursor.execute('ALTER TABLE players ADD COLUMN average_score REAL')
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute('ALTER TABLE players ADD COLUMN is_injured INTEGER DEFAULT 0')
                except sqlite3.OperationalError:
                    pass
                
                try:
                    cursor.execute('ALTER TABLE players ADD COLUMN is_suspended INTEGER DEFAULT 0')
                except sqlite3.OperationalError:
                    pass

                # Kiszámoljuk, hogy van-e aktív sérülés vagy eltiltás
                active_injuries = player_data.get('activeInjuries', [])
                is_injured = 1 if any(inj.get('active') for inj in active_injuries if inj) else 0
                
                active_suspensions = player_data.get('activeSuspensions', [])
                is_suspended = 1 if any(susp.get('active') for susp in active_suspensions if susp) else 0

                cursor.execute('''
                    INSERT OR REPLACE INTO players 
                    (id, slug, display_name, age, position, club_name, average_score, is_injured, is_suspended, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    player_data.get('id'),
                    player_data.get('slug'),
                    player_data.get('displayName'),
                    player_data.get('age'),
                    player_data.get('position'),
                    player_data.get('activeClub', {}).get('name') if player_data.get('activeClub') else "Unknown",
                    player_data.get('averageScore'),
                    is_injured,
                    is_suspended
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Hiba a játékos mentésekor ({player_data.get('slug')}): {e}")

    def get_all_players(self):
        """Lekéri az összes elmentett játékost"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM players')
            return cursor.fetchall()

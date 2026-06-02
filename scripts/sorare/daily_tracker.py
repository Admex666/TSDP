import logging
import time
from datetime import datetime
from database import DatabaseManager
from historical_collector import HistoricalDataCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SorareDailyPriceTracker:
    def __init__(self):
        self.db = DatabaseManager()
        self.collector = HistoricalDataCollector()

    def run_daily_tracking(self):
        """
        Lekéri az adatbázisban lévő összes játékost, frissíti az aktív 
        piaci floor árukat, és időbélyeggel elmenti őket az auctions táblába.
        """
        logger.info("Napi idősoros árszinkronizáció indítása...")
        
        # Játékosok lekérése a DB-ből
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, slug, display_name FROM players")
        players = cursor.fetchall()
        conn.close()
        
        if not players:
            logger.warning("Nincsenek játékosok az adatbázisban a napi követéshez.")
            return
            
        logger.info(f"{len(players)} játékos piacának frissítése...")
        
        # Valós ETH/EUR árfolyam lekérése
        eth_eur_rate = self.collector.get_eth_eur_rate()
        now = datetime.now()
        updated_count = 0
        
        for p_id, slug, display_name in players:
            logger.info(f"Árak szinkronizálása: {display_name}...")
            
            # Valós hirdetések lekérése
            offers = self.collector.fetch_real_floor_prices(slug)
            
            if offers:
                for idx, offer in enumerate(offers):
                    p_eth = offer['price_eth']
                    p_eur = p_eth * eth_eur_rate
                    
                    # Elmentjük az árat
                    self.db.save_auction({
                        'id': f"track_{p_id}_{offer['id']}_{now.strftime('%Y%m%d')}",
                        'player_id': p_id,
                        'price_eur': round(p_eur, 2),
                        'price_eth': round(p_eth, 5),
                        'price_type': 'direct_listing',
                        'date': now.strftime('%Y-%m-%dT%H:%M:%SZ')
                    })
                
                logger.info(f"-> {display_name} árai sikeresen rögzítve.")
                updated_count += 1
            else:
                logger.warning(f"-> Nem találtam aktív hirdetést a következő játékoshoz: {display_name}")
                
            # Rate limit elkerülése
            time.sleep(1.5)
            
        logger.info(f"Napi árszinkronizáció befejeződött! Összesen {updated_count} játékos ára frissítve.")

if __name__ == "__main__":
    tracker = SorareDailyPriceTracker()
    tracker.run_daily_tracking()

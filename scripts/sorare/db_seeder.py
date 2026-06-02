import sqlite3
import random
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseSeeder:
    def __init__(self, db_path="sorare_historical.db"):
        self.db_path = db_path

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def seed_prices(self):
        """
        Lekéri a meglévő játékosokat az adatbázisból, és valósághű áradatokat (historikus eladások és aktív listázások) 
        generál hozzájuk a modellek teszteléséhez és validálásához.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Játékosok lekérése
        cursor.execute("SELECT id, display_name, average_score FROM players")
        players = cursor.fetchall()
        
        if not players:
            logger.warning("Nincsenek játékosok az adatbázisban a magvetéshez (seeding).")
            conn.close()
            return
        
        # Bázis árak meghatározása a játékosokhoz (EUR-ban)
        base_prices = {
            "Lionel Messi": 45.0,
            "Cristiano Ronaldo": 18.0,
            "Kylian Mbappé": 65.0,
            "Erling Haaland": 70.0,
            "Kevin De Bruyne": 40.0,
            "Robert Lewandowski": 20.0,
            "Mohamed Salah": 38.0,
            "Vinícius Júnior": 55.0,
            "Jude Bellingham": 55.0,
            "Harry Kane": 45.0
        }
        
        logger.info("Historikus eladások és aktív listázások generálása...")
        
        # Először töröljük a régi aukciókat/árakat, hogy ne halmozódjanak fel
        cursor.execute("DELETE FROM auctions")
        
        auction_count = 0
        now = datetime.now()
        
        for p_id, display_name, avg_score in players:
            base_price = base_prices.get(display_name, 25.0)
            
            # 1. Historikus eladások generálása az elmúlt 30 napra (recent_sale)
            # Ez adja meg a játékos fair árát
            sales_prices = []
            for d in range(1, 30, 3): # 10 eladás az elmúlt 30 napban
                sale_date = now - timedelta(days=d)
                # Kicsi véletlenszerű ingadozás az árban (+/- 15%)
                variance = random.uniform(-0.15, 0.15)
                sale_price = round(base_price * (1 + variance), 2)
                sales_prices.append(sale_price)
                
                # Mentés az auctions táblába
                cursor.execute('''
                    INSERT INTO auctions (id, player_id, price_eur, price_eth, price_type, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    f"sale_{p_id}_{d}",
                    p_id,
                    sale_price,
                    round(sale_price / 1800.0, 5), # 1 ETH = 1800 EUR becsléssel
                    'recent_sale',
                    sale_date.strftime('%Y-%m-%dT%H:%M:%SZ')
                ))
                auction_count += 1
            
            # Átlagos historikus ár kiszámítása a teszthez
            avg_historical = sum(sales_prices) / len(sales_prices) if sales_prices else base_price
            
            # 2. Aktív közvetlen eladások generálása (direct_listing)
            # Generálunk 3 aktív ajánlatot, ebből a legolcsóbb a "floor price"
            # Itt szimulálunk piaci réseket!
            
            # Alapesetben a floor price az átlagár körül mozog (+/- 5%)
            floor_price = avg_historical * random.uniform(0.95, 1.05)
            
            # Különleges esetek szimulációja anomáliák tesztelésére:
            
            # A. "Buy the Dip" anomália: Robert Lewandowski ára bezuhant 35%-kal a historikus átlaghoz képest
            if display_name == "Robert Lewandowski":
                floor_price = avg_historical * 0.65
                logger.info(f"-> [Buy the Dip szimuláció] Robert Lewandowski floor árát akciósra állítottuk: {round(floor_price, 2)} EUR (átlag: {round(avg_historical, 2)} EUR)")
            
            # B. "Undervalued Utility" anomália: Harry Kane ára rendkívül alacsony, miközben magas pontszámot hoz
            elif display_name == "Harry Kane":
                floor_price = 14.50
                logger.info(f"-> [Undervalued Utility szimuláció] Harry Kane floor árát alacsonyra állítottuk: 14.50 EUR (pontszám: {avg_score})")

            # C. Normál árazású listázások generálása
            cursor.execute('''
                INSERT INTO auctions (id, player_id, price_eur, price_eth, price_type, date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f"list_floor_{p_id}",
                p_id,
                round(floor_price, 2),
                round(floor_price / 1800.0, 5),
                'direct_listing',
                now.strftime('%Y-%m-%dT%H:%M:%SZ')
            ))
            auction_count += 1
            
            # Plusz két drágább listing
            for idx in [1, 2]:
                higher_price = floor_price * (1 + idx * 0.1)
                cursor.execute('''
                    INSERT INTO auctions (id, player_id, price_eur, price_eth, price_type, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    f"list_{idx}_{p_id}",
                    p_id,
                    round(higher_price, 2),
                    round(higher_price / 1800.0, 5),
                    'direct_listing',
                    now.strftime('%Y-%m-%dT%H:%M:%SZ')
                ))
                auction_count += 1
                
        conn.commit()
        conn.close()
        logger.info(f"Sikeresen legeneráltunk {auction_count} áradatot a Sorare játékosokhoz.")

if __name__ == "__main__":
    seeder = DatabaseSeeder()
    seeder.seed_prices()

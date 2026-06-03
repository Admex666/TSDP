import logging
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.api_client import SorareApiClient
from core.historical_collector import HistoricalDataCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SorareBulkScouter:
    def __init__(self):
        self.api_client = SorareApiClient()
        self.collector = HistoricalDataCollector()

    def fetch_club_player_slugs(self, club_slug):
        """
        Lekéri egy klub összes aktív játékosának slug-ját a Sorare API-ról.
        """
        query = """
        query GetClubPlayers($slug: String!) {
          football {
            club(slug: $slug) {
              name
              activePlayers {
                nodes {
                  slug
                }
              }
            }
          }
        }
        """
        try:
            variables = {"slug": club_slug}
            result = self.api_client.execute_query(query, variables)
            
            if "errors" in result:
                logger.error(f"GraphQL Hiba a klub ({club_slug}) lekérésekor: {result['errors']}")
                return []
                
            club_data = result.get("data", {}).get("football", {}).get("club")
            if not club_data:
                logger.warning(f"Klub nem található: {club_slug}")
                return []
                
            club_name = club_data.get("name")
            player_nodes = club_data.get("activePlayers", {}).get("nodes", [])
            
            slugs = [p.get("slug") for p in player_nodes if p and p.get("slug")]
            logger.info(f"Sikeresen lekérve {len(slugs)} játékos a {club_name} ({club_slug}) keretéből.")
            return slugs
            
        except Exception as e:
            logger.error(f"Hiba a klub ({club_slug}) keretének letöltésekor: {e}")
            return []

    def run_bulk_scouting(self, club_slugs):
        """
        Végigmegy a megadott klubokon, összegyűjti az összes játékos slug-ját, 
        majd tömegesen letölti a statisztikáikat és valós piaci áraikat.
        """
        all_slugs = []
        logger.info(f"Tömeges scouting indítása a következő klubokra: {club_slugs}")
        
        for club_slug in club_slugs:
            slugs = self.fetch_club_player_slugs(club_slug)
            all_slugs.extend(slugs)
            time.sleep(1) # Kíméletes lekérdezés a rate limit miatt
            
        # Duplikációk kiszűrése (biztonsági okokból)
        all_slugs = list(set(all_slugs))
        logger.info(f"Összesen {len(all_slugs)} egyedi játékos slug-ot gyűjtöttünk össze.")
        
        if not all_slugs:
            logger.warning("Nem találtunk játékosokat a scouting során.")
            return
            
        # Játékosok tömeges letöltése a gyűjtővel (4-es chunk mérettel az API korlátai miatt)
        logger.info("Játékosok adatainak, meccseinek és valós árainak letöltése az adatbázisba...")
        self.collector.fetch_and_save_players(all_slugs, chunk_size=4)
        logger.info("Tömeges scouting sikeresen befejeződött!")

if __name__ == "__main__":
    scouter = SorareBulkScouter()
    
    # 5 darab kiemelt klub (magyar és európai élcsapatok), ami összesen kb. 150-200 játékost jelent
    target_clubs = [
        "real-madrid-madrid",
        "fc-barcelona",
        "bayern-munchen-munchen",
        "ferencvarosi-tc"
    ]
    
    # Lefuttatjuk a tömeges scoutingot
    scouter.run_bulk_scouting(target_clubs)

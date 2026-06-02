import pandas as pd
import logging
from api_client import SorareApiClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_and_save_players(slugs, output_file="players_test.csv"):
    """
    Lekéri a megadott játékosok adatait és elmenti őket egy CSV fájlba.
    """
    client = SorareApiClient()
    
    # GraphQL lekérdezés több játékosra
    query = """
    query GetPlayers($slugs: [String!]!) {
      football {
        players(slugs: $slugs) {
          id
          displayName
          age
          position
          activeClub {
            name
          }
        }
      }
    }
    """
    
    variables = {"slugs": slugs}
    
    try:
        logger.info(f"Adatok lekérése a következő játékosokhoz: {slugs}")
        result = client.execute_query(query, variables)
        
        # Ellenőrizzük, hogy van-e hiba a válaszban (pl. GraphQL szintaktika)
        if "errors" in result:
            logger.error(f"GraphQL Hiba: {result['errors']}")
            return
        
        players_data = result.get("data", {}).get("football", {}).get("players", [])
        
        if not players_data:
            logger.warning("Nem érkezett játékos adat a szervertől.")
            return
            
        # Adatok formázása Pandas DataFrame-hez
        formatted_data = []
        for p in players_data:
            if p: # Lehet null, ha a slug nem létezik
                formatted_data.append({
                    "id": p.get("id"),
                    "name": p.get("displayName"),
                    "age": p.get("age"),
                    "position": p.get("position"),
                    "club": p.get("activeClub", {}).get("name") if p.get("activeClub") else "Unknown"
                })
        
        # DataFrame létrehozása és mentése
        df = pd.DataFrame(formatted_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Sikeresen elmentve {len(df)} játékos adatai a {output_file} fájlba.")
        
    except Exception as e:
        logger.error(f"Hiba az adatok lekérése vagy mentése közben: {e}")

if __name__ == "__main__":
    # Teszt: 3 ismert játékos adatainak lekérése
    test_slugs = [
        "lionel-andres-messi-cuccittini",
        "cristiano-ronaldo-dos-santos-aveiro",
        "kylian-mbappe-lottin"
    ]
    fetch_and_save_players(test_slugs)

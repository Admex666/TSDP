import os
import requests
import json
import time
from dotenv import load_dotenv
import logging

# Logger beállítása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SorareApiClient:
    """
    Kliens osztály a Sorare GraphQL API-val való kommunikációhoz.
    A GUIDELINES.md alapján biztosítjuk a biztonságos adatkezelést,
    a rate limit hibakezelést és az egyértelmű hibaüzeneteket.
    """
    API_URL = "https://api.sorare.com/graphql"

    def __init__(self):
        # .env fájl betöltése a jelszavak/kulcsok elrejtéséhez (Biztonsági szabály)
        load_dotenv()
        self.api_key = os.getenv("SORARE_API_KEY")
        
        if not self.api_key or self.api_key == "ide_jon_a_sorare_api_kulcs":
            logger.warning("Nincs beállítva SORARE_API_KEY a .env fájlban. Lehet, hogy egyes kérések el lesznek utasítva.")
        
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key and self.api_key != "ide_jon_a_sorare_api_kulcs":
            # API kulcs beállítása a headerben
            self.headers["APIKEY"] = self.api_key

    def execute_query(self, query: str, variables: dict = None) -> dict:
        """
        Végrehajt egy GraphQL lekérdezést a Sorare API-n.
        
        :param query: A GraphQL query string formában.
        :param variables: A lekérdezés változói dictionary-ként.
        :return: A válasz JSON tartalma dictionary-ként.
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.API_URL, 
                    json=payload, 
                    headers=self.headers,
                    timeout=10
                )
                
                # Ha rate limit-be ütköztünk (429)
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Rate limit elérve (429). Várakozás {wait_time} másodpercig...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status() 
                return response.json()
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Hiba történt a GraphQL lekérdezés során: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Szerver válasza: {e.response.text}")
                
                if attempt == max_retries - 1:
                    raise
                
                logger.info(f"Újrapróbálkozás ({attempt + 1}/{max_retries})...")
                time.sleep(2)

if __name__ == "__main__":
    # Egyszerű teszt, hogy működik-e a kliens
    client = SorareApiClient()
    
    # Teszt lekérdezés: Egy konkrét játékos (pl. Lionel Messi) alapadatainak lekérése
    # A slug a játékos neve alapján generálódik a Sorare-en
    test_query = """
    query GetPlayer($slug: String!) {
      football {
        player(slug: $slug) {
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
    
    variables = {"slug": "lionel-andres-messi-cuccittini"}
    
    try:
        logger.info("Teszt lekérdezés indítása...")
        result = client.execute_query(test_query, variables)
        logger.info(f"Eredmény: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error("A teszt lekérdezés sikertelen volt.")

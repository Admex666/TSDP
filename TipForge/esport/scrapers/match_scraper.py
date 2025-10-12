"""
Match results scraper egy event-hez.
"""

from .base_scraper import BaseScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MatchScraper(BaseScraper):
    """Egy event összes meccsének scrape-elése."""
    
    def scrape_event_matches(self, event_id: str) -> pd.DataFrame:
        """
        Egy event összes meccsének lekérése.
        
        Args:
            event_id: HLTV event ID (pl. "7902")
        
        Returns:
            DataFrame: match_id, date, team_home, team_home_id, team_away, team_away_id, 
                      score_home, score_away, map_type, rounds, link
        """
        
        def _scrape():
            url = f"https://www.hltv.org/results?event={event_id}"
            self._init_driver()
            self.driver.get(url)
            
            logger.info(f"📊 Meccsek scrape-elése: event_id={event_id}")
            
            wait = WebDriverWait(self.driver, 15)
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "results-sublist")))
            
            self._random_delay()
            
            results_data = []
            sublists = self.driver.find_elements(By.CLASS_NAME, "results-sublist")
            
            for sublist in sublists:
                try:
                    # Mérkőzés dátuma
                    match_date = sublist.find_element(By.CLASS_NAME, "standard-headline").text.strip()
                    
                    # Az összes mérkőzés az adott dátumban
                    matches = sublist.find_elements(By.CSS_SELECTOR, ".result-con a")
                    
                    for match in matches:
                        try:
                            link = match.get_attribute("href").strip()
                            match_id = link.split('/')[4]
                            
                            # Csapatok neve
                            team_home = match.find_element(By.CSS_SELECTOR, ".team1 .team").text.strip()
                            team_away = match.find_element(By.CSS_SELECTOR, ".team2 .team").text.strip()
                            
                            # Pontok a sorrend alapján
                            score_spans = match.find_elements(By.CSS_SELECTOR, ".result-score span")
                            score_home = int(score_spans[0].text.strip())
                            score_away = int(score_spans[1].text.strip())
                            
                            # Map típus
                            map_type = match.find_element(By.CSS_SELECTOR, ".map-and-stars .map-text").text.strip()
                            
                            # Rounds (bo1/bo3/bo5)
                            rounds = int(map_type[-1]) if map_type.startswith("bo") else 1
                            
                            results_data.append({
                                "match_id": match_id,
                                "event_id": event_id,
                                "date": match_date,
                                "team_home": team_home,
                                "team_away": team_away,
                                "score_home": score_home,
                                "score_away": score_away,
                                "map_type": map_type,
                                "rounds": rounds,
                                "link": link
                            })
                            
                            logger.debug(f"  ✅ {team_home} vs {team_away} ({match_id})")
                            
                        except Exception as e_match:
                            logger.warning(f"  ⚠️ Mérkőzés hiba: {e_match}")
                            continue
                            
                except Exception as e_sublist:
                    logger.warning(f"⚠️ Dátum hiba: {e_sublist}")
                    continue
            
            logger.info(f"✅ {len(results_data)} meccs scrape-elve")
            return pd.DataFrame(results_data)
        
        result = self._retry_scrape(_scrape)
        self.close()
        return result if result is not None else pd.DataFrame()
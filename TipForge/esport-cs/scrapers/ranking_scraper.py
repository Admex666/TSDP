"""
HLTV team rankings scraper.
"""

from .base_scraper import BaseScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RankingScraper(BaseScraper):
    """HLTV rangsor scrape-elése."""
    
    def scrape_rankings(self) -> pd.DataFrame:
        """
        Aktuális HLTV team rankings scrape-elése.
        
        Returns:
            DataFrame: date, rank, team_id, team_name, points, profile_link
        """
        
        def _scrape():
            self._init_driver()
            self.driver.get("https://www.hltv.org/ranking/teams/")
            
            logger.info(f"🏆 Rankings scraping...")
            
            # Várj, amíg betölt a lista
            wait = WebDriverWait(self.driver, 15)
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".ranked-team.standard-box")))
            
            self._random_delay()
            
            rankings = []
            rank_divs = self.driver.find_elements(By.CSS_SELECTOR, ".ranked-team.standard-box")
            
            for rank_div in rank_divs:
                try:
                    rank = rank_div.find_element(By.CLASS_NAME, "position").text
                    team_name = rank_div.find_element(By.CLASS_NAME, "name").text
                    points = rank_div.find_element(By.CLASS_NAME, "points").text.replace('(', '').replace(')', '').replace(" HLTV points", "")
                    
                    team_link = rank_div.find_element(By.TAG_NAME, "a").get_attribute("href")
                    team_id = team_link.split('/')[-2]
                    profile_link = rank_div.find_element(By.CLASS_NAME, "moreLink").get_attribute("href")
                    
                    rankings.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'rank': int(rank.replace('#', '')),
                        'team_id': team_id,
                        'team_name': team_name,
                        'points': int(points),
                        'profile_link': profile_link
                    })
                except Exception as e:
                    logger.warning(f"  ⚠️ Ranking hiba: {e}")
                    continue
            
            logger.info(f"✅ {len(rankings)} ranking scrape-elve")
            return pd.DataFrame(rankings)
        
        result = self._retry_scrape(_scrape)
        self.close()
        return result if result is not None else pd.DataFrame()
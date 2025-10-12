"""
Event archive scraper - Major events metadata és URL-ek.
"""

from .base_scraper import BaseScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class EventScraper(BaseScraper):
    """Event archive scraper - kinyeri az összes major event URL-t és metadata-t."""
    
    def scrape_event_archive(self, archive_url: str = "https://www.hltv.org/events/archive?eventType=MAJOR") -> pd.DataFrame:
        """
        Event archive scrape-elése - összes major event.
        
        Args:
            archive_url: HLTV events archive URL
        
        Returns:
            DataFrame: month, event_name, event_id, teams, prize, link
        """
        
        def _scrape():
            self._init_driver()
            self.driver.get(archive_url)
            
            logger.info(f"📋 Event archive scraping: {archive_url}")
            
            # Várunk, amíg betöltődnek az események hónapjai
            wait = WebDriverWait(self.driver, 15)
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "events-month")))
            
            self._random_delay()
            
            events_data = []
            months_divs = self.driver.find_elements(By.CLASS_NAME, "events-month")
            
            logger.info(f"🔍 Talált hónapok: {len(months_divs)}")
            
            for month_div in months_divs:
                try:
                    # Hónap
                    month_name = month_div.find_element(By.CLASS_NAME, "standard-headline").text.strip()
                    
                    # Minden esemény az adott hónapban
                    event_links = month_div.find_elements(By.CSS_SELECTOR, "a.small-event.standard-box")
                    
                    for event in event_links:
                        try:
                            event_name = event.find_element(By.CSS_SELECTOR, ".event-col .text-ellipsis").text.strip()
                            
                            # Teams count
                            team_cells = event.find_elements(By.CSS_SELECTOR, ".table tr:first-child td.small-col")
                            team_count = team_cells[0].text.strip() if team_cells else "N/A"
                            
                            # Prize pool
                            prize_cells = event.find_elements(By.CSS_SELECTOR, ".table tr:first-child td.prizePoolEllipsis")
                            prize = prize_cells[0].get_attribute("title").strip() if prize_cells else "N/A"
                            
                            # Link és event_id
                            link = event.get_attribute("href").strip()
                            event_id = link.split('/')[4]
                            
                            events_data.append({
                                "month": month_name,
                                "event_name": event_name,
                                "event_id": event_id,
                                "teams": team_count,
                                "prize": prize,
                                "link": link
                            })
                            
                            logger.debug(f"  ✅ {event_name} ({event_id})")
                            
                        except Exception as e_event:
                            logger.warning(f"  ⚠️ Esemény hiba: {e_event}")
                            continue
                            
                except Exception as e_month:
                    logger.warning(f"⚠️ Hónap hiba: {e_month}")
                    continue
            
            logger.info(f"✅ {len(events_data)} event scrape-elve")
            return pd.DataFrame(events_data)
        
        result = self._retry_scrape(_scrape)
        self.close()
        return result if result is not None else pd.DataFrame()
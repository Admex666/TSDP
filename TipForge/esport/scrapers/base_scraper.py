"""
Alap Selenium scraper osztály közös funkcionalitással.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import random
import logging
from utils.config import *

logger = logging.getLogger(__name__)


class BaseScraper:
    """Alaposztály minden scraperhez."""
    
    def __init__(self, headless: bool = HEADLESS_MODE):
        self.headless = headless
        self.driver = None
    
    def _init_driver(self):
        """Selenium driver inicializálása."""
        try:
            options = Options()
            if self.headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            logger.debug("✅ Selenium driver inicializálva")
        except Exception as e:
            logger.error(f"❌ Driver inicializálási hiba: {e}")
            raise
    
    def _random_delay(self):
        """Véletlen delay rate limiting miatt."""
        delay = random.uniform(SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX)
        time.sleep(delay)
    
    def _retry_scrape(self, scrape_func, *args, **kwargs):
        """
        Retry mechanizmus scrape függvényekhez.
        
        Args:
            scrape_func: A scrape függvény
            *args, **kwargs: A függvény paraméterei
        
        Returns:
            A függvény visszatérési értéke vagy None hiba esetén
        """
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                result = scrape_func(*args, **kwargs)
                return result
            except TimeoutException:
                logger.warning(f"⚠️ Timeout (próbálkozás {attempt}/{RETRY_ATTEMPTS})")
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(3 * attempt)  # Exponenciális backoff
                else:
                    logger.error(f"❌ Végleg timeout: {scrape_func.__name__}")
                    return None
            except WebDriverException as e:
                logger.error(f"❌ WebDriver hiba: {e}")
                if attempt < RETRY_ATTEMPTS:
                    self._restart_driver()
                    time.sleep(3 * attempt)
                else:
                    return None
            except Exception as e:
                logger.error(f"❌ Általános hiba: {e}")
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(3 * attempt)
                else:
                    return None
        
        return None
    
    def _restart_driver(self):
        """Driver újraindítása hiba esetén."""
        try:
            if self.driver:
                self.driver.quit()
        except:
            pass
        
        logger.info("🔄 Driver újraindítása...")
        self._init_driver()
    
    def close(self):
        """Driver bezárása."""
        if self.driver:
            try:
                self.driver.quit()
                logger.debug("Driver bezárva")
            except:
                pass
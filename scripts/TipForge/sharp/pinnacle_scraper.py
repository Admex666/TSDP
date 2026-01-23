from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PinnacleScraper:
    """Scrapes odds from Pinnacle"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        
    def _setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        # Anti-bot measures
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        return webdriver.Chrome(options=chrome_options)

    def scrape_matches(self, url: str) -> List[Dict]:
        """
        Scrapes multiple match odds from a Pinnacle matchups page.
        """
        driver = None
        matches = []
        try:
            logger.info(f"Opening Pinnacle URL: {url}")
            driver = self._setup_driver()
            
            # Execute some CDP commands to hide Selenium presence
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            })
            
            driver.get(url)
            
            # Wait for match rows to appear
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[class*="row-"]')))
            
            # Scroll a bit to trigger lazy loading if any
            driver.execute_script("window.scrollTo(0, 1000);")
            time.sleep(2)
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Find match rows
            rows = soup.find_all('div', class_=lambda x: x and 'row-' in x)
            logger.info(f"Found {len(rows)} potential match rows on Pinnacle")
            
            for row in rows:
                try:
                    # Team names: spans with class gameInfoLabel- Inside <a>
                    team_spans = row.find_all('span', class_=lambda x: x and 'gameInfoLabel-' in x)
                    if len(team_spans) < 2:
                        continue
                        
                    home_team = team_spans[0].text.strip()
                    away_team = team_spans[1].text.strip()
                    
                    # User Request: Only process if "(Match)" is present FOR ESPORTS
                    # Tennis matches typically don't have this suffix.
                    if "esports" in url and "(Match)" not in home_team and "(Match)" not in away_team:
                        continue
                    
                    # Clean team names (remove "(Match)" suffix often found on Pinnacle)
                    home_team = home_team.split(' (')[0]
                    away_team = away_team.split(' (')[0]
                    
                    # Odds: buttons with class market-btn without title attribute
                    # These are usually 1X2 for soccer
                    odds_btns = row.find_all('button', class_='market-btn')
                    odds_values = []
                    
                    for btn in odds_btns:
                        # 1X2 buttons usually don't have a title (Handicap/Total have titles like '0' or '2.5')
                        if not btn.has_attr('title'):
                            odds_text = btn.text.strip()
                            if odds_text:
                                try:
                                    # Odds might be in a nested span or just text
                                    # Sometimes formatted as "2.580"
                                    val = float(odds_text.split('\n')[-1]) # Take the last line if multi-line
                                    odds_values.append(val)
                                except:
                                    continue
                                    
                    # We expect 3 odds for 1X2, or 2 for 2-way sports
                    if len(odds_values) >= 2:
                        matches.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'odds': odds_values,
                            'source': 'pinnacle'
                        })
                        
                except Exception as row_e:
                    logger.debug(f"Pinnacle row error: {row_e}")
                    continue
                    
            return matches
            
        except Exception as e:
            logger.error(f"Pinnacle scraping error: {e}")
            return []
        finally:
            if driver:
                driver.quit()

if __name__ == "__main__":
    # Test
    scraper = PinnacleScraper(headless=False)
    test_url = "https://www.pinnacle.com/en/soccer/england-premier-league/matchups/"
    results = scraper.scrape_matches(test_url)
    for res in results:
        print(f"{res['home_team']} vs {res['away_team']}: {res['odds']}")

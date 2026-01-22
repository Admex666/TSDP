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

class TippmixScraper:
    """Scrapes odds from TippmixPro"""
    
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
        # Add some stealth-ish headers
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        return webdriver.Chrome(options=chrome_options)

    def scrape_matches(self, url: str) -> List[Dict]:
        """
        Scrapes multiple match odds from a Tippmix category/league page.
        """
        driver = None
        matches = []
        try:
            logger.info(f"Opening Tippmix URL: {url}")
            driver = self._setup_driver()
            driver.get(url)
            
            # Handle cookies
            try:
                cookie_btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                )
                cookie_btn.click()
                time.sleep(1)
            except:
                pass

            # Switch to iframe
            try:
                logger.info("Waiting for #SportsIframe...")
                wait = WebDriverWait(driver, 20)
                wait.until(EC.presence_of_element_located((By.ID, "SportsIframe")))
                iframe = driver.find_element(By.ID, "SportsIframe")
                driver.switch_to.frame(iframe)
                logger.info("Switched to SportsIframe")
                
                # Wait for event items inside iframe
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "EventItem")))
                time.sleep(2) # Give it extra time to render odds
            except Exception as e:
                logger.warning(f"Iframe/Content issue: {e}")
                # Sometimes navigating directly to the iframe source works better if it's cross-origin
                pass

            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            event_rows = soup.find_all('div', class_='EventItem')
            logger.info(f"Found {len(event_rows)} event rows (.EventItem)")
            
            for row in event_rows:
                try:
                    # Team names - can be span or div
                    names = row.find_all(class_='Details__ParticipantName')
                    if len(names) < 2:
                        continue
                        
                    home_team = names[0].get_text(strip=True)
                    away_team = names[1].get_text(strip=True)
                    
                    # Odds Buttons - Find the group containing the 1X2 odds (usually the one with 3 buttons)
                    odds_groups = row.find_all('div', class_='EventItem__ButtonsGroup')
                    odds_values = []
                    
                    for group in odds_groups:
                        btns = group.find_all('button', class_='OddsButton')
                        if len(btns) == 3: # Likely 1X2
                            for btn in btns:
                                odds_val_el = btn.find('span', class_='OddsButton__Odds')
                                if odds_val_el:
                                    try:
                                        val_str = odds_val_el.get_text(strip=True).replace(',', '.')
                                        if val_str:
                                            odds_values.append(float(val_str))
                                    except:
                                        continue
                            if len(odds_values) == 3:
                                break # Found our 1X2
                    
                    # Fallback for 2-way markets if no 3nd exists
                    if not odds_values:
                        for group in odds_groups:
                            btns = group.find_all('button', class_='OddsButton')
                            if len(btns) == 2:
                                for btn in btns:
                                    odds_val_el = btn.find('span', class_='OddsButton__Odds')
                                    if odds_val_el:
                                        try:
                                            val_str = odds_val_el.get_text(strip=True).replace(',', '.')
                                            odds_values.append(float(val_str))
                                        except:
                                            continue
                                if len(odds_values) == 2:
                                    break
                    
                    if home_team and away_team and len(odds_values) >= 2:
                        matches.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'odds': odds_values,
                            'source': 'tippmix'
                        })
                except Exception as row_e:
                    logger.debug(f"Row error: {row_e}")
                    continue

            return matches


            
        except Exception as e:
            logger.error(f"Tippmix scraping error: {e}")
            return []
        finally:
            if driver:
                driver.quit()

if __name__ == "__main__":
    # Test with a sample URL
    scraper = TippmixScraper(headless=False)
    # Sample: Football -> Premier League
    test_url = "https://www.tippmixpro.hu/hu/fogadas/labdarugas/anglia/premier-league"
    results = scraper.scrape_matches(test_url)
    for res in results:
        print(f"{res['home_team']} vs {res['away_team']}: {res['odds']}")

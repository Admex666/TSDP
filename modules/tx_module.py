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

def _setup_driver(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        return webdriver.Chrome(options=chrome_options)
    
def scrape(self, url: str, game_index: Optional[int] = None) -> Optional[Dict]:
    """
    Scrape betting odds from Tippmix
    
    Args:
        url: Tippmix URL
        game_index: Optional game number to filter (1, 2, 3, etc.)
    
    Returns:
        Dict with structure:
        {
            'timestamp': str,
            'markets': [
                {
                    'name': str,
                    'game_index': int,  # ÚJ!
                    'options': [...]
                },
                ...
            ]
        }
    """
    driver = None
    try:
        driver = self._setup_driver()
        driver.get(url)
        
        # Handle cookie consent
        try:
            cookie_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            cookie_btn.click()
            time.sleep(1)
        except:
            pass
        
        # Switch to iframe
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "iframe"))
        )
        iframe = driver.find_elements(By.TAG_NAME, "iframe")[0]
        driver.switch_to.frame(iframe)
        time.sleep(2)
        
        # Extract markets
        articles = driver.find_elements(By.TAG_NAME, "article")
        markets = []
        
        for article in articles:
            try:
                # Market name
                legend_el = article.find_element(By.CLASS_NAME, "Market__CollapseText")
                market_name = legend_el.get_attribute("title") or legend_el.text.strip()
                
                # Extract game index from market name
                market_game_index = self._extract_game_index_from_market(market_name)
                
                # Filter by game_index if specified
                if game_index is not None and market_game_index != game_index:
                    continue
                
                # Odds buttons
                odds_buttons = article.find_elements(By.CLASS_NAME, "OddsButton")
                options = []
                
                for btn in odds_buttons:
                    try:
                        name_el = btn.find_element(By.CLASS_NAME, "OddsButton__Text")
                        odds_el = btn.find_element(By.CLASS_NAME, "OddsButton__Odds")
                        
                        option_name = name_el.text.strip()
                        odds_str = odds_el.text.strip().replace(',', '.')
                        odds_value = float(odds_str)
                        
                        options.append({
                            'name': option_name,
                            'odds': odds_value
                        })
                    except:
                        continue
                
                if options:
                    markets.append({
                        'name': market_name,
                        'game_index': market_game_index,  # ÚJ!
                        'options': options
                    })
                    
            except:
                continue
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'markets': markets
        }
        
        filter_msg = f" (filtered to game {game_index})" if game_index else ""
        return result
        
    except Exception as e:
        return None
    finally:
        if driver:
            driver.quit()
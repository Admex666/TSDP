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
    
def scrape(url, headless=False):
    """
    Scrape betting odds from Tippmix
    
    Args:
        url: Tippmix URL
    
    Returns:
        Dict with structure:
        {
            'timestamp': str,
            'markets': [
                {
                    'name': str,
                },
                ...
            ]
        }
    """

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    

    try:
        driver = webdriver.Chrome(options=chrome_options)
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

        # Az articles keresés előtt várj egy specifikus elemre
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "Market__CollapseText"))
        )
                
        # Extract markets
        articles = driver.find_elements(By.TAG_NAME, "article")
        markets = []
        
        for article in articles:
            try:
                # Market name
                legend_el = article.find_element(By.CLASS_NAME, "Market__CollapseText")
                market_name = legend_el.get_attribute("title") or legend_el.text.strip()
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
                        'options': options
                    })
                    
            except:
                continue
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'markets': markets
        }
        
        return result
        
    except Exception as e:
        return None
    finally:
        if driver:
            driver.quit()


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import pandas as pd

def get_league_odds(url, headless=False):
    """
    Kinyeri a meccs adatokat a tippmixpro oldalról.
    
    Args:
        driver: Selenium WebDriver instance
        url: Az oldal URL-je
    
    Returns:
        Lista dictionary-kből a meccs adatokkal
    """

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)

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
    
    # Handle iframes
    WebDriverWait(driver, 10).until(
       EC.presence_of_element_located((By.ID, "SportsIframe"))
    )
    iframe = driver.find_element(By.ID, "SportsIframe")
    driver.switch_to.frame(iframe)
    
    # Várunk amíg betöltődnek az EventItem elemek
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "EventItem"))
    )
    
    matches = []
    current_year = datetime.now().year
    
    event_items = driver.find_elements(By.CLASS_NAME, "EventItem")

    for item in event_items:
        try:
            # Dátum kinyerése (MM.dd formátumból YYYY-MM-dd-be)
            date_elem = item.find_element(By.CSS_SELECTOR, ".MatchTime__InfoPart--Date")
            date_raw = date_elem.text.strip()  # pl. "11.22"
            
            # Átalakítás YYYY-MM-dd formátumra
            month, day = date_raw.rstrip('.').split('.')
            date_formatted = f"{current_year}-{month.zfill(2)}-{day.zfill(2)}"
            # Csapatok kinyerése
            participants = item.find_elements(By.CSS_SELECTOR, ".Details__ParticipantName")
            home_team = participants[0].text.strip()
            away_team = participants[1].text.strip()
            
            # Oddsok kinyerése
            odds_buttons = item.find_elements(By.CSS_SELECTOR, ".OddsButton")
            home_odds = None
            away_odds = None
            
            for btn in odds_buttons:
                short_text = btn.find_element(By.CSS_SELECTOR, ".OddsButton__ShortText").text.strip()
                odds_value = btn.find_element(By.CSS_SELECTOR, ".OddsButton__Odds").text.strip()
                # Vessző cseréje pontra a float konverzióhoz
                odds_value = odds_value.replace(',', '.')
                
                if short_text == "Hazai":
                    home_odds = float(odds_value)
                elif short_text == "Vendég":
                    away_odds = float(odds_value)
            
            match_data = {
                "date": date_formatted,
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "away_odds": away_odds
            }
            
            matches.append(match_data)
            
        except Exception as e:
            print(f"Hiba egy meccs feldolgozásánál: {e}")
            continue

    driver.quit()
    
    return pd.DataFrame(matches)
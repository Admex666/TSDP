"""
League of Legends Live Match & Odds Scrapers
Combines AndyDanger match stats and Tippmix odds scraping
"""

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


class MatchStatsScraper:
    """Scrapes live match statistics from AndyDanger"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        
    def _setup_driver(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        return webdriver.Chrome(options=chrome_options)
    
    def scrape(self, url: str) -> Optional[Dict]:
        """
        Scrape match statistics from AndyDanger
        
        Returns:
            Dict with structure:
            {
                'timestamp': str,
                'game_time': str (e.g., '29:00'),
                'blue_team': {...},
                'red_team': {...},
                'players': [...]
            }
        """
        driver = None
        try:
            logger.info("Starting match stats scraper...")
            driver = self._setup_driver()
            driver.get(url)
            
            # Wait for content
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located(
                (By.CLASS_NAME, "status-live-game-card-content")
            ))
            time.sleep(3)
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract data
            result = {
                'timestamp': datetime.now().isoformat(),
                'game_time': self._extract_game_time(soup),
                'blue_team': {},
                'red_team': {},
                'players': []
            }
            
            team_stats = self._extract_team_stats(soup)
            result['blue_team'] = team_stats['blue']
            result['red_team'] = team_stats['red']
            result['players'] = self._extract_player_stats(soup)
            
            logger.info(f"✅ Successfully scraped match at {result['game_time']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Scraping error: {str(e)}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def _extract_game_time(self, soup) -> str:
        """Extract current game time"""
        try:
            gamestate_div = soup.find('div', class_=lambda x: x and x.startswith('gamestate-'))
            if gamestate_div:
                time_div = gamestate_div.find_next_sibling('div')
                return time_div.get_text(strip=True) if time_div else "Unknown"
        except:
            pass
        return "Unknown"
    
    def _extract_team_stats(self, soup) -> Dict:
        """Extract team-level statistics"""
        stats = {'blue': {}, 'red': {}}
        
        for team_name, team_class in [('blue', 'blue-team'), ('red', 'red-team')]:
            team_div = soup.find('div', class_=team_class)
            if team_div:
                stats[team_name] = {
                    'kills': self._extract_stat_value(team_div, 'kills'),
                    'towers': self._extract_stat_value(team_div, 'towers'),
                    'inhibitors': self._extract_stat_value(team_div, 'inhibitors'),
                    'barons': self._extract_stat_value(team_div, 'barons'),
                    'gold': self._extract_gold_value(team_div),
                    'dragons': self._extract_dragons(team_div)
                }
        
        return stats
    
    def _extract_stat_value(self, team_div, stat_class: str) -> int:
        """Extract a single stat value"""
        try:
            stat_div = team_div.find('div', class_=f'team-stats {stat_class}')
            if stat_div:
                children = list(stat_div.children)
                if children:
                    last_child = children[-1]
                    value = last_child.strip() if last_child.name is None else last_child.get_text(strip=True)
                    return int(value)
        except:
            pass
        return 0
    
    def _extract_gold_value(self, team_div) -> int:
        """Extract gold value (has special formatting)"""
        try:
            gold_div = team_div.find('div', class_='team-stats gold')
            if gold_div:
                span = gold_div.find('span')
                if span:
                    gold_str = span.get_text(strip=True).replace(',', '')
                    return int(gold_str)
        except:
            pass
        return 0
    
    def _extract_dragons(self, team_div) -> List[str]:
        """Extract dragon types"""
        dragons = []
        dragon_svgs = team_div.find_all('svg', class_='dragon')
        
        dragon_map = {
            '#A8805D': 'Earth',
            '#67C4B0': 'Ocean',
            '#ADD2ED': 'Cloud',
            '#F0BE1A': 'Infernal',
            '#8C52FF': 'Hextech',
            '#5CD6A9': 'Chemtech'
        }
        
        for svg in dragon_svgs:
            path = svg.find('path', class_='shape')
            if path:
                fill_color = path.get('fill', '')
                dragon_type = dragon_map.get(fill_color, f'Unknown')
                dragons.append(dragon_type)
        
        return dragons
    
    def _extract_player_stats(self, soup) -> List[Dict]:
        """Extract player-level statistics"""
        players = []
        player_rows = soup.find_all('tr', class_='player-stats-row')
        
        for row in player_rows:
            player_data = self._extract_player_row(row)
            if player_data:
                players.append(player_data)
        
        return players
    
    def _extract_player_row(self, row) -> Optional[Dict]:
        """Extract data from a single player row"""
        try:
            champion_info = row.find('div', class_='player-champion-info')
            if not champion_info:
                return None
            
            # Champion name
            champion_name_spans = champion_info.find_all('span')
            champion_name = "Unknown"
            for span in champion_name_spans:
                text = span.get_text(strip=True)
                if text and text not in ['', 'T1', 'TOPESPORTS', 'TES']:
                    champion_name = text
                    break
            
            # Player name
            player_name_elem = champion_info.find('span', class_='player-card-player-name')
            player_name = player_name_elem.get_text(strip=True) if player_name_elem else "Unknown"
            
            # Level
            level_elem = champion_info.find('span', class_='player-champion-info-level')
            level = int(level_elem.get_text(strip=True)) if level_elem else 0
            
            # CS (creep score)
            columns = row.find_all('td')
            cs = 0
            gold = 0
            kills = 0
            deaths = 0
            assists = 0
            
            for column in columns:
                text = column.get_text(strip=True)
                
                # Detect column type by content
                if 'player-stats-kda' in str(column):
                    if kills == 0:
                        kills = int(text) if text.isdigit() else 0
                    elif deaths == 0:
                        deaths = int(text) if text.isdigit() else 0
                    elif assists == 0:
                        assists = int(text) if text.isdigit() else 0
                elif ',' in text and text.replace(',', '').isdigit():
                    gold = int(text.replace(',', ''))
                elif text.isdigit() and len(text) <= 4:
                    if cs == 0:
                        cs = int(text)
            
            return {
                'champion': champion_name,
                'player_name': player_name,
                'level': level,
                'cs': cs,
                'gold': gold,
                'kills': kills,
                'deaths': deaths,
                'assists': assists
            }
            
        except Exception as e:
            logger.warning(f"Error extracting player data: {e}")
            return None


class OddsScraper:
    """Scrapes live betting odds from Tippmix"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
    
    def _setup_driver(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        return webdriver.Chrome(options=chrome_options)
    
    def scrape(self, url: str) -> Optional[Dict]:
        """
        Scrape betting odds from Tippmix
        
        Returns:
            Dict with structure:
            {
                'timestamp': str,
                'markets': [
                    {
                        'name': str,
                        'options': [
                            {'name': str, 'odds': float},
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        driver = None
        try:
            logger.info("Starting odds scraper...")
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
            
            logger.info(f"✅ Successfully scraped {len(markets)} markets")
            return result
            
        except Exception as e:
            logger.error(f"❌ Odds scraping error: {str(e)}")
            return None
        finally:
            if driver:
                driver.quit()


# Convenience functions
def scrape_match_stats(url: str) -> Optional[Dict]:
    """Quick function to scrape match stats"""
    scraper = MatchStatsScraper(headless=True)
    return scraper.scrape(url)


def scrape_odds(url: str) -> Optional[Dict]:
    """Quick function to scrape odds"""
    scraper = OddsScraper(headless=True)
    return scraper.scrape(url)


if __name__ == "__main__":
    # Test scrapers
    match_url = "https://andydanger.github.io/live-lol-esports/#/live/113475871523985235/game-index/3"
    odds_url = "https://www.tippmixpro.hu/hu/fogadas/i/esemenyek/100/league-of-legends-lol/vilag/emea-masters-summer/karmine-corp-blue-los-heretics/284726865528393728/palyak"
    
    print("Testing Match Stats Scraper...")
    stats = scrape_match_stats(match_url)
    if stats:
        print(f"Game time: {stats['game_time']}")
        print(f"Blue kills: {stats['blue_team']['kills']}")
        print(f"Red kills: {stats['red_team']['kills']}")
    
    print("\nTesting Odds Scraper...")
    odds = scrape_odds(odds_url)
    if odds:
        print(f"Markets found: {len(odds['markets'])}")
        if odds['markets']:
            print(f"First market: {odds['markets'][0]['name']}")
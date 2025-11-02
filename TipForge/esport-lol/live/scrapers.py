"""
League of Legends Live Match & Odds Scrapers - PlayWright Edition
Streamlit Cloud optimized
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
import time
from typing import Dict, List, Optional
from datetime import datetime
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ÚJ: Lazy browser installation
def ensure_playwright_installed():
    """Ensure Playwright browsers are installed"""
    try:
        # Check if chromium is already installed
        cache_dir = os.path.expanduser("~/.cache/ms-playwright")
        if os.path.exists(cache_dir):
            chromium_dirs = [d for d in os.listdir(cache_dir) if 'chromium' in d.lower()]
            if chromium_dirs:
                logger.info("✅ PlayWright browsers already installed")
                return True
        
        # Install browsers
        logger.info("🔄 Installing PlayWright browsers (first run)...")
        subprocess.run(
            ["python", "-m", "playwright", "install", "chromium", "--with-deps"],
            check=True,
            capture_output=True
        )
        logger.info("✅ PlayWright browsers installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to install PlayWright browsers: {e}")
        return False

class MatchStatsScraper:
    """Scrapes live match statistics from AndyDanger"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        
    def scrape(self, url: str) -> Optional[Dict]:
        """Scrape match statistics"""
        # ÚJ: Ensure browsers are installed before scraping
        if not ensure_playwright_installed():
            logger.error("❌ Cannot scrape: PlayWright browsers not installed")
            return None
        
        try:
            logger.info("Starting match stats scraper with PlayWright...")
            
            with sync_playwright() as p:
                # Launch with minimal options for Streamlit Cloud
                browser = p.chromium.launch(
                    headless=self.headless,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu'
                    ]
                )
                
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = context.new_page()
                
                # Navigate
                logger.info(f"Navigating to: {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # Wait for content
                page.wait_for_selector(".status-live-game-card-content", timeout=20000)
                time.sleep(3)
                
                # Get HTML
                html = page.content()
                browser.close()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract game index from URL
            game_index = self._extract_game_index(url)
            
            # Extract data
            result = {
                'timestamp': datetime.now().isoformat(),
                'game_time': self._extract_game_time(soup),
                'game_index': game_index,
                'blue_team': {},
                'red_team': {},
                'players': []
            }
            
            team_stats = self._extract_team_stats(soup)
            result['blue_team'] = team_stats['blue']
            result['red_team'] = team_stats['red']
            result['players'] = self._extract_player_stats(soup)
            
            logger.info(f"✅ Successfully scraped match at {result['game_time']} (Game {game_index})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Scraping error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _extract_game_index(self, url: str) -> int:
        """Extract game index from URL"""
        try:
            import re
            match = re.search(r'/game-index/(\d+)', url)
            if match:
                return int(match.group(1))
        except:
            pass
        return 1
    
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
        """Extract gold value"""
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
                dragon_type = dragon_map.get(fill_color, 'Unknown')
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
            
            # CS and other stats
            columns = row.find_all('td')
            cs = 0
            gold = 0
            kills = 0
            deaths = 0
            assists = 0
            
            for column in columns:
                text = column.get_text(strip=True)
                
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
    
    def scrape(self, url: str, game_index: Optional[int] = None) -> Optional[Dict]:
        """Scrape betting odds from Tippmix"""
        # ÚJ: Ensure browsers are installed before scraping
        if not ensure_playwright_installed():
            logger.error("❌ Cannot scrape: PlayWright browsers not installed")
            return None
        
        try:
            logger.info("Starting odds scraper with PlayWright...")
            
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=self.headless,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu'
                    ]
                )
                
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = context.new_page()
                
                # Navigate
                logger.info(f"Navigating to: {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # Handle cookie consent
                try:
                    page.click("#onetrust-accept-btn-handler", timeout=5000)
                    time.sleep(1)
                except:
                    logger.info("No cookie consent button found")
                
                # Wait for iframe
                page.wait_for_selector("iframe", timeout=10000)
                
                # Get iframe locator
                iframe = page.frame_locator("iframe").first
                time.sleep(2)
                
                # Extract markets
                articles = iframe.locator("article").all()
                markets = []
                
                logger.info(f"Found {len(articles)} articles")
                
                for idx, article in enumerate(articles):
                    try:
                        # Market name
                        legend_el = article.locator(".Market__CollapseText").first
                        market_name = legend_el.get_attribute("title") or legend_el.inner_text()
                        
                        # Extract game index
                        market_game_index = self._extract_game_index_from_market(market_name)
                        
                        # Filter by game_index
                        if game_index is not None and market_game_index != game_index:
                            continue
                        
                        # Odds buttons
                        odds_buttons = article.locator(".OddsButton").all()
                        options = []
                        
                        for btn in odds_buttons:
                            try:
                                name_el = btn.locator(".OddsButton__Text").first
                                odds_el = btn.locator(".OddsButton__Odds").first
                                
                                option_name = name_el.inner_text().strip()
                                odds_str = odds_el.inner_text().strip().replace(',', '.')
                                odds_value = float(odds_str)
                                
                                options.append({
                                    'name': option_name,
                                    'odds': odds_value
                                })
                            except Exception as e:
                                logger.warning(f"Error extracting odds button: {e}")
                                continue
                        
                        if options:
                            markets.append({
                                'name': market_name,
                                'game_index': market_game_index,
                                'options': options
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error extracting market {idx}: {e}")
                        continue
                
                browser.close()
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'markets': markets
                }
                
                filter_msg = f" (filtered to game {game_index})" if game_index else ""
                logger.info(f"✅ Successfully scraped {len(markets)} markets{filter_msg}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Odds scraping error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _extract_game_index_from_market(self, market_name: str) -> int:
        """Extract game index from market name"""
        try:
            import re
            match = re.search(r'(\d+)\.\s*pálya', market_name)
            if match:
                return int(match.group(1))
        except:
            pass
        return 0


# Convenience functions
def scrape_match_stats(url: str) -> Optional[Dict]:
    scraper = MatchStatsScraper(headless=True)
    return scraper.scrape(url)


def scrape_odds(url: str, game_index: Optional[int] = None) -> Optional[Dict]:
    scraper = OddsScraper(headless=True)
    return scraper.scrape(url, game_index=game_index)


if __name__ == "__main__":
    # Test
    match_url = "https://andydanger.github.io/live-lol-esports/#/live/113475871523985235/game-index/3"
    odds_url = "https://www.tippmixpro.hu/hu/fogadas/i/esemenyek/100/league-of-legends-lol/vilag/emea-masters-summer/karmine-corp-blue-los-heretics/284726865528393728/palyak"
    
    print("Testing Match Stats...")
    stats = scrape_match_stats(match_url)
    if stats:
        print(f"✅ Game time: {stats['game_time']}")
    
    print("\nTesting Odds...")
    odds = scrape_odds(odds_url, game_index=3)
    if odds:
        print(f"✅ Markets: {len(odds['markets'])}")
import pandas as pd
import requests
import time
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Union
from io import StringIO  # Added for StringIO usage

class FBrefDataLoader:
    """
    Handles loading and caching of FBref data including match schedules, 
    lineups, and player statistics.
    """
    
    BASE_URL = "https://fbref.com/en"
    
    # List of modern User-Agents to rotate
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
    ]
    
    BASE_URL = "https://fbref.com/en"
    
    def __init__(self, data_dir: str = "data", cache_expiry_days: int = 1):
        """
        Initialize the loader with Selenium WebDriver.
        """
        import random
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Add a real user agent
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Initialize driver
        try:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            self.driver.set_page_load_timeout(30)
        except Exception as e:
            print(f"Failed to initialize Chrome Driver: {e}")
            self.driver = None

    def _get_cache_path(self, key: str) -> Path:
        """Generate a file path for caching."""
        clean_key = key.replace("/", "_").replace("?", "_").replace(":", "")
        return self.raw_dir / f"{clean_key}.html"

    def _fetch_url(self, url: str, use_cache: bool = True) -> str:
        """
        Fetch URL content using Selenium.
        """
        import random
        import time
        
        if not self.driver:
            raise RuntimeError("Selenium WebDriver not initialized.")
            
        cache_path = self._get_cache_path(url)
        
        if use_cache and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        
        print(f"Fetching (Selenium): {url}")
        
        # Random delay
        delay = random.uniform(3.0, 6.0)
        time.sleep(delay)
        
        try:
            self.driver.get(url)
            # Short sleep to let JS run
            time.sleep(2)
            
            html = self.driver.page_source
            
            # Basic validation (if title contains "Just a moment", we might be blocked, but headless often passes)
            
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(html)
                
            return html
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            raise e

    def load_match_schedule(self, season: str, competition_id: str, comp_name: str = "Premier-League") -> pd.DataFrame:
        """
        Load match schedule for a specific season and competition.
        """
        url = f"{self.BASE_URL}/comps/{competition_id}/{season}/schedule/{season}-{comp_name}-Scores-and-Fixtures"
        
        html = self._fetch_url(url)
        
        try:
            # Use match='Wk' to identifying schedule table
            dfs = pd.read_html(StringIO(html), match='Wk')
            if not dfs:
                raise ValueError("No schedule table found")
            df = dfs[0]
        except Exception as e:
            print(f"Error parsing schedule table: {e}")
            return pd.DataFrame()
            
        # Filter out spacer rows (where Wk is NaN or "Wk")
        if 'Wk' in df.columns:
            df = df[df['Wk'].notna() & (df['Wk'] != 'Wk')]
            
        # Extract Match Report URLs 
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', attrs={'id': lambda x: x and 'sched' in x})
        
        valid_links = []
        if table and table.tbody:
            for tr in table.tbody.find_all('tr'):
                if 'class' in tr.attrs and 'spacer' in tr.attrs['class']:
                    continue
                if 'class' in tr.attrs and 'thead' in tr.attrs['class']:
                    continue
                    
                report_td = tr.find('td', {'data-stat': 'match_report'})
                if report_td and report_td.find('a'):
                    valid_links.append(report_td.find('a')['href'])
                else:
                    valid_links.append(None)
        
        # Ensure length matches (heuristic check)
        if len(valid_links) == len(df):
            df['match_report_url'] = valid_links
            
        # Standardize columns
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']) 
        
        # Parse Score usually "x-y"
        if 'Score' in df.columns:
            def parse_score(s):
                if pd.isna(s) or ('–' not in str(s) and '-' not in str(s)):
                    return None, None
                try:
                    parts = str(s).replace('–', '-').split('-')
                    return int(parts[0]), int(parts[1])
                except:
                    return None, None
                    
            df[['HomeGoals', 'AwayGoals']] = df['Score'].apply(lambda x: pd.Series(parse_score(x)))
            
        return df

    def load_match_lineups(self, match_url: str) -> Dict:
        """
        Load starting XI for a specific match.
        """
        if match_url.startswith("/"):
             if match_url.startswith("/en") and self.BASE_URL.endswith("/en"):
                full_url = self.BASE_URL[:-3] + match_url
             else:
                full_url = self.BASE_URL + match_url
        else:
            full_url = match_url
            
        html = self._fetch_url(full_url)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        lineups = {'home': [], 'away': []}
        lineup_divs = soup.find_all('div', class_='lineup')
        if len(lineup_divs) < 2:
            print(f"Warning: Could not find 2 lineup divs for {match_url}")
            return lineups
            
        def extract_starters(div) -> List[Dict]:
            players = []
            table = div.find('table')
            if not table: return []
            
            rows = table.find_all('tr')
            is_bench = False
            for row in rows:
                header = row.find('th')
                if header and 'Bench' in header.text:
                    is_bench = True
                    continue
                if is_bench: continue
                    
                name_cell = row.find('a')
                if name_cell and 'href' in name_cell.attrs:
                    player_name = name_cell.text.strip()
                    player_link = name_cell['href']
                    # ID extraction: /en/players/ID/Name
                    try:
                        player_id = player_link.split('/')[3]
                    except:
                        player_id = "unknown"
                    
                    jersey_cell = row.find('td', {'class': 'shirtnumber'})
                    jersey = jersey_cell.text.strip() if jersey_cell else "0"
                    
                    # Try to find position
                    # Usually in a cell with data-stat="position" or just by index?
                    # FBref lineup tables are sometimes tricky.
                    # Let's look for data-stat="position"
                    pos_cell = row.find('td', {'data-stat': 'position'})
                    pos = pos_cell.text.strip() if pos_cell else "Unknown"
                    
                    players.append({
                        'id': player_id,
                        'name': player_name,
                        'link': player_link,
                        'jersey': jersey,
                        'position': pos
                    })
            return players
            
        lineups['home'] = extract_starters(lineup_divs[0])
        lineups['away'] = extract_starters(lineup_divs[1])
        return lineups

    def load_player_match_logs(self, player_id: str, season: str, log_type: str = "summary") -> pd.DataFrame:
        """
        Load match logs for a player in a specific season.
        """
        url = f"{self.BASE_URL}/players/{player_id}/matchlogs/{season}/{log_type}"
        
        try:
            html = self._fetch_url(url)
            # Use match='Date' to find log table
            dfs = pd.read_html(StringIO(html), match='Date')
            if not dfs:
                print(f"Warning: No match logs table found for {player_id}")
                return pd.DataFrame()
            df = dfs[0]
            
            if 'Date' in df.columns:
                df = df[df['Date'].notna()]
                df = df[df['Date'] != 'Date']
                
            if isinstance(df.columns, pd.MultiIndex):
                new_cols = []
                for col in df.columns.values:
                    if col[0].startswith('Unnamed'):
                        new_cols.append(col[1])
                    else:
                        new_cols.append(f"{col[0]}_{col[1]}")
                df.columns = new_cols
                
            df['player_id'] = player_id
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df
        except Exception as e:
            print(f"Error loading logs for {player_id}: {e}")
            return pd.DataFrame()

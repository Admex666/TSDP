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
    
    def __init__(self, data_dir: str = "data", cache_expiry_days: int = 1, n_workers: int = 4):
        """
        Initialize the loader with a pool of Selenium WebDrivers.
        """
        import random
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        from queue import Queue
        from concurrent.futures import ThreadPoolExecutor
        
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers = n_workers
        
        # Setup Headless Chrome Options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless=new")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Initialize Driver Pool
        self.driver_pool = Queue()
        print(f"Initializing {n_workers} Selenium drivers...")
        
        try:
            # Pre-install driver binary once
            driver_path = ChromeDriverManager().install()
            service = Service(driver_path)
            
            for i in range(n_workers):
                driver = webdriver.Chrome(service=service, options=self.chrome_options)
                driver.set_page_load_timeout(30)
                self.driver_pool.put(driver)
                print(f"  - Driver {i+1} ready.")
                
        except Exception as e:
            print(f"Failed to initialize Chrome Driver Pool: {e}")
            # If initialization fails, we might still have some drivers, or none.
            # Logic below should handle empty queue if we wait, but here it's fatal for performance.
            
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "players").mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Generate a file path for caching."""
        clean_key = key.replace("/", "_").replace("?", "_").replace(":", "")
        return self.raw_dir / f"{clean_key}.html"
        
    def _fetch_url_selenium(self, url: str) -> str:
        """
        Fetch URL content using Selenium from the Driver Pool.
        Checks local raw/ cache first.
        """
        import random
        import time
        
        # Check cache/mock first, NO driver needed!
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            # print(f"Loading from cache: {cache_path}") 
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        
        # Acquire Driver
        if self.driver_pool.empty():
            # Wait or block? Queue.get() blocks by default.
            pass
            
        driver = self.driver_pool.get() # Blocks until one is available
        
        try:
            print(f"Fetching (Selenium): {url}")
            # Random delay reduced for parallel but still polite per IP?
            # With parallel, we hit server faster. Keep delay but maybe smaller?
            delay = random.uniform(2.0, 4.0)
            time.sleep(delay)
            
            driver.get(url)
            time.sleep(1.5)
            html = driver.page_source
            
            return html
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            raise e
        finally:
            # Always return driver to pool
            self.driver_pool.put(driver)

    def fetch_urls_parallel(self, urls: List[str]) -> Dict[str, str]:
        """
        Fetch multiple URLs in parallel using the driver pool.
        Returns dict {url: html}.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import tqdm
        
        results = {}
        # Filter out cached ones to enable true progress tracking for live fetches?
        # Actually logic is inside _fetch_url_selenium. 
        # But we want to show progress of FETCHING.
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_url = {executor.submit(self._fetch_url_selenium, url): url for url in urls}
            
            # Use tqdm for progress
            for future in tqdm.tqdm(as_completed(future_to_url), total=len(urls), desc="Fetching players", leave=False):
                url = future_to_url[future]
                try:
                    html = future.result()
                    results[url] = html
                except Exception as e:
                    print(f"Failed parallel fetch {url}: {e}")
                    results[url] = None
        return results



    def load_match_schedule(self, season: str, competition_id: str, comp_name: str = "Premier-League") -> pd.DataFrame:
        """
        Load match schedule.
        1. Check data/processed/{season}_schedule.csv
        2. If not found, scrape, parse, save to CSV.
        """
        csv_path = self.processed_dir / f"{season}_{competition_id}_schedule.csv"
        if csv_path.exists():
            print(f"Loading schedule from {csv_path}")
            return pd.read_csv(csv_path)

        url = f"{self.BASE_URL}/comps/{competition_id}/{season}/schedule/{season}-{comp_name}-Scores-and-Fixtures"
        
        # Scrape
        html = self._fetch_url_selenium(url)
        
        # Parse
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find the table in soup
        # The Schedule table usually has id starting with "sched_"
        table = soup.find('table', attrs={'id': lambda x: x and x.startswith(f"sched_{season}_{competition_id}")})
        if not table:
             table = soup.find('table', attrs={'id': lambda x: x and 'sched' in x})
        
        if not table:
             print(f"Warning: Could not find schedule table for {season} {competition_id}")
             return pd.DataFrame() # Return empty if no table found
        
        data_records = []
        rows = table.find('tbody').find_all('tr')
        for row in rows:
            if 'class' in row.attrs and 'spacer' in row.attrs['class']: continue
            if 'thead' in row.find_parents(): continue # Skip header if inside tbody (unlikely)
            
            cols = row.find_all(['th', 'td'])
            row_data = {}
            for col in cols:
                stat = col.get('data-stat')
                if stat:
                    row_data[stat] = col.text.strip()
            
            # Link
            mr_cell = row.find('td', {'data-stat': 'match_report'})
            if mr_cell and mr_cell.find('a'):
                row_data['match_report_url'] = mr_cell.find('a')['href']
            
            data_records.append(row_data)
        
        df = pd.DataFrame(data_records)
            
        # Post-process columns to match expectations (Home, Away, Score, Date)
        # data-stat map: home_team -> Home, away_team -> Away, score -> Score, date -> Date
        rename_map = {
            'home_team': 'Home', 'away_team': 'Away', 'score': 'Score', 'date': 'Date', 'time': 'Time',
            'round': 'Wk', 'expected_goals': 'xG_Home', 'expected_goals_away': 'xG_Away'
        }
        df = df.rename(columns=rename_map)
        
        # Parsing Scores
        if 'Score' in df.columns:
            # Replace various dash types with a standard hyphen for splitting
            df['Score'] = df['Score'].astype(str).str.replace('–', '-').str.replace('—', '-')
            # Split and convert to numeric, coercing errors to NaN
            df[['HomeGoals', 'AwayGoals']] = df['Score'].str.split('-', expand=True).apply(pd.to_numeric, errors='coerce')
            
        # Normalize URLs
        if 'match_report_url' in df.columns:
            df['match_report_url'] = df['match_report_url'].apply(
                lambda x: (self.BASE_URL + x) if isinstance(x, str) and x.startswith("/") else x
            )
            # Remove /en/en issues if any 
            df['match_report_url'] = df['match_report_url'].str.replace("/en/en", "/en")

        # Standardize Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']) 

        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Saved schedule to {csv_path}")
        
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
            
        html = self._fetch_url_selenium(full_url) # Changed to _fetch_url_selenium
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

    def load_player_match_logs(self, player_id: str, season: str = "2023-2024") -> pd.DataFrame:
        """
        Load match logs for a player.
        1. Check data/processed/players/{player_id}_{season}_logs.csv
        2. If not found, scrape Summary AND Keepers logs (if available).
        3. Merge and save to CSV.
        """
        csv_path = self.processed_dir / "players" / f"{player_id}_{season}_logs.csv"
        if csv_path.exists():
            # print(f"Loading player logs from {csv_path}") 
            df = pd.read_csv(csv_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df
            
        # Fallback to scraping
        # Summary Log
        url_summary = f"{self.BASE_URL}/players/{player_id}/matchlogs/{season}/summary"
        df_summary = pd.DataFrame() # Initialize empty DataFrame
        
        try:
            html_summary = self._fetch_url_selenium(url_summary)
            # Use match='Date' to find log table
            dfs = pd.read_html(StringIO(html_summary), match='Date')
            if not dfs:
                print(f"Warning: No summary match logs table found for {player_id} for season {season}")
            else:
                df_summary = dfs[0]
            
            # Keepers Log (Optional)
            url_keepers = f"{self.BASE_URL}/players/{player_id}/matchlogs/{season}/keeper" # Changed to singular
            try:
                html_keepers = self._fetch_url_selenium(url_keepers)
                dfs_k = pd.read_html(StringIO(html_keepers), match='Date')
                if dfs_k:
                    df_keepers = dfs_k[0]
                    
                    if not df_summary.empty and not df_keepers.empty:
                        # Clean columns for both before merge if multi-index
                        for df_to_clean in [df_summary, df_keepers]:
                            if isinstance(df_to_clean.columns, pd.MultiIndex):
                                new_cols = []
                                for col in df_to_clean.columns.values:
                                    if col[0].startswith('Unnamed'):
                                        new_cols.append(col[1])
                                    else:
                                        new_cols.append(f"{col[0]}_{col[1]}")
                                df_to_clean.columns = new_cols

                        # Ensure 'Date' column exists and is clean for merging
                        if 'Date' in df_summary.columns and 'Date' in df_keepers.columns:
                            df_summary = df_summary[df_summary['Date'].notna()]
                            df_summary = df_summary[df_summary['Date'] != 'Date']
                            df_keepers = df_keepers[df_keepers['Date'].notna()]
                            df_keepers = df_keepers[df_keepers['Date'] != 'Date']

                            # Drop duplicate columns from keepers that are in summary before merge, except 'Date'
                            cols_to_use = df_keepers.columns.difference(df_summary.columns).tolist()
                            if 'Date' not in cols_to_use: # Ensure 'Date' is in cols_to_use for merge
                                cols_to_use.append('Date')
                            
                            # Merge on Date
                            df_summary = pd.merge(df_summary, df_keepers[cols_to_use], on='Date', how='left', suffixes=('', '_keepers'))
                        else:
                            print(f"Warning: 'Date' column missing in one of the dataframes for {player_id} for season {season}. Skipping keepers log merge.")
                    elif df_summary.empty and not df_keepers.empty:
                        # If summary was empty but keepers has data, use keepers as the base
                        df_summary = df_keepers
                        if isinstance(df_summary.columns, pd.MultiIndex):
                            new_cols = []
                            for col in df_summary.columns.values:
                                if col[0].startswith('Unnamed'):
                                    new_cols.append(col[1])
                                else:
                                    new_cols.append(f"{col[0]}_{col[1]}")
                            df_summary.columns = new_cols
                        df_summary = df_summary[df_summary['Date'].notna()]
                        df_summary = df_summary[df_summary['Date'] != 'Date']

            except Exception as e:
                # Likely not a keeper or page doesn't exist, or parsing failed.
                # print(f"Info: Could not fetch/parse keepers log for {player_id} for season {season}: {e}")
                pass # Silently ignore if keepers log fails
                
        except Exception as e:
            print(f"Error loading logs for {player_id} for season {season}: {e}")
            return pd.DataFrame() # Return empty if summary log fails
            
        # Final cleaning and processing for the combined DataFrame
        if not df_summary.empty:
            if isinstance(df_summary.columns, pd.MultiIndex):
                new_cols = []
                for col in df_summary.columns.values:
                    if col[0].startswith('Unnamed'):
                        new_cols.append(col[1])
                    else:
                        new_cols.append(f"{col[0]}_{col[1]}")
                df_summary.columns = new_cols
                
            df_summary['player_id'] = player_id
            df_summary['Date'] = pd.to_datetime(df_summary['Date'], errors='coerce')
            
            # Save to CSV
            df_summary.to_csv(csv_path, index=False)
            return df_summary
            
        return df_summary

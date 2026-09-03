import os
import sys
import re
import time
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "nbi_historical_odds_2015_2026.csv")
os.makedirs(DATA_DIR, exist_ok=True)

ODDSPORTAL_TEAM_MAP = {
    "Ferencvaros": "Ferencvárosi TC",
    "Ujpest": "Újpest FC",
    "Debrecen": "Debreceni VSC",
    "Fehervar FC": "Fehérvár FC",
    "MOL Fehervar": "Fehérvár FC",
    "Videoton": "Fehérvár FC",
    "Videoton FC": "Fehérvár FC",
    "Paks": "Paksi FC",
    "Puskas Academy": "Puskás Akadémia FC",
    "Puskas Akademia": "Puskás Akadémia FC",
    "DVTK": "Diósgyőri VTK",
    "Diosgyor": "Diósgyőri VTK",
    "Zalaegerszegi": "Zalaegerszegi TE FC",
    "Zalaegerszeg": "Zalaegerszegi TE FC",
    "MTK Budapest": "MTK Budapest",
    "Kisvarda": "Kisvárda FC",
    "Kecskemeti TE": "Kecskeméti TE",
    "Kecskemet": "Kecskeméti TE",
    "Mezokovesd-Zsory": "Mezőkövesd Zsóry FC",
    "Mezokovesd": "Mezőkövesd Zsóry FC",
    "Nyiregyhaza": "Nyíregyháza Spartacus",
    "Honved": "Budapest Honvéd FC",
    "Vasas": "Vasas FC",
    "Gyor": "ETO FC",
    "Gyirmot": "Gyirmót FC Győr",
    "Haladas": "Szombathelyi Haladás",
    "Budafoki": "Budafoki MTE",
    "Kaposvar": "Kaposvári Rákóczi FC",
    "Bekescsaba": "Békéscsaba 1912 Előre",
    "Balmazujvaros Sport": "Balmazújváros Sport Kft.",
    "Kazincbarcika": "Kazincbarcikai SC"
}

def clean_team_name(name):
    if not name:
        return ""
    name = name.strip()
    return ODDSPORTAL_TEAM_MAP.get(name, name)

def parse_oddsportal_html(html, season_slug=""):
    soup = BeautifulSoup(html, "html.parser")
    match_links = soup.find_all('a', href=lambda h: h and ('/football/h2h/' in h or '/football/hungary/' in h))
    
    rows_data = []
    seen_matches = set()
    
    for a in match_links:
        row = a.find_parent('div', class_=lambda c: c and ('border-b' in c or 'hover:bg' in c))
        if not row:
            continue
            
        strings = list(row.stripped_strings)
        if len(strings) < 5:
            continue
            
        odds_found = []
        for s in strings:
            if re.match(r'^\d+\.\d{2}$', s):
                odds_found.append(float(s))
                
        if len(odds_found) < 3:
            continue
            
        odds_1 = odds_found[-3]
        odds_x = odds_found[-2]
        odds_2 = odds_found[-1]
        
        if '-' in strings:
            dash_idx = strings.index('-')
            home_team_raw = strings[dash_idx - 1]
            away_team_raw = strings[dash_idx + 1]
            
            home_score = None
            away_score = None
            if dash_idx >= 2 and strings[dash_idx - 2].isdigit():
                home_score = int(strings[dash_idx - 2])
            if dash_idx + 2 < len(strings) and strings[dash_idx + 2].isdigit():
                away_score = int(strings[dash_idx + 2])
        else:
            continue
            
        home_team = clean_team_name(home_team_raw)
        away_team = clean_team_name(away_team_raw)
        
        result = None
        if home_score is not None and away_score is not None:
            if home_score > away_score: result = 'H'
            elif home_score == away_score: result = 'D'
            else: result = 'A'
            
        m_key = (home_team, away_team, home_score, away_score)
        if m_key in seen_matches:
            continue
        seen_matches.add(m_key)
        
        date_str = ""
        parent_group = row.find_parent('div', class_=lambda c: c and 'flex-col' in c)
        if parent_group:
            prev_headers = parent_group.find_all_previous(string=lambda s: s and any(m in s for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) and any(c.isdigit() for c in s))
            for ph in prev_headers:
                ph_str = ph.strip()
                if len(ph_str) <= 25:
                    try:
                        dt = pd.to_datetime(ph_str)
                        date_str = dt.strftime('%Y-%m-%d')
                        break
                    except Exception:
                        pass
                        
        rows_data.append({
            'season': season_slug,
            'date': date_str,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'result': result,
            'odds_1': odds_1,
            'odds_x': odds_x,
            'odds_2': odds_2
        })
        
    return rows_data

def scrape_all_nb1_seasons(headless=True):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    
    # 12 seasons: current down to 2015-2016
    seasons = [
        ("current", "https://www.oddsportal.com/football/hungary/nb-i/results/"),
        ("2025-2026", "https://www.oddsportal.com/football/hungary/nb-i-2025-2026/results/"),
        ("2024-2025", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2024-2025/results/"),
        ("2023-2024", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2023-2024/results/"),
        ("2022-2023", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2022-2023/results/"),
        ("2021-2022", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2021-2022/results/"),
        ("2020-2021", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2020-2021/results/"),
        ("2019-2020", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2019-2020/results/"),
        ("2018-2019", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2018-2019/results/"),
        ("2017-2018", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2017-2018/results/"),
        ("2016-2017", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2016-2017/results/"),
        ("2015-2016", "https://www.oddsportal.com/football/hungary/otp-bank-liga-2015-2016/results/")
    ]
    
    existing_df = pd.DataFrame()
    scraped_seasons = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            # Only count as scraped if season has at least 150 matches (completed season)
            season_counts = existing_df['season'].value_counts()
            scraped_seasons = set(season_counts[season_counts >= 150].index)
            print(f"[+] Loaded {len(existing_df)} existing records. Completed seasons: {scraped_seasons}")
        except Exception:
            pass
            
    all_records = existing_df.to_dict('records') if not existing_df.empty else []
    
    driver = webdriver.Chrome(options=opts)
    
    try:
        for s_idx, (season_slug, base_url) in enumerate(seasons):
            if season_slug in scraped_seasons and season_slug != "current":
                print(f"[*] Season {season_slug} already fully scraped ({len(existing_df[existing_df['season']==season_slug])} matches). Skipping.")
                continue
                
            print(f"\n========================================================")
            print(f"[{s_idx+1}/{len(seasons)}] Scraping Season: {season_slug}")
            print(f"URL: {base_url}")
            print(f"========================================================")
            
            season_matches = []
            
            for page in range(1, 6):
                page_url = f"{base_url.rstrip('/')}/#/page/{page}/" if page > 1 else base_url
                print(f"  Fetching Page {page}: {page_url}")
                
                try:
                    driver.get(page_url)
                    time.sleep(3.5)
                    
                    for _ in range(4):
                        driver.execute_script("window.scrollBy(0, 800);")
                        time.sleep(0.25)
                        
                    parsed = parse_oddsportal_html(driver.page_source, season_slug)
                    print(f"    -> Parsed {len(parsed)} matches from page {page}")
                    
                    if not parsed:
                        print("    -> No matches found on this page, season finished.")
                        break
                        
                    season_matches.extend(parsed)
                    
                    if len(parsed) < 30:
                        print("    -> Last page reached.")
                        break
                except Exception as e:
                    print(f"    [-] Error loading page {page}: {e}")
                    break
                    
            print(f"[+] Total matches scraped for {season_slug}: {len(season_matches)}")
            
            if season_matches:
                all_records.extend(season_matches)
                df_interim = pd.DataFrame(all_records).drop_duplicates(subset=['season', 'home_team', 'away_team', 'home_score', 'away_score'])
                df_interim.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
                
                root_csv = os.path.join(os.path.dirname(DATA_DIR), "nbi_historical_odds_2015_2026.csv")
                df_interim.to_csv(root_csv, index=False, encoding='utf-8-sig')
                print(f"[+] Saved interim dataset ({len(df_interim)} rows) to: {OUTPUT_CSV}")
                
            time.sleep(1)
            
    finally:
        driver.quit()
        
    final_df = pd.DataFrame(all_records).drop_duplicates(subset=['season', 'home_team', 'away_team', 'home_score', 'away_score'])
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    root_csv = os.path.join(os.path.dirname(DATA_DIR), "nbi_historical_odds_2015_2026.csv")
    final_df.to_csv(root_csv, index=False, encoding='utf-8-sig')
    
    print("\n========================================================")
    print(f"[+] SUCCESS! Scraped a total of {len(final_df)} matches with 1X2 odds.")
    print(f"[+] Saved to: {OUTPUT_CSV}")
    print("========================================================")
    return final_df

if __name__ == "__main__":
    scrape_all_nb1_seasons(headless=True)

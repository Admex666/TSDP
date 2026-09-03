import os
import sys
import re
import time
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Ensure Windows uses ProactorEventLoop for subprocesses
if sys.platform == 'win32':
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "modules"))
if modules_path not in sys.path:
    sys.path.append(modules_path)

from transfermarkt_scraper import TransfermarktScraper

CANONICAL_CSV = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")

TEAM_NAME_STANDARDIZATION = {
    "ETO FC Győr": "ETO FC",
    "ETO FC Györ": "ETO FC",
    "Videoton FC": "Fehérvár FC",
    "Videoton FC Fehérvár": "Fehérvár FC",
    "MOL Vidi FC": "Fehérvár FC",
    "MOL Fehérvár FC": "Fehérvár FC",
    "Kispest–Honvéd FC": "Budapest Honvéd FC"
}

TEAM_CODE_MAP = {
    'Diósgyőri VTK': 'DVTK',
    'Vasas FC': 'VASAS',
    'Ferencvárosi TC': 'FTC',
    'Újpest FC': 'UJPEST',
    'Paksi FC': 'PAKS',
    'Budapest Honvéd FC': 'HONVED',
    'Fehérvár FC': 'FEHERVAR',
    'Szombathelyi Haladás': 'HALADAS',
    'Békéscsaba 1912 Előre': 'BEKESCSABA',
    'Debreceni VSC': 'DVSC',
    'Balmazújváros Sport Kft.': 'BALMAZ',
    'Mezőkövesd Zsóry FC': 'MEZOKOVESD',
    'Gyirmót FC Győr': 'GYIRMOT',
    'Kaposvári Rákóczi FC': 'KAPOSVAR',
    'Zalaegerszegi TE FC': 'ZTE',
    'Budafoki MTE': 'BUDAFOK',
    'Kecskeméti TE': 'KTE',
    'Kisvárda FC': 'KISVARDA',
    'ETO FC': 'ETO',
    'Nyíregyháza Spartacus': 'NYIREGYHAZA',
    'Kazincbarcikai SC': 'KBARCIKA',
    'MTK Budapest': 'MTK',
    'Puskás Akadémia FC': 'PUSKAS'
}

def clean_team_name(name):
    if not name:
        return ""
    name = name.strip()
    return TEAM_NAME_STANDARDIZATION.get(name, name)

def format_date(date_str):
    if not date_str:
        return ""
    date_str = date_str.strip()
    try:
        return datetime.strptime(date_str, "%m/%d/%y").strftime("%Y-%m-%d")
    except Exception:
        try:
            return pd.to_datetime(date_str).strftime("%Y-%m-%d")
        except Exception:
            return date_str

def format_time(time_str):
    if not time_str:
        return ""
    time_str = time_str.strip()
    try:
        return datetime.strptime(time_str, "%I:%M %p").strftime("%H:%M")
    except Exception:
        return time_str

def parse_transfermarkt_html(html_content, season_id):
    """Parses Transfermarkt gesamtspielplan HTML into matches DataFrame."""
    soup = BeautifulSoup(html_content, 'html.parser')
    boxes = soup.find_all('div', class_=lambda c: c and 'box' in c)
    matches = []
    
    for box in boxes:
        headline = box.find(class_='content-box-headline')
        if not headline or "Matchday" not in headline.get_text():
            continue
            
        matchday_text = headline.get_text(strip=True)
        m_day = re.search(r'\d+', matchday_text)
        matchday_num = int(m_day.group(0)) if m_day else 0
        
        table = box.find('table')
        if not table:
            continue
            
        current_date = ""
        current_time = ""
        rows = table.select('tbody tr')
        
        for row in rows:
            if 'bg_blau_20' in row.get('class', []):
                a_date = row.find('a', href=re.compile(r'/datum/'))
                if a_date:
                    current_date = a_date.get_text(strip=True)
                row_text = row.get_text(strip=True)
                t_match = re.search(r'\d{1,2}:\d{2}\s*(?:AM|PM)?', row_text)
                if t_match:
                    current_time = t_match.group(0)
                continue
                
            spielbericht_link = row.find('a', href=re.compile(r'/spielbericht/'))
            verein_links = row.find_all('a', href=re.compile(r'/verein/'))
            
            if not spielbericht_link and not verein_links:
                continue
                
            row_date_link = row.find('a', href=re.compile(r'/datum/'))
            if row_date_link:
                current_date = row_date_link.get_text(strip=True)
                
            tds = row.find_all('td')
            for td in tds:
                text_td = td.get_text(strip=True)
                if re.search(r'\d{1,2}:\d{2}\s*(?:AM|PM)?', text_td):
                    current_time = text_td
                    break
                    
            team_names = []
            for v_link in verein_links:
                name = v_link.get('title') or v_link.get_text(strip=True)
                if name and name not in team_names and not v_link.find('img'):
                    team_names.append(name)
            if len(team_names) < 2:
                team_names = []
                for v_link in verein_links:
                    name = v_link.get_text(strip=True)
                    if name and name not in team_names and not v_link.find('img'):
                        team_names.append(name)
                        
            home_team = clean_team_name(team_names[0] if len(team_names) >= 1 else "")
            away_team = clean_team_name(team_names[1] if len(team_names) >= 2 else "")
            
            home_score = None
            away_score = None
            result = None
            goal_diff = None
            is_played = False
            
            if spielbericht_link:
                res_text = spielbericht_link.get_text(strip=True)
                score_match = re.search(r'(\d+):(\d+)', res_text)
                if score_match:
                    home_score = int(score_match.group(1))
                    away_score = int(score_match.group(2))
                    is_played = True
                    if home_score > away_score:
                        result = 'H'
                    elif home_score == away_score:
                        result = 'D'
                    else:
                        result = 'A'
                    goal_diff = home_score - away_score
                    
            h_code = TEAM_CODE_MAP.get(home_team, home_team[:4].upper())
            a_code = TEAM_CODE_MAP.get(away_team, away_team[:4].upper())
            match_id = f"{season_id}_{matchday_num}_{h_code}_{a_code}"
            
            f_date = format_date(current_date)
            f_time = format_time(current_time)
            
            matches.append({
                'match_id': match_id,
                'season_id': season_id,
                'matchday': matchday_num,
                'date': f_date,
                'time': f_time,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'result': result,
                'goal_diff': goal_diff,
                'is_played': is_played
            })
            
    return pd.DataFrame(matches)

def scrape_current_season_matches(season_id=2026, headless=True):
    """
    Scrapes all fixtures (both played and unplayed/future) for the given season from Transfermarkt.
    Uses Playwright when possible, and falls back to requests if inside an active asyncio thread.
    """
    url = f"https://www.transfermarkt.us/nemzeti-bajnoksag/gesamtspielplan/wettbewerb/UNG1?saison_id={season_id}&spieltagVon=1&spieltagBis=33"
    print(f"Scraping current season {season_id} from Transfermarkt: {url}")
    
    html_content = ""
    
    # 1. Try Playwright first
    try:
        scraper = TransfermarktScraper(use_playwright=True)
        scraper.init_playwright(headless=headless)
        html_content = scraper._get_url_html(url)
        
        if scraper.page:
            for i in range(1, 6):
                scraper.page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {i/5})")
                scraper.page.wait_for_timeout(200)
            html_content = scraper.page.content()
            
        scraper.close_playwright()
    except Exception as pw_err:
        print(f"[-] Playwright encounter: {pw_err}. Falling back to direct HTTP requests...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                html_content = resp.text
                print("[+] Direct HTTP fetch successful!")
            else:
                print(f"[-] HTTP error: Status code {resp.status_code}")
        except Exception as req_err:
            print(f"[-] Direct HTTP fetch error: {req_err}")
            
    if not html_content:
        return pd.DataFrame()
        
    df = parse_transfermarkt_html(html_content, season_id)
    print(f"[+] Scraped {len(df)} total fixtures for season {season_id}.")
    print(f"    - Played: {len(df[df['is_played']])}")
    print(f"    - Upcoming: {len(df[~df['is_played']])}")
    return df

def update_season_and_master():
    df_2026 = scrape_current_season_matches(season_id=2026, headless=True)
    if not df_2026.empty:
        out_2026 = os.path.join(os.path.dirname(__file__), "nbi_matches_2026_current.csv")
        df_2026.to_csv(out_2026, index=False, encoding='utf-8-sig')
        print(f"[+] Saved 2026 season fixtures to: {out_2026}")
    return df_2026

if __name__ == "__main__":
    update_season_and_master()

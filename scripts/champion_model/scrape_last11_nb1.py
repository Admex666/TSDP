import os
import sys
import re
import time
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup

# A modules könyvtár hozzáadása a sys.path-hoz
modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "modules"))
if modules_path not in sys.path:
    sys.path.append(modules_path)

from transfermarkt_scraper import TransfermarktScraper

output_csv = os.path.join(os.path.dirname(__file__), "nbi_matches_2015_2025.csv")

def format_date(date_str):
    """Átalakítja az MM/DD/YY formátumot YYYY-MM-DD formátumra."""
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
    """Átalakítja a 12 órás AM/PM időpontot 24 órás HH:MM formátumra."""
    if not time_str:
        return ""
    time_str = time_str.strip()
    try:
        return datetime.strptime(time_str, "%I:%M %p").strftime("%H:%M")
    except Exception:
        return time_str

def main():
    scraper = TransfermarktScraper(use_playwright=True)
    all_matches = []
    
    # 2015-től 2025-ig (11 szezon)
    seasons = list(range(2015, 2026))
    
    try:
        print(f"Playwright elindítása (11 szezon lekérése: 2015 - 2025)...")
        scraper.init_playwright(headless=False)
        
        for season_id in seasons:
            url = f"https://www.transfermarkt.us/nemzeti-bajnoksag/gesamtspielplan/wettbewerb/UNG1?saison_id={season_id}&spieltagVon=1&spieltagBis=33"
            print(f"\n--- Szezon betöltése: {season_id} ---")
            print(f"URL: {url}")
            
            # 1. Oldal betöltése
            html_content = scraper._get_url_html(url)
            
            # 2. Görgetés az oldal aljára a biztonság kedvéért
            if scraper.page:
                for i in range(1, 6):
                    scraper.page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {i/5})")
                    scraper.page.wait_for_timeout(300)
                html_content = scraper.page.content()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Keressük az összes boxot, ami fordulót tartalmaz
            boxes = soup.find_all('div', class_=lambda c: c and 'box' in c)
            season_match_count = 0
            
            for box in boxes:
                headline = box.find(class_='content-box-headline')
                if not headline or "Matchday" not in headline.get_text():
                    continue
                    
                matchday_text = headline.get_text(strip=True)
                m_day = re.search(r'\d+', matchday_text)
                matchday_num = m_day.group(0) if m_day else matchday_text
                
                table = box.find('table')
                if not table:
                    continue
                    
                current_date = ""
                current_time = ""
                rows = table.select('tbody tr')
                
                for row in rows:
                    # Dátum és időpont fejléc sora (bg_blau_20)
                    if 'bg_blau_20' in row.get('class', []):
                        a_date = row.find('a', href=re.compile(r'/datum/'))
                        if a_date:
                            current_date = a_date.get_text(strip=True)
                        
                        row_text = row.get_text(strip=True)
                        t_match = re.search(r'\d{1,2}:\d{2}\s*(?:AM|PM)?', row_text)
                        if t_match:
                            current_time = t_match.group(0)
                        continue
                        
                    # Meccs sor ellenőrzése
                    spielbericht_link = row.find('a', href=re.compile(r'/spielbericht/'))
                    verein_links = row.find_all('a', href=re.compile(r'/verein/'))
                    
                    if not spielbericht_link and not verein_links:
                        continue
                        
                    # Dátum frissítése ha van meccssorban egyedi datum link
                    row_date_link = row.find('a', href=re.compile(r'/datum/'))
                    if row_date_link:
                        current_date = row_date_link.get_text(strip=True)
                        
                    # Időpont frissítése ha van a meccs sorában
                    tds = row.find_all('td')
                    row_time = ""
                    for td in tds:
                        text_td = td.get_text(strip=True)
                        if re.search(r'\d{1,2}:\d{2}\s*(?:AM|PM)?', text_td):
                            row_time = text_td
                            break
                            
                    if row_time:
                        current_time = row_time
                            
                    # Hazai és Vendég csapatok kinyerése
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
                                
                    home_team = team_names[0] if len(team_names) >= 1 else ""
                    away_team = team_names[1] if len(team_names) >= 2 else ""
                    
                    # Eredmény és gólok
                    home_score, away_score = "", ""
                    if spielbericht_link:
                        res_text = spielbericht_link.get_text(strip=True)
                        score_match = re.search(r'(\d+):(\d+)', res_text)
                        if score_match:
                            home_score = score_match.group(1)
                            away_score = score_match.group(2)
                            
                    all_matches.append({
                        'season_id': season_id,
                        'matchday': matchday_num,
                        'date': format_date(current_date),
                        'time': format_time(current_time),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score
                    })
                    season_match_count += 1
            
            print(f"[+] Szezon {season_id} kész: {season_match_count} mérkőzés kinyerve.")
            time.sleep(1)
                
        df = pd.DataFrame(all_matches)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"\n[+] TELJES SIKER!")
        print(f"[+] Összesen {len(df)} mérkőzés kinyerve 11 szezonból és elmentve ide: {output_csv}")
        print("\n--- Első 10 mérkőzés a DataFrame-ből ---")
        print(df.head(10).to_string(index=False))
        print("\n--- Utolsó 10 mérkőzés a DataFrame-ből ---")
        print(df.tail(10).to_string(index=False))
        
    except Exception as e:
        print(f"\n[-] Hiba történt a lekérés során: {e}")
    finally:
        scraper.close_playwright()

if __name__ == "__main__":
    main()

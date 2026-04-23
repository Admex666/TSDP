import pandas as pd
import time
import os
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def get_page_content(url, page):
    try:
        # Increase timeout and use networkidle
        page.goto(url, wait_until="networkidle", timeout=60000)
        content = page.content()
        # Transfermarkt sometimes has a cookie consent overlay or similar
        return content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_magyar_kupa(page, season=2025):
    base_url = "https://www.transfermarkt.com"
    cup_url = f"{base_url}/magyar-kupa/gesamtspielplan/pokalwettbewerb/UNGP/saison_id/{season}"
    
    print(f"Fetching matches from: {cup_url}")
    html = get_page_content(cup_url, page)
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')
    matches = []
    
    # We'll look for all table rows and identify matches
    rows = soup.select('tr')
    current_round = "Unknown"
    
    for row in rows:
        # Check if this row is a header for a round
        round_header = row.select_one('td.hauptlink, h2')
        if round_header and not row.select_one('a[href*="/spielbericht/"]'):
            # This might be a round header or just a spacer
            text = round_header.get_text(strip=True)
            if any(r in text for r in ["Round", "Final", "Quarter", "Semi"]):
                current_round = text
        
        # Match row structure from subagent:
        # Home: td with class 'text-right' and 'hauptlink'
        # Away: td with class 'no-border-links' and 'hauptlink' (and usually NOT 'zentriert')
        # Result: td with class 'zentriert' and 'hauptlink'
        
        home_td = row.select_one('td.text-right.hauptlink') or row.select_one('td.rechts.hauptlink')
        away_td = row.select_one('td.no-border-links.hauptlink:not(.zentriert)') or row.select_one('td.links.hauptlink')
        result_td = row.select_one('td.zentriert.hauptlink')
        
        if home_td and away_td and result_td:
            home_a = home_td.select_one('a')
            away_a = away_td.select_one('a')
            result_a = result_td.select_one('a[href*="/spielbericht/"]')
            
            if home_a and away_a and result_a:
                matches.append({
                    'round': current_round,
                    'home_team': home_a.get_text(strip=True),
                    'home_url': base_url + home_a.get('href'),
                    'away_team': away_a.get_text(strip=True),
                    'away_url': base_url + away_a.get('href'),
                    'result': result_a.get_text(strip=True),
                    'match_url': base_url + result_a.get('href'),
                })

    df_matches = pd.DataFrame(matches)
    if not df_matches.empty:
        df_matches = df_matches.drop_duplicates().reset_index(drop=True)
    
    print(f"Found {len(df_matches)} matches.")
    return df_matches

def get_team_league_level(url, page):
    print(f"Fetching league level for: {url}")
    html = get_page_content(url, page)
    if not html:
        return "Unknown"
    
    soup = BeautifulSoup(html, 'html.parser')
    try:
        # 1. Primary: Search for "League level:" label
        for label in soup.select('span.data-header__label'):
            if "League level" in label.get_text():
                content = label.find_next('span', class_='data-header__content')
                if content:
                    return content.get_text(strip=True)
        
        # 2. Fallback: The club info section usually has the league name as a primary link
        club_info = soup.select_one('div.data-header__club-info')
        if club_info:
            league_a = club_info.select_one('span.data-header__club a')
            if league_a:
                return league_a.get_text(strip=True)
        
        # 3. Fallback: Header box big
        header_box = soup.select_one('div.data-header__box--big')
        if header_box:
            league_a = header_box.select_one('a.data-header__box__club-link')
            if league_a:
                # The text is often in alt attribute of img or adjacent span
                img = league_a.select_one('img')
                if img and img.get('alt'):
                    return img.get('alt')
                    
    except Exception as e:
        print(f"Error parsing league level: {e}")
        
    return "Unknown"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        
        # 1. Scrape matches
        df_matches = scrape_magyar_kupa(page, 2025)
        
        if df_matches is not None and not df_matches.empty:
            # 2. Collect unique teams
            all_teams = pd.concat([
                df_matches[['home_team', 'home_url']].rename(columns={'home_team': 'team', 'home_url': 'url'}),
                df_matches[['away_team', 'away_url']].rename(columns={'away_team': 'team', 'away_url': 'url'})
            ]).drop_duplicates('url')
            
            print(f"Total unique teams: {len(all_teams)}")
            
            team_levels = {}
            for i, (idx, row) in enumerate(all_teams.iterrows(), 1):
                level = get_team_league_level(row['url'], page)
                team_levels[row['url']] = level
                print(f"[{i}/{len(all_teams)}] {row['team']}: {level}")
                time.sleep(0.4) # Slightly faster
                
            df_matches['home_league'] = df_matches['home_url'].map(team_levels)
            df_matches['away_league'] = df_matches['away_url'].map(team_levels)
            
            output_file = "magyar_kupa_results_2025.csv"
            df_matches.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"Saved results to {output_file}")
        else:
            print("No matches found.")
        
        browser.close()

if __name__ == "__main__":
    main()

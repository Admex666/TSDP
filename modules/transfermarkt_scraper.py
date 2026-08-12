# -*- coding: utf-8 -*-
import os
import json
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

class TransfermarktScraper:
    def __init__(self, use_playwright=True):
        self.headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(self.base_dir, 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.players_cache_file = os.path.join(self.cache_dir, 'players.json')
        self.clubs_cache_file = os.path.join(self.cache_dir, 'clubs.json')
        
        self.players_cache = self._load_json(self.players_cache_file)
        self.clubs_cache = self._load_json(self.clubs_cache_file)
        
        self.use_playwright = use_playwright
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def _load_json(self, filepath):
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_cache(self):
        with open(self.players_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.players_cache, f, ensure_ascii=False, indent=4)
        with open(self.clubs_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.clubs_cache, f, ensure_ascii=False, indent=4)
        print("Cache saved successfully.")

    def init_playwright(self, headless=False):
        if not self.use_playwright:
            return
        from playwright.sync_api import sync_playwright
        print("Initializing Playwright browser context...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.context = self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        self.page = self.context.new_page()
        # Set short default navigation timeout
        self.page.set_default_navigation_timeout(30000)

    def close_playwright(self):
        if self.page:
            self.page.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        print("Playwright browser closed.")

    def wait_for_captcha(self):
        """Megvárja, amíg a felhasználó megoldja a captchát a megnyílt böngészőben."""
        if not self.page:
            return
        content = self.page.content().lower()
        if "confirm you are human" in content or "captcha" in content or "cloudflare" in content:
            print("\n[!] CAPTCHA vagy Cloudflare blokkolás észlelve!")
            print("[!] Kérjük, oldd meg a captchát a megnyílt böngészőablakban!")
            print("[!] A kód automatikusan folytatódik, amint a captcha oldal eltűnik...")
            while True:
                time.sleep(2)
                try:
                    current_content = self.page.content().lower()
                    if "confirm you are human" not in current_content and "captcha" not in current_content:
                        break
                except Exception:
                    # Ha bezárulna vagy hiba lenne
                    break
            print("[+] Captcha megoldva, folytatás...")
            time.sleep(1)

    def handle_cookie_consent(self):
        """Beleszól a Sourcepoint cookie ablakba és rákattint az Accept-re."""
        if not self.page:
            return
        try:
            # Sourcepoint iframe keresése
            iframe = self.page.frame_locator('iframe[id^="sp_message_iframe"]')
            accept_button = iframe.locator('button:has-text("Accept & continue"), button[title="Accept & continue"]').first
            # Várunk max 3 másodpercet, amíg láthatóvá válik
            accept_button.wait_for(state="visible", timeout=3000)
            accept_button.click()
            print("[+] Cookie consent accepted.")
            self.page.wait_for_timeout(1000)
        except Exception:
            # Ha már el van fogadva vagy nem jelenik meg, megyünk tovább
            pass

    def _get_url_html(self, url, max_retries=3):
        """Letölti az URL HTML tartalmát. Playwright-ot használ ha elérhető, egyébként request-et."""
        if self.page:
            for attempt in range(max_retries):
                try:
                    self.page.goto(url, wait_until="domcontentloaded")
                    self.wait_for_captcha()
                    self.handle_cookie_consent()
                    # Ha teljesítményoldalt töltünk be, legörgetünk és megvárjuk a Svelte táblázatot
                    if "leistungsdatendetails" in url:
                        try:
                            for step in range(1, 5):
                                self.page.evaluate(f"window.scrollTo(0, {step * 500})")
                                self.page.wait_for_timeout(400)
                            self.page.wait_for_selector("div[role='table'], table.items", timeout=5000)
                        except Exception:
                            pass
                    else:
                        self.page.wait_for_timeout(1000)
                    return self.page.content()
                except Exception as e:
                    print(f"Hiba a Playwright betöltésnél ({url}) - Próbálkozás {attempt+1}/{max_retries}: {e}")
                    time.sleep(3)
            raise RuntimeError(f"Nem sikerült betölteni a(z) {url} oldalt Playwright-al.")
        else:
            # Fallback requests
            res = requests.get(url, headers=self.headers, timeout=15)
            res.raise_for_status()
            return res.text

    def _get_url_json(self, url, max_retries=3):
        """JSON-t kér le az URL-ről. Playwright-ot használ ha elérhető, egyébként request-et."""
        if self.page:
            for attempt in range(max_retries):
                try:
                    self.page.goto(url, wait_until="networkidle")
                    self.wait_for_captcha()
                    self.handle_cookie_consent()
                    # A json általában egy <pre> tagben vagy közvetlenül a test-ben van
                    json_text = self.page.locator("pre").text_content() if self.page.locator("pre").is_visible() else self.page.content()
                    
                    if "<html" in json_text.lower():
                        # Ha a content mégis HTML-t tartalmaz, megpróbáljuk Regexszel kiszűrni a JSON-t
                        json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                        if json_match:
                            json_text = json_match.group(0)
                        else:
                            # Próbáljuk újra
                            raise ValueError("A válasz HTML és nem tartalmaz JSON objektumot.")
                    
                    return json.loads(json_text)
                except Exception as e:
                    print(f"Hiba a JSON betöltésnél ({url}) - Próbálkozás {attempt+1}/{max_retries}: {e}")
                    time.sleep(3)
            raise RuntimeError(f"Nem sikerült betölteni a(z) {url} JSON-t Playwright-al.")
        else:
            res = requests.get(url, headers=self.headers, timeout=15)
            res.raise_for_status()
            return res.json()

    def _extract_id(self, url):
        if not url or not isinstance(url, str): return None
        
        # 1. Target specific Entity IDs first (verein, spieler, transfer, marktwertverlauf)
        m_verein = re.search(r'/verein/(\d+)', url)
        if m_verein: return m_verein.group(1)
        
        m_spieler = re.search(r'/spieler/(\d+)', url)
        if m_spieler: return m_spieler.group(1)
        
        m_transfer = re.search(r'/transfer/(\d+)', url)
        if m_transfer: return m_transfer.group(1)
        
        m_mw = re.search(r'/marktwertverlauf/(\d+)', url)
        if m_mw: return m_mw.group(1)
        
        m_wett = re.search(r'/wettbewerb/([^/]+)', url)
        if m_wett: return m_wett.group(1)
        
        match = re.search(r'/(?:spieler|verein|wettbewerb|transfer|marktwertverlauf)/([^/]+)', url)
        if match:
            return match.group(1)
        return None

    def get_historical_transfers(self, league_id, season_id, league_name="league"):
        """Lekéri egy bajnokság adott szezonbeli igazolásait HTML scraping segítségével."""
        url = f"https://www.transfermarkt.com/{league_name}/transfers/wettbewerb/{league_id}/plus/?saison_id={season_id}"
        html = self._get_url_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        
        all_transfers = []
        boxes = soup.find_all('div', class_='box')
        
        for box in boxes:
            club_link = box.select_one('div.table-header a.hauptlink, h2.table-header a.hauptlink')
            if not club_link:
                club_link = box.find('a', href=re.compile(r'/transfers/verein/'))
            if not club_link: continue
            
            current_club_id = self._extract_id(club_link.get('href'))
            current_club_name = club_link.text.strip() or club_link.get('title', '').strip()
            
            tables = box.find_all('div', class_='responsive-table')
            for table_idx, table in enumerate(tables):
                th_first = table.find('th')
                th_text = th_first.text.strip().lower() if th_first else ""
                
                if any(kw in th_text for kw in ['in', 'zugänge', 'arrivals', 'érkezők']):
                    is_arrival = True
                elif any(kw in th_text for kw in ['out', 'abgänge', 'departures', 'távozók']):
                    is_arrival = False
                else:
                    is_arrival = (table_idx == 0)
                
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 4: continue
                    
                    col_count = len(cols)
                    idx_club = 7 if col_count >= 9 else 6
                    idx_fee = 8 if col_count >= 9 else 7
                    idx_mv = 5 if col_count >= 9 else 4
                    
                    player_span = cols[0].select_one('.hide-for-small')
                    player_link = player_span.find('a') if player_span else cols[0].find('a')
                    if not player_link: continue
                    
                    other_club_link = cols[idx_club].find('a')
                    other_club_id = self._extract_id(other_club_link.get('href')) if other_club_link else None
                    other_club_name = other_club_link.text.strip() if other_club_link else "Unknown"
                    
                    # Cél bajnokság ID kinyerése ha van
                    # Pl. a klubcímer linkből vagy a bajnokság linkből
                    other_league_id = None
                    other_league_link = cols[idx_club].find('img')
                    if other_league_link and other_league_link.get('title'):
                        # A címben benne lehet a liga neve, de az ID-t nehezebb kinyerni. 
                        # Inkább a linkekből keressük:
                        league_a = cols[idx_club].select_one('a[href*="/wettbewerb/"]')
                        if league_a:
                            other_league_id = self._extract_id(league_a.get('href'))
                    
                    transfer_data = {
                        'player_id': self._extract_id(player_link.get('href')),
                        'player_name': player_link.text.strip(),
                        'age': cols[1].text.strip(),
                        'market_value': cols[idx_mv].text.strip(),
                        'fee': cols[idx_fee].text.strip(),
                        'season_id': season_id,
                        'league_id': league_id
                    }
                    
                    if is_arrival:
                        transfer_data.update({
                            'from_club_id': other_club_id,
                            'from_club_name': other_club_name,
                            'from_league_id': other_league_id,
                            'to_club_id': current_club_id,
                            'to_club_name': current_club_name,
                            'to_league_id': league_id
                        })
                    else:
                        transfer_data.update({
                            'from_club_id': current_club_id,
                            'from_club_name': current_club_name,
                            'from_league_id': league_id,
                            'to_club_id': other_club_id,
                            'to_club_name': other_club_name,
                            'to_league_id': other_league_id
                        })
                    all_transfers.append(transfer_data)
                    
        return pd.DataFrame(all_transfers)

    def get_player_transfer_history_api(self, player_id):
        """Lekéri egy játékos teljes átigazolási múltját a belső JSON API végpontról."""
        url = f"https://tmapi-alpha.transfermarkt.technology/transfer/history/player/{player_id}"
        data = self._get_url_json(url)
        
        terminated = data.get('data', {}).get('history', {}).get('terminated', [])
        results = []
        
        for transfer in terminated:
            details = transfer.get('details', {})
            source = transfer.get('transferSource', {})
            dest = transfer.get('transferDestination', {})
            
            fee_compact = details.get('fee', {}).get('compact', {})
            fee_str = f"{fee_compact.get('content', '')}{fee_compact.get('suffix', '')}" if fee_compact else "0"
            
            # Határozzuk meg, hogy kölcsön-e
            is_loan = details.get('isLoan', False)
            is_loan_return = details.get('isLoanReturn', False)
            
            results.append({
                'transfer_id': transfer.get('id'),
                'season': details.get('season', {}).get('display'),
                'season_id': details.get('season', {}).get('id'),
                'date': details.get('date'),
                'age': details.get('age'),
                'from_club_id': source.get('clubId'),
                'from_club_name': source.get('clubName') or source.get('club', {}).get('name'),
                'from_league_id': source.get('competitionId'),
                'from_country': source.get('countryName'),
                'to_club_id': dest.get('clubId'),
                'to_club_name': dest.get('clubName') or dest.get('club', {}).get('name'),
                'to_league_id': dest.get('competitionId'),
                'to_country': dest.get('countryName'),
                'fee_value': details.get('fee', {}).get('value'),
                'fee_formatted': fee_str,
                'market_value': details.get('marketValue', {}).get('value'),
                'is_loan': 1 if is_loan else 0,
                'is_loan_return': 1 if is_loan_return else 0,
                'player_id': player_id
            })
            
        return pd.DataFrame(results)
    def get_complete_player_data(self, player_id):
        """
        Lekéri a játékos összes adatát (alapadatok, teljesítmények, piaci érték idősor, sérülések)
        egyetlen profiloldal betöltésével és háttér lekérdezésekkel.
        """
        from config import LEAGUE_MAP
        player_id = str(player_id)
        
        # 1. Betöltjük a profiloldalt
        url_profile = f"https://www.transfermarkt.com/spieler/profil/spieler/{player_id}"
        html = self._get_url_html(url_profile)
        soup = BeautifulSoup(html, 'html.parser')
        
        # 1/A. Alapadatok kiszedése
        details = {
            'player_id': player_id,
            'name': None,
            'birth_date': None,
            'nationality': None,
            'primary_position': None,
            'foot': None,
            'height': None
        }
        name_header = soup.select_one('h1.data-header__headline-wrapper')
        if name_header:
            details['name'] = name_header.get_text(strip=True).replace('#', '').strip()
            details['name'] = re.sub(r'^\d+\s*', '', details['name'])
            
        info_table = soup.find('div', class_='info-table')
        if info_table:
            def find_info_value(label_text):
                label_span = info_table.find(lambda tag: tag.name == 'span' and label_text in tag.text)
                if label_span:
                    val = label_span.find_next_sibling('span')
                    if val:
                        if label_text == 'Citizenship':
                            flags = val.find_all('img', class_='flaggenrahmen')
                            if flags:
                                return "; ".join([img.get('title') for img in flags])
                        return val.get_text(strip=True)
                return None
                
            details['birth_date'] = find_info_value('Date of birth') or find_info_value('Birth date')
            details['nationality'] = find_info_value('Citizenship')
            details['primary_position'] = find_info_value('Position')
            details['foot'] = find_info_value('Foot')
            details['height'] = find_info_value('Height')
            
        # Update cache
        self.players_cache[player_id] = details
        
        # 2. Teljesítmény adatok lekérése háttér fetch-el
        df_perf = pd.DataFrame()
        js_perf = f"""
        fetch('/ceapi/performance-game/{player_id}')
            .then(response => {{
                if (!response.ok) throw new Error('HTTP status ' + response.status);
                return response.json();
            }})
        """
        try:
            perf_data = self.page.evaluate(js_perf)
            performances = perf_data.get('data', {}).get('performance', [])
            
            agg = {}
            for game in performances:
                g_info = game.get('gameInformation', {})
                if g_info.get('isNationalGame', False):
                    continue
                    
                season = g_info.get('season', {}).get('display')
                comp_id = g_info.get('competitionId')
                club_info = game.get('clubsInformation', {}).get('club', {})
                club_id = club_info.get('clubId')
                
                if not all([season, comp_id, club_id]):
                    continue
                    
                key = (season, comp_id, club_id)
                if key not in agg:
                    agg[key] = {
                        'appearances': 0,
                        'minutes': 0,
                        'goals': 0,
                        'assists': 0
                    }
                    
                stats = game.get('statistics', {})
                play_time = stats.get('playingTimeStatistics', {})
                mins = play_time.get('playedMinutes')
                
                if mins is not None:
                    agg[key]['appearances'] += 1
                    agg[key]['minutes'] += int(mins)
                    
                goal_stats = stats.get('goalStatistics', {})
                goals = goal_stats.get('goalsScoredTotal')
                assists = goal_stats.get('assists')
                
                if goals is not None:
                    agg[key]['goals'] += int(goals)
                if assists is not None:
                    agg[key]['assists'] += int(assists)
                    
            perf_rows = []
            for (s, comp, cl), val in agg.items():
                comp_name = LEAGUE_MAP[comp]['name'] if comp in LEAGUE_MAP else comp
                perf_rows.append({
                    'player_id': player_id,
                    'season': s,
                    'competition_id': comp,
                    'competition_name': comp_name,
                    'club_id': cl,
                    'club_name': None,
                    'appearances': val['appearances'],
                    'goals': val['goals'],
                    'assists': val['assists'],
                    'minutes': val['minutes']
                })
            df_perf = pd.DataFrame(perf_rows)
        except Exception as e:
            print(f"Error fetching performance for {player_id}: {e}")
            
        # 3. Piaci érték lekérése háttér fetch-el
        df_mv = pd.DataFrame()
        js_mv = f"""
        fetch('/ceapi/marketValueDevelopment/graph/{player_id}')
            .then(response => {{
                if (!response.ok) throw new Error('HTTP status ' + response.status);
                return response.json();
            }})
        """
        try:
            mv_data = self.page.evaluate(js_mv)
            mv_list = mv_data.get('list', [])
            mv_rows = []
            for entry in mv_list:
                mv_rows.append({
                    'player_id': player_id,
                    'timestamp': entry.get('x'),
                    'market_value': entry.get('y'),
                    'club_name': entry.get('verein'),
                    'date_string': entry.get('datum_mw'),
                    'market_value_formatted': entry.get('mw')
                })
            df_mv = pd.DataFrame(mv_rows)
        except Exception as e:
            print(f"Error fetching market value for {player_id}: {e}")
            
        # 4. Sérülések lekérése háttér fetch-el (több oldalas kezeléssel és a Total táblázattal)
        df_inj = pd.DataFrame()
        df_inj_total = pd.DataFrame()
        
        js_inj_page = lambda page_num: f"""
        fetch('/spieler/verletzungen/spieler/{player_id}/page/{page_num}')
            .then(response => {{
                if (!response.ok) throw new Error('HTTP status ' + response.status);
                return response.text();
            }})
        """
        
        try:
            # Letöltjük az 1. oldalt
            html_p1 = self.page.evaluate(js_inj_page(1))
            soup_p1 = BeautifulSoup(html_p1, 'html.parser')
            
            # Segédfüggvény az egyedi sérülések beolvasásához
            def parse_detail_table(soup):
                table = soup.find('table', class_='items')
                if not table:
                    return []
                rows = table.select('tbody tr')
                page_injuries = []
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) < 5:
                        continue
                    season_val = cols[0].text.strip()
                    injury_type = cols[1].text.strip()
                    start_date = cols[2].text.strip()
                    end_date = cols[3].text.strip()
                    
                    days_raw = cols[4].text.strip().replace(" days", "").replace(" day", "").replace("-", "0").replace(".", "")
                    try:
                        days_missed = int(days_raw) if days_raw else 0
                    except ValueError:
                        days_missed = 0
                        
                    games_missed = 0
                    games_span = cols[4].find('span')
                    if games_span:
                        games_raw = games_span.text.strip().replace(" matches", "").replace(" match", "").replace("-", "0")
                        try:
                            games_missed = int(games_raw) if games_raw else 0
                        except ValueError:
                            games_missed = 0
                    else:
                        if len(cols) >= 6:
                            games_raw = cols[5].text.strip().replace("-", "0")
                            try:
                                games_missed = int(games_raw) if games_raw else 0
                            except ValueError:
                                games_missed = 0
                                
                    page_injuries.append({
                        'player_id': player_id,
                        'season': season_val,
                        'injury_type': injury_type,
                        'start_date': start_date,
                        'end_date': end_date,
                        'days_missed': days_missed,
                        'games_missed': games_missed
                    })
                return page_injuries

            detailed_injuries = parse_detail_table(soup_p1)
            
            # Total táblázat beolvasása (ha létezik az oldalon)
            total_rows = []
            boxes = soup_p1.find_all('div', class_='box')
            total_box = None
            for b in boxes:
                headline = b.select_one('.content-box-headline, .table-header, .box-header')
                if headline and "total" in headline.text.lower():
                    total_box = b
                    break
                    
            if total_box:
                table_total = total_box.find('table', class_='items')
                if table_total:
                    rows_total = table_total.select('tbody tr')
                    for row in rows_total:
                        cols = row.find_all('td')
                        if len(cols) < 4:
                            continue
                        season_val = cols[0].text.strip()
                        days_val = int(cols[1].text.strip().replace(" days", "").replace(" day", "").replace("-", "0").replace(".", ""))
                        count_val = int(cols[2].text.strip().replace("-", "0"))
                        games_val = int(cols[3].text.strip().replace("-", "0"))
                        total_rows.append({
                            'player_id': player_id,
                            'season': season_val,
                            'days_missed_total': days_val,
                            'injury_count_total': count_val,
                            'games_missed_total': games_val
                        })
                df_inj_total = pd.DataFrame(total_rows)
                
            # Lapozás detektálása
            max_page = 1
            pagination = soup_p1.find('ul', class_='tm-pagination')
            if pagination:
                links = pagination.select('li.tm-pagination__list-item a.tm-pagination__link')
                for link in links:
                    text_link = link.text.strip()
                    if text_link.isdigit():
                        max_page = max(max_page, int(text_link))
                        
            # További oldalak lekérése háttér fetch-el
            js_inj_page = lambda page_num: f"""
            fetch('/spieler/verletzungen/spieler/{player_id}/page/{page_num}')
                .then(response => {{
                    if (!response.ok) throw new Error('HTTP status ' + response.status);
                    return response.text();
                }})
            """
            for p in range(2, max_page + 1):
                try:
                    html_p = self.page.evaluate(js_inj_page(p)) if self.page else self._get_url_html(f"{url}/page/{p}")
                    soup_p = BeautifulSoup(html_p, 'html.parser')
                    p_injuries = parse_detail_table(soup_p)
                    detailed_injuries.extend(p_injuries)
                    time.sleep(0.3)
                except Exception as e_page:
                    print(f"Error fetching injury page {p} for {player_id}: {e_page}")
                    
            return pd.DataFrame(detailed_injuries)
        except Exception as e:
            print(f"Error fetching injuries for {player_id}: {e}")
            return details, df_perf, df_mv, df_inj, df_inj_total

    def get_player_details(self, player_id):
        """Lekéri egy játékos metaadatait a profiloldalról, cache-elés támogatásával."""
        if str(player_id) in self.players_cache:
            return self.players_cache[str(player_id)]
            
        url = f"https://www.transfermarkt.com/spieler/profil/spieler/{player_id}"
        html = self._get_url_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        
        data = {
            'player_id': player_id,
            'name': None,
            'birth_date': None,
            'nationality': None,
            'primary_position': None,
            'foot': None,
            'height': None
        }
        
        name_header = soup.select_one('h1.data-header__headline-wrapper')
        if name_header:
            data['name'] = name_header.get_text(strip=True).replace('#', '').strip()
            data['name'] = re.sub(r'^\d+\s*', '', data['name'])
            
        info_table = soup.find('div', class_='info-table')
        if info_table:
            # Segédkereső függvény a címkékhez
            def find_info_value(label_text):
                label_span = info_table.find(lambda tag: tag.name == 'span' and label_text in tag.text)
                if label_span:
                    val = label_span.find_next_sibling('span')
                    if val:
                        # Ha a nemzetiség flag kép
                        if label_text == 'Citizenship':
                            flags = val.find_all('img', class_='flaggenrahmen')
                            if flags:
                                return "; ".join([img.get('title') for img in flags])
                        return val.get_text(strip=True)
                return None
                
            data['birth_date'] = find_info_value('Date of birth') or find_info_value('Birth date')
            data['nationality'] = find_info_value('Citizenship')
            data['primary_position'] = find_info_value('Position')
            data['foot'] = find_info_value('Foot')
            data['height'] = find_info_value('Height')
            
        self.players_cache[str(player_id)] = data
        return data

    def get_player_performance(self, player_id):
        """Lekéri a játékos szezononkénti statisztikáit a belső CEAPI JSON API végpontról."""
        from config import LEAGUE_MAP
        url = f"https://www.transfermarkt.com/ceapi/performance-game/{player_id}"
        
        try:
            data = self._get_url_json(url)
        except Exception as e:
            print(f"Error fetching performance API for {player_id}: {e}. Falling back to empty DataFrame.")
            return pd.DataFrame()
            
        performances = data.get('data', {}).get('performance', [])
        
        agg = {}
        for game in performances:
            g_info = game.get('gameInformation', {})
            
            # Csak klubcsapatok meccseit vesszük figyelembe
            if g_info.get('isNationalGame', False):
                continue
                
            season = g_info.get('season', {}).get('display')
            comp_id = g_info.get('competitionId')
            club_info = game.get('clubsInformation', {}).get('club', {})
            club_id = club_info.get('clubId')
            
            if not all([season, comp_id, club_id]):
                continue
                
            key = (season, comp_id, club_id)
            if key not in agg:
                agg[key] = {
                    'appearances': 0,
                    'minutes': 0,
                    'goals': 0,
                    'assists': 0
                }
                
            stats = game.get('statistics', {})
            play_time = stats.get('playingTimeStatistics', {})
            mins = play_time.get('playedMinutes')
            
            if mins is not None:
                agg[key]['appearances'] += 1
                agg[key]['minutes'] += int(mins)
                
            goal_stats = stats.get('goalStatistics', {})
            goals = goal_stats.get('goalsScoredTotal')
            assists = goal_stats.get('assists')
            
            if goals is not None:
                agg[key]['goals'] += int(goals)
            if assists is not None:
                agg[key]['assists'] += int(assists)
                
        performance_data = []
        for (s, comp, cl), val in agg.items():
            comp_name = LEAGUE_MAP[comp]['name'] if comp in LEAGUE_MAP else comp
            performance_data.append({
                'player_id': player_id,
                'season': s,
                'competition_id': comp,
                'competition_name': comp_name,
                'club_id': cl,
                'club_name': None,  # Később a cache-ből/transzferekből kikereshető
                'appearances': val['appearances'],
                'goals': val['goals'],
                'assists': val['assists'],
                'minutes': val['minutes']
            })
            
        return pd.DataFrame(performance_data)

    def get_player_market_value_history(self, player_id):
        """Lekéri a játékos piaci érték idősorát a belső CEAPI végpontról."""
        url = f"https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{player_id}"
        data = self._get_url_json(url)
        
        mv_list = data.get('list', [])
        results = []
        
        for entry in mv_list:
            # x a timestamp ms-ben, y a piaci érték, verein a klub, datum_mw a dátum string
            x_timestamp = entry.get('x')
            y_value = entry.get('y')
            club_name = entry.get('verein')
            date_str = entry.get('datum_mw')
            
            # Formázott piaci érték string (pl. "€5.00m")
            mw_str = entry.get('mw')
            
            results.append({
                'player_id': player_id,
                'timestamp': x_timestamp,
                'market_value': y_value,
                'club_name': club_name,
                'date_string': date_str,
                'market_value_formatted': mw_str
            })
            
        return pd.DataFrame(results)

    def get_player_injury_history(self, player_id):
        """Lekéri a játékos teljes, többoldalas sérüléstörténetét HTML scraping és háttér-fetch segítségével."""
        url = f"https://www.transfermarkt.com/spieler/verletzungen/spieler/{player_id}"
        html = self._get_url_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Segédfüggvény az egyedi sérülések beolvasásához
        def parse_detail_table(soup):
            table = soup.find('table', class_='items')
            if not table:
                return []
            rows = table.select('tbody tr')
            page_injuries = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 5:
                    continue
                season_val = cols[0].text.strip()
                injury_type = cols[1].text.strip()
                start_date = cols[2].text.strip()
                end_date = cols[3].text.strip()
                
                days_raw = cols[4].text.strip().replace(" days", "").replace(" day", "").replace("-", "0").replace(".", "")
                try:
                    days_missed = int(days_raw) if days_raw else 0
                except ValueError:
                    days_missed = 0
                    
                games_missed = 0
                games_span = cols[4].find('span')
                if games_span:
                    games_raw = games_span.text.strip().replace(" matches", "").replace(" match", "").replace("-", "0")
                    try:
                        games_missed = int(games_raw) if games_raw else 0
                    except ValueError:
                        games_missed = 0
                else:
                    if len(cols) >= 6:
                        games_raw = cols[5].text.strip().replace("-", "0")
                        try:
                            games_missed = int(games_raw) if games_raw else 0
                        except ValueError:
                            games_missed = 0
                            
                page_injuries.append({
                    'player_id': player_id,
                    'season': season_val,
                    'injury_type': injury_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'days_missed': days_missed,
                    'games_missed': games_missed
                })
            return page_injuries

        detailed_injuries = parse_detail_table(soup)
        
        # Lapozás detektálása
        max_page = 1
        pagination = soup.find('ul', class_='tm-pagination')
        if pagination:
            links = pagination.select('li.tm-pagination__list-item a.tm-pagination__link')
            for link in links:
                text_link = link.text.strip()
                if text_link.isdigit():
                    max_page = max(max_page, int(text_link))
                    
        # További oldalak lekérése háttér fetch-el
        js_inj_page = lambda page_num: f"""
        fetch('/spieler/verletzungen/spieler/{player_id}/page/{page_num}')
            .then(response => {{
                if (!response.ok) throw new Error('HTTP status ' + response.status);
                return response.text();
            }})
        """
        for p in range(2, max_page + 1):
            try:
                html_p = self.page.evaluate(js_inj_page(p)) if self.page else self._get_url_html(f"{url}/page/{p}")
                soup_p = BeautifulSoup(html_p, 'html.parser')
                p_injuries = parse_detail_table(soup_p)
                detailed_injuries.extend(p_injuries)
                time.sleep(0.3)
            except Exception as e_page:
                print(f"Error fetching injury page {p} for {player_id}: {e_page}")
                
        return pd.DataFrame(detailed_injuries)

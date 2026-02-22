import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from datetime import datetime

class LoviScraper:
    def __init__(self):
        self.base_url = "https://bet.lovi.hu"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "hu-HU,hu;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        self.session = requests.Session()
        # Initialize cookies
        try:
            self.session.get(f"{self.base_url}/hu", headers=self.headers, timeout=10)
        except Exception as e:
            print(f"Warning: Could not initialize session: {e}")

    def get_meetings_for_date(self, date_str):
        """
        Lekéri az adott napi HUN helyszíneket és azok futamait.
        """
        url = f"{self.base_url}/hu/races?country=HUN&date={date_str}"
        print(f"Fetching HUN meetings for {date_str}...")
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching date {date_str}: {e}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. HUN szekció megkeresése
        hun_wrapper = soup.select_one('.countryWrapper.HUN')
        if not hun_wrapper:
            print(f"No Hungarian (HUN) section found on {date_str}.")
            return []

        # 2. Megkeressük a meeting azonosítókat a HUN szekción belül
        meeting_ids = []
        for wrapper in hun_wrapper.select('.meetingWrapper'):
            m_id = wrapper.get('data-meeting')
            if m_id:
                meeting_ids.append(m_id)
        
        if not meeting_ids:
            print(f"No meeting IDs found in HUN section.")
            return []

        print(f"Found HUN meeting IDs: {meeting_ids}")
        
        # 3. Minden meetinghez lekérjük a futamokat
        all_hun_race_ids = []
        for m_id in meeting_ids:
            # A meeting lekéréséhez NEM kell a /hu prefix, mert úgy 404-et adhat
            m_url = f"{self.base_url}/meetings/meeting?id={m_id}"
            try:
                # AJAX fejlécek
                ajax_headers = self.headers.copy()
                ajax_headers["X-Requested-With"] = "XMLHttpRequest"
                
                resp = self.session.get(m_url, headers=ajax_headers, timeout=10)
                if resp.status_code == 200:
                    # Lehet JSON vagy direkt HTML is
                    try:
                        data = resp.json()
                        content = data.get('content', '')
                    except:
                        content = resp.text
                        
                    m_soup = BeautifulSoup(content, 'html.parser')
                    links = m_soup.find_all('a', href=re.compile(r'/race\?id=\d+'))
                    # Ha nincs <a>, nézzük a li[data-url] attribútumokat
                    if not links:
                        lis = m_soup.select('li[data-url]')
                        for li in lis:
                            match = re.search(r'id=(\d+)', li['data-url'])
                            if match:
                                r_id = match.group(1)
                                if r_id not in all_hun_race_ids:
                                    all_hun_race_ids.append(r_id)
                    else:
                        for a in links:
                            match = re.search(r'id=(\d+)', a['href'])
                            if match:
                                r_id = match.group(1)
                                if r_id not in all_hun_race_ids:
                                    all_hun_race_ids.append(r_id)
            except Exception as e:
                print(f"Error fetching meeting {m_id}: {e}")

        # 4. Biztonsági mentés: ha az AJAX nem ment, nézzük meg a HUN wrappert linkekért
        if not all_hun_race_ids:
            links = hun_wrapper.find_all('a', href=re.compile(r'/race\?id=\d+'))
            for a in links:
                match = re.search(r'id=(\d+)', a['href'])
                if match and match.group(1) not in all_hun_race_ids:
                    all_hun_race_ids.append(match.group(1))

        return all_hun_race_ids

    def parse_race_page(self, race_id):
        """
        Lekér és feldolgoz egy konkrét futamot.
        """
        url = f"{self.base_url}/hu/race?id={race_id}"
        print(f"  Fetching race {race_id}...")
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"  Error fetching race {race_id}: {e}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Metaadatok
        title_elem = soup.select_one('.racecard-title-box')
        title = title_elem.get_text(strip=True) if title_elem else "N/A"
        
        info_list = soup.select('.racecard-info-list li')
        meta = {}
        for li in info_list:
            text = li.get_text(strip=True)
            
            # 1. Start időpont (pl. 14:00)
            if re.match(r'^\d{1,2}:\d{2}$', text):
                meta["start"] = text
                continue
            
            # 2. Indulók száma (vagy "Indulók száma: 8" vagy csak egy szám "8")
            if 'Indulók' in text and ':' in text:
                try:
                    meta["participants_count"] = int(re.search(r'\d+', text).group())
                    continue
                except: pass
            
            if text.isdigit():
                meta["participants_count"] = int(text)
                continue

            # 3. Egyéb kulcs-érték párok (Táv, Összdíjazás, stb.)
            if ':' in text:
                parts = text.split(':', 1)
                k = parts[0].strip()
                v = parts[1].strip()
                # Ha a kulcs véletlenül egy szám (pl. "14": "00"), az valószínűleg egy félre-splitelt időpont
                if k.isdigit() and v.isdigit():
                    meta["start"] = f"{k}:{v}"
                else:
                    meta[k] = v
            else:
                # Ha nem tudjuk mi ez, de még nincs 'start', hátha az
                if 'start' not in meta and ':' in text:
                    meta['start'] = text
                
        # Eredmények (ha már lefutott)
        results = []
        finish_table = soup.select_one('.finishTable')
        if finish_table:
            for row in finish_table.select('tbody tr'):
                cols = row.find_all('td')
                if len(cols) >= 5:
                    results.append({
                        "rank": cols[0].get_text(strip=True),
                        "program_num": cols[1].get_text(strip=True),
                        "horse": cols[2].get_text(strip=True),
                        "odds_at_finish": cols[3].get_text(strip=True),
                        "driver": cols[4].get_text(strip=True)
                    })
        
        # Osztalékok
        dividends = []
        finish_bet_types = soup.select('#finishBetType li:not(.resultHeader)')
        for li in finish_bet_types:
            bet_type = li.select_one('.bettypes span').get_text(strip=True)
            table = li.select_one('table')
            if table:
                for row in table.select('tbody tr'):
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        dividends.append({
                            "type": bet_type,
                            "combination": cols[0].get_text(strip=True),
                            "desc": cols[1].get_text(strip=True),
                            "payout": cols[2].get_text(strip=True)
                        })

        # Indulók és indulási oddsok
        participants = []
        racecard_items = soup.select('.racecardList li')
        for li in racecard_items:
            name_elem = li.select_one('.name a')
            if not name_elem: continue
            
            p_num = li.select_one('.counter .count1').get_text(strip=True) if li.select_one('.counter .count1') else "N/A"
            horse_name = name_elem.get_text(strip=True)
            odds = li.select_one('.odds').get_text(strip=True) if li.select_one('.odds') else "-"
            
            j_t = li.select_one('.jockeytrainer')
            jockey = "N/A"
            trainer = "N/A"
            if j_t:
                spans = j_t.find_all('span')
                if len(spans) >= 1: jockey = spans[0].get_text(strip=True)
                if len(spans) >= 2: trainer = spans[1].get_text(strip=True)
            
            dist_elem = li.select_one('.distance')
            dist = dist_elem.get_text(strip=True) if dist_elem else "N/A"
            
            participants.append({
                "program_num": p_num,
                "horse": horse_name,
                "distance": dist,
                "jockey": jockey,
                "trainer": trainer,
                "starting_odds": odds
            })
            
        return {
            "race_id": race_id,
            "title": title,
            "meta": meta,
            "participants": participants,
            "results": results,
            "dividends": dividends
        }

    def scrape_date(self, date_str):
        race_ids = self.get_meetings_for_date(date_str)
        if not race_ids:
            print(f"No races found for {date_str}.")
            return None
            
        print(f"Found {len(race_ids)} potential race IDs. Starting detailed scrape...")
        
        all_data = {
            "date": date_str,
            "source": "bet.lovi.hu",
            "scrape_time": datetime.now().isoformat(),
            "races": []
        }
        
        for r_id in race_ids:
            race_data = self.parse_race_page(r_id)
            if race_data:
                all_data["races"].append(race_data)
            time.sleep(1) # Polite scraping
            
        return all_data

if __name__ == "__main__":
    import sys
    target_date = sys.argv[1] if len(sys.argv) > 1 else "2025-04-12"
    
    scraper = LoviScraper()
    result = scraper.scrape_date(target_date)
    
    if result:
        output_file = f"research/lovi_scrape_{target_date}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nDone! Saved {len(result['races'])} races to {output_file}")
    else:
        print("Scrape failed or no data found.")

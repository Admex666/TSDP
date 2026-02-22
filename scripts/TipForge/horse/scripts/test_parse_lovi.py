from bs4 import BeautifulSoup
import os

def parse_lovi_race(html_path):
    if not os.path.exists(html_path):
        print(f"File {html_path} not found")
        return
        
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # Race Meta
    race_title = soup.select_one('.racecard-title-box')
    race_title = race_title.get_text(strip=True) if race_title else "N/A"
    print(f"Race: {race_title}")
    
    # Participants in the list
    participants = []
    # The table with finish order
    finish_table = soup.select_one('.finishTable')
    if finish_table:
        print("\nFinal Results:")
        for row in finish_table.select('tbody tr'):
            cols = row.find_all('td')
            if len(cols) >= 5:
                place = cols[0].get_text(strip=True)
                program_num = cols[1].get_text(strip=True)
                horse_name = cols[2].get_text(strip=True)
                odds = cols[3].get_text(strip=True)
                driver = cols[4].get_text(strip=True)
                print(f"#{place} | {program_num}. {horse_name} | Odds: {odds} | Driver: {driver}")

    # Full Participant List (with starting odds)
    print("\nAll Participants & Starting Odds:")
    racecard_list = soup.select('.racecardList li')
    for li in racecard_list:
        name_elem = li.select_one('.name a')
        odds_elem = li.select_one('.odds')
        jockey_elem = li.select_one('.jockeytrainer span:first-child')
        
        if name_elem and odds_elem:
            horse_name = name_elem.get_text(strip=True)
            odds = odds_elem.get_text(strip=True)
            jockey = jockey_elem.get_text(strip=True) if jockey_elem else "N/A"
            print(f"Horse: {horse_name} | Odds: {odds} | Jockey: {jockey}")

import sys

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else 'race_page.html'
    parse_lovi_race(fname)

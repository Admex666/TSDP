import os
from bs4 import BeautifulSoup
import pandas as pd

raw_dir = "data/reference"

def inspect_schedule():
    files = [f for f in os.listdir(raw_dir) if "schedule" in f]
    if not files: return
    with open(os.path.join(raw_dir, files[0]), 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        # Find schedule table
        table = soup.find('table', attrs={'id': lambda x: x and 'sched' in x})
        if table:
            print(f"Schedule Table ID: {table.get('id')}")
            headers = [th.text.strip() for th in table.select('thead tr th')]
            print(f"Schedule Columns: {headers[:10]}...")

def inspect_match():
    files = [f for f in os.listdir(raw_dir) if "match.html" in f]
    if not files: return
    with open(os.path.join(raw_dir, files[0]), 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        lineups = soup.find_all('div', class_='lineup')
        print(f"Found {len(lineups)} lineup divs")
        if lineups:
            table = lineups[0].find('table')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    if row.find('a'):
                        print(f"Sample Lineup Row: {row}")
                        break

def inspect_log():
    for fname in ["player_summary.html", "player_keepers.html"]:
        if fname not in os.listdir(raw_dir): continue
        print(f"Inspecting {fname}...")
        with open(os.path.join(raw_dir, fname), 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            # Try to find table with id containing 'matchlogs'
            tables = soup.find_all('table', attrs={'id': lambda x: x and 'matchlogs' in x})
            if not tables:
                 print(f"No matchlogs table found in {fname}. All table IDs:")
                 all_tables = soup.find_all('table')
                 for t in all_tables:
                     print(f" - {t.get('id')}")
            
            for table in tables:
                print(f"Found Table ID: {table.get('id')}")
                headers = [th.text.strip() for th in table.select('thead tr th')]
                print(f"Columns: {headers[:15]}...")


if __name__ == "__main__":
    try:
        inspect_schedule()
        print("-" * 20)
        inspect_match()
        print("-" * 20)
        inspect_log()
    except Exception as e:
        print(e)

import os
from bs4 import BeautifulSoup
import pandas as pd

raw_dir = "data/raw"

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
    files = [f for f in os.listdir(raw_dir) if "matches" in f and "Burnley" in f]
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
    files = [f for f in os.listdir(raw_dir) if "players" in f]
    if not files: return
    with open(os.path.join(raw_dir, files[0]), 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        table = soup.find('table', attrs={'id': lambda x: x and 'matchlogs' in x})
        if table:
            print(f"Log Table ID: {table.get('id')}")
            # check columns
            headers = [th.text.strip() for th in table.select('thead tr th')]
            print(f"Log Columns: {headers[:15]}...")

if __name__ == "__main__":
    try:
        inspect_schedule()
        print("-" * 20)
        inspect_match()
        print("-" * 20)
        inspect_log()
    except Exception as e:
        print(e)

import os
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "data_mock"
RAW_DIR = os.path.join(DATA_DIR, "raw")

def generate_schedule_html(filename):
    # Minimal FBref schedule table
    html = """
    <html>
    <body>
    <table id="sched_2023-2024_9_1">
        <thead>
            <tr>
                <th>Wk</th><th>Day</th><th>Date</th><th>Time</th><th>Home</th><th>xG</th><th>Score</th><th>xG</th><th>Away</th><th>Match Report</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td><td>Sat</td><td data-stat="date">2023-08-01</td><td>12:30</td>
                <td data-stat="home_team">Training FC</td><td>1.2</td><td data-stat="score">1–1</td><td>1.1</td>
                <td data-stat="away_team">Sparring Utd</td>
                <td data-stat="match_report"><a href="/en/matches/mock0/Training-Sparring">Match Report</a></td>
            </tr>
             <tr>
                <td>1</td><td>Sat</td><td data-stat="date">2023-08-11</td><td>15:00</td>
                <td data-stat="home_team">Burnley</td><td>0.5</td><td data-stat="score">0–3</td><td>2.1</td>
                <td data-stat="away_team">Manchester City</td>
                <td data-stat="match_report"><a href="/en/matches/mock1/Burnley-City">Match Report</a></td>
            </tr>
        </tbody>
    </table>
    </body>
    </html>
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated schedule: {filename}")

def generate_match_lineup_html(filename, home_team="Home", away_team="Away"):
    # Matches real structure with Position column
    html = f"""
    <html>
    <body>
    <div class="lineup" id="a">
        <table>
            <thead><tr><th>Player</th><th>Pos</th></tr></thead>
            <tbody>
                <tr>
                    <td><a href="/en/players/p1/GK_Home">GK Home</a></td>
                    <td data-stat="position">GK</td>
                    <td class="shirtnumber">1</td>
                </tr>
                <tr>
                    <td><a href="/en/players/p2/DEF_Home">DEF Home</a></td>
                    <td data-stat="position">DF</td>
                    <td class="shirtnumber">4</td>
                </tr>
            </tbody>
        </table>
    </div>
    <div class="lineup" id="b">
        <table>
             <thead><tr><th>Player</th><th>Pos</th></tr></thead>
            <tbody>
                <tr>
                    <td><a href="/en/players/p3/GK_Away">GK Away</a></td>
                    <td data-stat="position">GK</td>
                    <td class="shirtnumber">1</td>
                </tr>
                <tr>
                    <td><a href="/en/players/p4/ATT_Away">ATT Away</a></td>
                    <td data-stat="position">FW</td>
                    <td class="shirtnumber">9</td>
                </tr>
            </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated match: {filename}")

def generate_player_log_html(filename, player_name):
    # Matches real ID matchlogs_all
    html = f"""
    <html>
    <body>
    <h1>{player_name} Match Logs</h1>
    <table id="matchlogs_all">
        <thead>
            <tr>
                <th>Date</th><th>Comp</th><th>Round</th><th>Venue</th><th>Result</th>
                <th>Gls</th><th>Ast</th><th>Sh</th><th>SoT</th><th>xG</th>
                <th>Tkl</th><th>Int</th><th>Blocks</th><th>Err</th> <!-- Added Err for GK/DEF -->
                <th>SCA</th><th>GCA</th><th>Min</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td data-stat="date">2023-05-01</td><td>Premier League</td><td>Matchweek 30</td><td>Home</td><td>W 2-0</td>
                <td data-stat="goals">1</td><td data-stat="assists">0</td><td data-stat="shots">3</td><td data-stat="shots_on_target">2</td><td data-stat="xg">0.8</td>
                <td data-stat="tackles">1</td><td data-stat="interceptions">0</td><td data-stat="blocks">0</td><td data-stat="errors">0</td>
                <td data-stat="sca">2</td><td data-stat="gca">1</td><td data-stat="minutes">90</td>
            </tr>
            <tr>
                <td data-stat="date">2023-05-10</td><td>Premier League</td><td>Matchweek 32</td><td>Away</td><td>D 1-1</td>
                <td data-stat="goals">0</td><td data-stat="assists">1</td><td data-stat="shots">1</td><td data-stat="shots_on_target">0</td><td data-stat="xg">0.1</td>
                <td data-stat="tackles">2</td><td data-stat="interceptions">1</td><td data-stat="blocks">1</td><td data-stat="errors">0</td>
                <td data-stat="sca">3</td><td data-stat="gca">0</td><td data-stat="minutes">90</td>
            </tr>
        </tbody>
    </table>
    </body>
    </html>
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated log: {filename}")

import sys
sys.path.append(os.getcwd()) # Ensure src is importable
from src.scraping.fbref_loader import FBrefDataLoader

def main():
    loader = FBrefDataLoader(data_dir=DATA_DIR) 
    # This creates data_mock/raw automatically via __init__? 
    # FBrefDataLoader init does `self.raw_dir.mkdir(parents=True, exist_ok=True)`
    # So we just need to init it.
    
    # 1. Schedule
    sched_url = "https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures"
    sched_path = loader._get_cache_path(sched_url)
    generate_schedule_html(sched_path)
    
    # 2. Match
    # Training
    train_url = "https://fbref.com/en/matches/mock0/Training-Sparring"
    train_path = loader._get_cache_path(train_url)
    generate_match_lineup_html(train_path, "Training FC", "Sparring Utd")

    # Burnley City
    match_url = "https://fbref.com/en/matches/mock1/Burnley-City"
    match_path = loader._get_cache_path(match_url)
    generate_match_lineup_html(match_path, "Burnley", "Man City")
    
    # 3. Player Logs
    # IDs: p1, p2, p3, p4 from our lineups above
    ids = ["p1", "p2", "p3", "p4"]
    for pid in ids:
        # Summary log
        url = f"https://fbref.com/en/players/{pid}/matchlogs/2023-2024/summary"
        path = loader._get_cache_path(url)
        generate_player_log_html(path, f"Player {pid}")
        
        # Keepers log (for p1 and p3 who are GKs in our lineup)
        if pid in ["p1", "p3"]:
             url_keepers = f"https://fbref.com/en/players/{pid}/matchlogs/2023-2024/keepers"
             path_keepers = loader._get_cache_path(url_keepers)
             generate_player_log_html(path_keepers, f"Player {pid} Keepers")

if __name__ == "__main__":
    main()

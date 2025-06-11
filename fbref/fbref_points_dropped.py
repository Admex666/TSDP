#%% Get all fixtures
import pandas as pd
from TSDP.fbref import fbref_module as fbref

url_all_fixtures = 'https://fbref.com/en/squads/18bb7c10/2024-2025/matchlogs/c9/schedule/Arsenal-Scores-and-Fixtures-Premier-League'
squad_name = 'Arsenal'
fixture_urls = []

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from io import StringIO

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get(url_all_fixtures)

a_elements = driver.find_elements(By.TAG_NAME, "a")
for a in a_elements:
    href = a.get_attribute("href")
    if pd.isna(href) == False:
        if ('/matches/' in href) & (squad_name in href):
            fixture_urls.append(href)

driver.quit()

# remove duplicates
fixture_urls = list(set(fixture_urls))

#%% Get all goals scored in matches
goals_season = pd.DataFrame()
games = pd.DataFrame()

for url in fixture_urls:
    match_id = len(games)
    shots_game = fbref.scrape(url, 'div_shots_all')
    # rename cols
    cols_old = [col for col in shots_game.columns if 'Unnamed' in col]
    cols_new = [col.split('_')[-1] for col in shots_game.columns if 'Unnamed' in col]
    shots_game.rename(columns={old:new for old, new in zip(cols_old, cols_new)}, inplace=True)
    
    goals_game = shots_game[shots_game.Outcome == 'Goal'].copy().reset_index(drop=True)
    if goals_game.empty:
        points_dropped, goals_for, goals_against = 0, 0, 0
    else:
        goals_game[['goals_for_cum', 'goals_against_cum']] = 0,0
        
        # scoreline
        goals_for_cum, goals_against_cum = 0,0 
        for i in goals_game.index:
            if goals_game.loc[i, 'Squad'] == squad_name:
                goals_for_cum += 1
            else:
                goals_against_cum += 1
            goals_game.loc[i, ['goals_for_cum', 'goals_against_cum']] = goals_for_cum, goals_against_cum
                    
        # we can drop points if we are in the lead
        had_the_lead = any(goals_game['goals_for_cum'] > goals_game['goals_against_cum'])
        
        # Calculate points
        goals_for, goals_against = goals_game.loc[len(goals_game)-1, ['goals_for_cum', 'goals_against_cum']]
        points_for = 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
        
        if had_the_lead and (points_for != 3):
            points_dropped = 3 - points_for
        else:
            points_dropped = 0
        
        # append season goals df
        goals_game['match_id'] = match_id
        goals_game_f = goals_game[['match_id', 'Minute', 'Player', 'Squad']]
        goals_season = pd.concat([goals_season, goals_game_f])
    
    # append games df
    games.loc[match_id,'url'] = url
    games.loc[match_id, 'opponent'] = shots_game[shots_game.Squad != squad_name]['Squad'].iloc[0]
    games.loc[match_id, 'GF'] = goals_for
    games.loc[match_id, 'GA'] = goals_against
    games.loc[match_id, 'points_dropped'] = points_dropped
    print(f'{match_id}/{len(fixture_urls)}')

points_dropped_season = int(games['points_dropped'].sum())
print(f"{squad_name} dropped {points_dropped_season} points from the " +
      f"lead, which is {round(points_dropped_season/len(fixture_urls), 2)} per game.")

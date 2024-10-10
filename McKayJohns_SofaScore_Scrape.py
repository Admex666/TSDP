import json
import requests
import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

URL = str(input('Give me the SofaScore URL of the match.'))

if type(URL) != str:
    print("Error. URL is not a string.")
else:
    # Gathering data from Chromedriver
    options = webdriver.ChromeOptions()
    options.set_capability('goog:loggingPrefs',
                           {"performance": "ALL", "browser": "ALL"})
    
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()),
                              options=options)
    
    ## Manipulating the chrome window
    driver.set_page_load_timeout(10)
    
    try:
        driver.get(URL)
    except Exception as e:
        print("Error loading page:", e)
    
    ## Wait for some time
    time.sleep(5)
    
    # Scrolling down
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Capture logs
    logs_raw = driver.get_log("performance")
    logs = [json.loads(lr['message'])['message'] for lr in logs_raw]
    
    ## Debugging: print all network requests to check if 'shotmap' exists
    for x in logs:
        url_path = x.get('params', {}).get('request', {}).get('url', '')
        if 'shotmap' in url_path:
            print(f"Found shotmap request: {url_path}")
            print(f"Request ID: {x['params'].get('requestId')}")
            shotmap_request = x
            break
        
        
    else:
        print("No shotmap request found.")
        driver.quit()  # Properly close the browser
        raise SystemExit  # Exit the script cleanly
    ## Retrieve the response body
    try:
        request_id = shotmap_request['params']['requestId']
        shotmap_response = driver.execute_cdp_cmd('Network.getResponseBody',
                                                  {'requestId': request_id})
        shotmap_data = json.loads(shotmap_response['body'])['shotmap']
        print("Shotmap found")
    except Exception as e:
        print("Error retrieving shotmap data:", e)
        
    ## Debugging: print all network requests to check if 'lineups' exists
    for y in logs:
        url_path_lineup = y.get('params', {}).get('request', {}).get('url', '')
        if 'lineups' in url_path_lineup:
            print(f"Found lineups request: {url_path}")
            print(f"Request ID: {x['params'].get('requestId')}")
            lineup_request = y
            break
        
        
    else:
        print("No lineup request found.")
        driver.quit()  # Properly close the browser
        raise SystemExit  # Exit the script cleanly
    ## Retrieve the response body
    try:
        request_id_lineup = lineup_request['params']['requestId']
        # Make a GET request to the API
        lineup_response = requests.get(url_path_lineup)
        
        # Raise an exception for bad status codes
        lineup_response.raise_for_status()
        
        # Get the response text as a string
        json_string_lineup = lineup_response.text

        lineup_data = json.loads(json_string_lineup)
        print('Lineups found')
    except Exception as e:
        print("Error retrieving lineup data:", e)
        
    
    # Scraping team names
    from bs4 import BeautifulSoup
    import time
    
    # Set up WebDriver
    # Open the webpage
    driver.get(URL)
    
    # Wait for the page to fully load
    time.sleep(5)  # You can adjust this or replace with WebDriverWait
    
    # Get the fully rendered page source
    page_source = driver.page_source
    
    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Locate the image element using the CSS selector provided
    image_element1 = soup.select_one('div:nth-child(1) > div > a > div > img')
    image_element2 = soup.select_one('div:nth-child(3) > div > a > div > img')
    
    if image_element1:
        # Extract the 'alt' attribute from the image element
        home_name = image_element1.get('alt')
        print(home_name)
    else:
        print("Image element (home team) not found")
    
    if image_element2:
        # Extract the 'alt' attribute from the image element
        away_name = image_element2.get('alt')
        print(away_name)
    else:
        print("Image element (away team) not found")
    
    driver.quit()



# Creating Shotmap DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = []
for shot in shotmap_data:
    shot_info = {
        'timeSeconds': shot['timeSeconds'],
        'Player Name': shot['player']['name'],
        'Team': shot['isHome'],
        'Shot Type': shot['shotType'],
        'X Coordinate': shot['playerCoordinates']['x'],
        'Y Coordinate': shot['playerCoordinates']['y'],
        'Position': shot['player']['position'],
        'xG': shot['xg'],
        'xGOT': shot['xgot']
    }
    data.append(shot_info)

df = pd.DataFrame(data)
df = df.sort_values(by='timeSeconds', ascending=True)
df.reset_index(inplace=True, )
df.drop(columns=['index'], inplace=True)

## transforming columns
df['Team'] = np.where(df['Team'] == True, home_name, away_name)
df['team_cumulative_xG'] = df.groupby('Team')['xG'].cumsum()

# Function to extract player information from lineup_data
def extract_player_info(player_data):
    player = player_data['player']
    return {
        'Player Name': player['name'],
        'Position': player['position'],
        'Jersey Number': player['jerseyNumber'],
        'Country': player['country']['name'],
        'Height': player.get('height', None),
        'Substitute': player_data['substitute'],
        'Rating': player_data.get('statistics', {}).get('rating', None)
    }

# Extract home and away team lineups
home_lineup = [extract_player_info(p) for p in lineup_data['home']['players']]
away_lineup = [extract_player_info(p) for p in lineup_data['away']['players']]

# Create DataFrames for home and away teams
df_home = pd.DataFrame(home_lineup)
df_away = pd.DataFrame(away_lineup)

# Add a column to identify the team
df_home['Team'] = 'Home'
df_away['Team'] = 'Away'

# Combine the DataFrames
df_lineup = pd.concat([df_home, df_away], ignore_index=True)

# Sort the DataFrame by team and substitute status
df_lineup = df_lineup.sort_values(['Team', 'Substitute', 'Position'])

# Reset the index
df_lineup = df_lineup.reset_index(drop=True)

df_lineup.Team = np.where(df_lineup.Team == 'Home',
                          home_name,
                          away_name)



# Displaying Cumulative xG by time

## Defining the values for each graph
plot_x_home = df[df['Team'] == home_name]['timeSeconds']
plot_y_home = df[df['Team'] == home_name]['team_cumulative_xG']
plot_x_away = df[df['Team'] == away_name]['timeSeconds']
plot_y_away = df[df['Team'] == away_name]['team_cumulative_xG']

## Extending the Time column, and adding max, because that's how we can shape it like a staircase 
plot_x_home_extended = pd.concat([plot_x_home, plot_x_home-1], ignore_index=True).sort_values(ascending=True).reset_index().drop(columns=['index'])
plot_x_away_extended = pd.concat([plot_x_away, plot_x_away-1], ignore_index=True).sort_values(ascending=True).reset_index().drop(columns=['index'])

plot_x_home_extended.loc[len(plot_x_home_extended)] = df['timeSeconds'].max()
plot_x_away_extended.loc[len(plot_x_away_extended)] = df['timeSeconds'].max()

## Doing the same for cum. xG, but here adding 0 as a value
plot_y_home_extended = pd.concat([plot_y_home, plot_y_home], ignore_index=True).sort_values(ascending=True).reset_index().drop(columns=['index'])
plot_y_away_extended = pd.concat([plot_y_away, plot_y_away], ignore_index=True).sort_values(ascending=True).reset_index().drop(columns=['index'])

plot_y_home_extended.loc[len(plot_y_home_extended)] = 0
plot_y_home_extended.sort_values(by="team_cumulative_xG", ascending = True, inplace=True)

plot_y_away_extended.loc[len(plot_y_away_extended)] = 0
plot_y_away_extended.sort_values(by="team_cumulative_xG", ascending = True, inplace=True)

## Showing the plots
plt.plot(plot_x_home_extended/60, plot_y_home_extended, color='g', label= home_name)
plt.plot(plot_x_away_extended/60, plot_y_away_extended, color='r', label= away_name)

plt.scatter(
            df[df['Shot Type'] == "goal"].iloc[:,0]/60, 
            df[df['Shot Type'] == "goal"].iloc[:,9],
            label='Goals')

## Setting limits, making it nicer
plt.xlim(df['timeSeconds'].min()/60, df['timeSeconds'].max()/60+0.05)
plt.ylim(df['team_cumulative_xG'].min(), df['team_cumulative_xG'].max()+0.05)

plt.xlabel('Minutes')
plt.ylabel('Expected goals')
plt.title('Cumulative expected goals')
plt.grid(axis='x')
plt.legend()
plt.text(10, df['team_cumulative_xG'].max()*0.7,
         '@adamjakus99',
         fontsize = 11,
         color='darkblue')

## Finalllyyy
plt.show()



# Seeing players individually
df_players_shots = df[['Player Name', 'xG', 'xGOT', 'Shot Type']]
df_players_shots['Goal'] = np.where(df_players_shots['Shot Type'] == 'goal', 1, 0)
df_playersum = df_players_shots.groupby('Player Name', as_index=False)[['xG','xGOT', 'Goal']].sum()

## Merging into playersum
df_playersum = pd.merge(df_playersum, df_lineup[['Player Name', 'Team']], how='left', on='Player Name')

## Creating player names with team in ()
df_playersum['Player_team'] = df_playersum['Player Name'].astype(str) + ' (' + df_playersum['Team'].astype(str) + ')'

df_playersum.sort_values(by=["Goal", "xG"], ascending=True, inplace=True)

## Visualising players

### playersum_min_xg or playersum_min_xgot?
sort_by_min_question = input("Do you want to see values by xG or xGOT?")
sort_by_min_value_question = input("What would you as a value?")

if sort_by_min_question in ['xGOT', 'xgot', 'XGOT']:
    min_value_xgot = float(sort_by_min_value_question)
    title_value = min_value_xgot
    sort_by_min = df_playersum[(df_playersum['xGOT'] >= min_value_xgot) | (df_playersum['Goal'] > 0)]
elif sort_by_min_question in ['xG', 'xg', 'XG']:
    min_value_xg = float(sort_by_min_value_question)
    title_value = min_value_xg
    sort_by_min = df_playersum[(df_playersum['xG'] >= min_value_xg) | (df_playersum['Goal'] > 0)]
else:
    print("I am not able to execute that, sorry.")
    

### do the plots
bar_width = 0.25
x_indexes = np.arange(len(sort_by_min['Player_team']))

plt.barh(x_indexes - bar_width, 
         sort_by_min['xG'],
         height=bar_width,
         label='xG',
         color='b')
plt.barh(x_indexes + bar_width,
         sort_by_min['xGOT'],
         height=bar_width,
         label='xGOT',
         color='g')
plt.barh(x_indexes,
         sort_by_min['Goal'],
         height=bar_width,
         label='Goal',
         color='r')

plt.yticks(ticks= x_indexes,
           labels= sort_by_min['Player_team'])

plt.title('Overall xG & xGOT by each player (over '+str(title_value) + ' ' + sort_by_min_question +')')
plt.legend()
plt.grid(axis='x')
plt.text(df_playersum[['xG', 'xGOT', 'Goal']].max().max()*0.7, (len(sort_by_min['Player_team'])-1)/2,
         '@adamjakus99',
         fontsize = 11,
         color='darkblue')

plt.show()



## Seeing players' xG vs xGOT individually
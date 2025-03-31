#%%
import json
import requests
import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

URL = 'https://www.sofascore.com/football/match/napoli-milan/Rdbsoeb#id:12501548'

#%%
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
    time.sleep(15)
    
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
    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Locate the image element using the CSS selector provided
    image_element1 = soup.select_one('div:nth-child(1) > div > a > div > img')
    image_element2 = soup.select_one('div:nth-child(3) > div > a > div > img')
    date_raw = soup.find_all('span', class_='textStyle_body.small c_neutrals.nLv1 lh_1')[0].text
    
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
    
    if date_raw:
        print('Date found')
    else:
        print("Date not found")
    
    driver.quit()

#%%Creating Shotmap DataFrame
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
        'xG': shot['xg'] if 'xg' in shot else None,
        'xGOT': shot['xgot'] if 'xgot' in shot else None}
    data.append(shot_info)

df = pd.DataFrame(data)
df = df.sort_values(by='timeSeconds', ascending=True)
df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)
df.rename(columns={'X Coordinate': 'X', 'Y Coordinate': 'Y'}, inplace=True)
## transforming columns
df['isHome'] = df['Team']
df['Team'] = np.where(df['Team'] == True, home_name, away_name)

if df.xG.all() == None:
    print('No xG found.')
else:
    count_none = 0
    for i in range(len(df.xG)):
        if df.xG[i] == None:
            count_none += 1
    if count_none != 0:
        print(f'{count_none} values not found.')
    else:
        print(f'xG values found for every shot')
        
    
if df.X.all() == None:
    print('No X coordinates found.')
else:
    count_none = 0
    for i in range(len(df.X)):
        if df.X[i] == None:
            count_none += 1
    if count_none != 0:
        print(f'{count_none} X coordinate values not found.')
    else:
        print(f'Coordinate values found for every shot.')

#%%
# calculating cumxg
df['team_cumulative_xG'] = df.groupby('Team')['xG'].cumsum()
df['team_cumulative_xGOT'] = df.groupby('Team')['xGOT'].cumsum()

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
df_home['isHome'] = True
df_away['isHome'] = False

# Combine the DataFrames
df_lineup = pd.concat([df_home, df_away], ignore_index=True)

# Sort the DataFrame by team and substitute status
df_lineup = df_lineup.sort_values(['isHome', 'Substitute', 'Position'])
df_lineup = df_lineup.reset_index(drop=True)

df_lineup['Team'] = np.where(df_lineup.isHome == True, home_name, away_name)

#%% Displaying Cumulative xG and xGOT by time
import matplotlib.font_manager as font_manager
# Colors and styling
mycolor = '#5ECB43'
background_color = '#3c3d3d'
t1_color = '#12a0d7'
t2_color = '#fb090b'
my_font_path = r'C:\Users\Adam\..Data\TSDP\Athletic\Nexa-ExtraLight.ttf'
my_font_props = font_manager.FontProperties(fname=my_font_path)

def plot_cumulative_metric(df, home_name, away_name, metric, title):
    # Extract values for each team
    plot_x_home = df[df['Team'] == home_name]['timeSeconds']
    plot_y_home = df[df['Team'] == home_name][metric]
    plot_x_away = df[df['Team'] == away_name]['timeSeconds']
    plot_y_away = df[df['Team'] == away_name][metric]
    
    # Extend time values for step-like effect
    plot_x_home_extended = pd.concat([plot_x_home, plot_x_home - 1], ignore_index=True).sort_values().reset_index(drop=True)
    plot_x_away_extended = pd.concat([plot_x_away, plot_x_away - 1], ignore_index=True).sort_values().reset_index(drop=True)
    
    plot_x_home_extended.loc[len(plot_x_home_extended)] = df['timeSeconds'].max()
    plot_x_away_extended.loc[len(plot_x_away_extended)] = df['timeSeconds'].max()
    
    # Extend cumulative values for step effect
    plot_y_home_extended = pd.concat([plot_y_home, plot_y_home], ignore_index=True).sort_values().reset_index(drop=True)
    plot_y_away_extended = pd.concat([plot_y_away, plot_y_away], ignore_index=True).sort_values().reset_index(drop=True)
    
    plot_y_home_extended.loc[len(plot_y_home_extended)] = 0
    plot_y_home_extended.sort_values(inplace=True)
    plot_y_away_extended.loc[len(plot_y_away_extended)] = 0
    plot_y_away_extended.sort_values(inplace=True)
    
    # Plot
    plt.figure(facecolor=background_color)
    plt.plot(plot_x_home_extended / 60, plot_y_home_extended, color=t1_color, label=home_name)
    plt.plot(plot_x_away_extended / 60, plot_y_away_extended, color=t2_color, label=away_name)
    
    # Scatter plot for goals
    goal_metric_index = 10 if metric == 'team_cumulative_xG' else 11
    plt.scatter(df[df['Shot Type'] == "goal"]['timeSeconds'] / 60, 
                df[df['Shot Type'] == "goal"].iloc[:, goal_metric_index],
                label='Goals', color=mycolor)
    
    # Formatting
    plt.xticks(ticks=np.arange(0, df['timeSeconds'].max() / 60 + 1, 15))
    plt.xlim(df['timeSeconds'].min() / 60, df['timeSeconds'].max() / 60 + 0.05)
    plt.ylim(df[metric].min(), df[metric].max() * 1.05)
    plt.xlabel('Minutes', color='white')
    plt.ylabel('Expected goals', color='white')
    plt.tick_params(colors='white')
    
    ax = plt.gca()
    ax.set_facecolor(background_color)
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.title(title, color='white')
    plt.grid(axis='x')
    plt.legend()
    plt.text(-5, -0.25 if metric == 'team_cumulative_xG' else -0.25,
             'ADAM JAKUS', color=mycolor, fontsize=13, fontproperties=my_font_props)
    
    plt.show()

# Usage
plot_cumulative_metric(df, home_name, away_name, 'team_cumulative_xGOT', 'Cumulative expected goals after shot on goal')
plot_cumulative_metric(df, home_name, away_name, 'team_cumulative_xG', 'Cumulative expected goals')

#%% xG and xGOT by each player
sort_by_min_in = 'xg' #xg or xgot
sort_by_min_value_in = 0.3 #min value

# Seeing players' shots individually
df_players_shots = df[['Player Name', 'xG', 'xGOT', 'Shot Type']]
df_players_shots['Goal'] = np.where(df_players_shots['Shot Type'] == 'goal', 1, 0)
df_playersum = df_players_shots.groupby('Player Name', as_index=False)[['xG','xGOT', 'Goal']].sum()

## Merging into playersum
df_playersum = pd.merge(df_playersum, df_lineup[['Player Name', 'Team']], how='left', on='Player Name')

## Creating player names with team in ()
df_playersum['Player_team'] = df_playersum['Player Name'].astype(str) + ' (' + df_playersum['Team'].astype(str) + ')'

df_playersum.sort_values(by=["Goal", "xG"], ascending=True, inplace=True)

## Visualising players
if sort_by_min_in in ['xGOT', 'xgot', 'XGOT']:
    min_value_xgot = float(sort_by_min_value_in)
    title_value = min_value_xgot
    sort_by_min = df_playersum[(df_playersum['xGOT'] >= min_value_xgot) | (df_playersum['Goal'] > 0)]
elif sort_by_min_in in ['xG', 'xg', 'XG']:
    min_value_xg = float(sort_by_min_value_in)
    title_value = min_value_xg
    sort_by_min = df_playersum[(df_playersum['xG'] >= min_value_xg) | (df_playersum['Goal'] > 0)]
else:
    print("I am not able to execute that, sorry.")
    

### Create the plots
plt.figure(facecolor=background_color)
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
           labels= sort_by_min['Player Name'], c='white')

plt.tick_params(colors='white')
ax = plt.gca()
ax.set_facecolor(background_color)
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
plt.title(f'Overall xG & xGOT by each player (over {title_value} {sort_by_min_in})',
          c='white')
plt.legend()
plt.grid(axis='x')
plt.text(-0.35, -0.65,
         'ADAM JAKUS', color=mycolor,
         fontsize = 13, fontproperties=my_font_props)

plt.show()

#%%Analyzing goalkeeper performances
## The goalies
df_goalies = df_lineup[(df_lineup.Position == 'G') & (df_lineup.Substitute == False)]
df_goalies['Player_team'] = df_goalies['Player Name'].astype(str) + ' (' + df_goalies['Team'].astype(str) + ')'

## Shots against keepers
df_on_target = df[df["Shot Type"].isin(['goal', 'save'])]

## Getting the xgot and goals values from shots on target
if home_name in df_on_target.Team.unique():
    home_xgot = df_on_target.groupby('Team')['xGOT'].sum()[home_name]
else:
    home_xgot = 0

if 'goal' in df_on_target[df_on_target.Team == home_name]['Shot Type'].unique():
    home_goals_ot = df_on_target.groupby(['Team','Shot Type'])['Shot Type'].count()[home_name]['goal']
else:
    home_goals_ot = 0
        
if away_name in df_on_target.Team.unique():
    away_xgot = df_on_target.groupby('Team')['xGOT'].sum()[away_name]
else:
    away_xgot = 0

if 'goal' in df_on_target[df_on_target.Team == away_name]['Shot Type'].unique():
    away_goals_ot = df_on_target.groupby(['Team','Shot Type'])['Shot Type'].count()[away_name]['goal']
else:
    away_goals_ot = 0

df_goalies['xGOT_against'] = np.where(df_goalies.isHome == True,
                                      away_xgot,
                                      home_xgot)
df_goalies['Goals_against'] = np.where(df_goalies.isHome == True,
                                      away_goals_ot,
                                      home_goals_ot)

## Visualising xGOT and Goals against on barplot
goalies_plot_indexes = np.arange(2)
goalies_bar_width = 0.3

plt.bar(goalies_plot_indexes - goalies_bar_width/2,
        df_goalies['xGOT_against'],
        color='orange',
        label='xGOT against',
        width= goalies_bar_width)
plt.bar(goalies_plot_indexes + goalies_bar_width/2,
        df_goalies.Goals_against,
        color='green',
        label='Goals against',
        width= goalies_bar_width)
plt.xticks(ticks= goalies_plot_indexes,
           labels= df_goalies['Player_team'])

plt.title('xGOT & Goals against keepers')
plt.legend()
plt.grid(axis='y')
plt.text(-0.3,
         df_goalies[['xGOT_against', 'Goals_against']].max().max()*0.8,
         '@adamjakus99',
         fontsize = 10,
         color='darkblue')

plt.show()

#%% Create shotmap
from mplsoccer.pitch import Pitch

pitch = Pitch(line_color = "white")
fig, ax = pitch.draw(figsize=(10, 7))

#Size of the pitch in yards
pitchLengthX = 120
pitchWidthY = 80
#Plot the shots by looping through them
for i,shot in df.iterrows():
    x=shot['X']/110*120
    y=shot['Y']/100*80
    team_name=shot['Team']
    goal= shot['Shot Type']=='goal'
    
    #set circlesize
    circleSize = 2
    #set circlecolor
    color_map = {
        'save': 'cyan',
        'miss': 'magenta',
        'post': 'yellow',
        'goal': 'lime',
        'block': 'orange'
    }

    circleColor = color_map.get(shot['Shot Type'], 'red')
            
    #plot team1
    if (team_name==home_name):
        if goal:
            shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color=circleColor, alpha=1)
            plt.text(x,pitchWidthY-y-3,shot['Player Name'],ha='center',va='center', c="white")
            if shot['xG'] != None:
                plt.text(x,pitchWidthY-y,round(shot['xG'],2),ha='center',va='center', fontsize=9, c=background_color)
        else:
            shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color=circleColor, alpha=0.5)
            
    #plot team2
    else:
        if goal:
            shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color=circleColor, alpha=1)
            plt.text(pitchLengthX-x,y - 3 ,shot['Player Name'],ha='center',va='center', c="white")
            if shot['xG'] != None:
                plt.text(pitchLengthX-x,y,round(shot['xG'],2),ha='center',va='center', fontsize=9, c=background_color)
        else:
            shotCircle=plt.Circle((pitchLengthX-x, y),circleSize,color=circleColor, alpha=0.5)
    ax.add_patch(shotCircle)

fig.set_facecolor(background_color)
ax.set_facecolor(background_color)

plt.title(f"Shotmap", fontsize=22, va='center', c='white')
# Plot goal numbers
home_goals = len(df.loc[(df['Shot Type'] == 'goal') & (df['Team'] == home_name)])
away_goals = len(df.loc[(df['Shot Type'] == 'goal') & (df['Team'] == away_name)])
plt.text(0.25, 0.81, home_goals, fontsize=36, color=t1_color,
         ha='center', va='center', transform=ax.transAxes)
plt.text(0.25, 0.9, home_name, fontsize=36, color=t1_color,
         ha='center', va='center', transform=ax.transAxes)

plt.text(0.75, 0.81, away_goals, fontsize=36, color=t2_color,
         ha='center', va='center', transform=ax.transAxes)
plt.text(0.75, 0.9, away_name, fontsize=36, color=t2_color,
         ha='center', va='center', transform=ax.transAxes)

plt.text(0.5, 0.979, f'{date_raw} | Source: SofaScore', ha='center', va='center', fontsize=13, color='white', transform=ax.transAxes)
plt.text(100, -4,
         'ADAM JAKUS', color=mycolor,
         fontsize=20, fontproperties=my_font_props)

# Create a color legend beneath the pitch
legend_labels = ['Save', 'Miss', 'Post', 'Goal', 'Block']
legend_colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange']

for idx, (label, color) in enumerate(zip(legend_labels, legend_colors)):
    circle = plt.Circle((pitchLengthX / 2 - 30 + idx * 15, pitchWidthY+2.5), 1, color=color, lw=1)
    ax.add_patch(circle)
    plt.text(pitchLengthX / 2 - 25 + idx * 15, pitchWidthY+2.5, label, ha='center', va='center', fontsize=11, c='white')

plt.show()

#%%
with pd.ExcelWriter(r"C:\TwitterSportsDataProject\SofaScore scrapes\SofaScore_Scrape.xlsx") as writer:
    df_playersum.to_excel(writer, sheet_name="Player shots", index=False)
    df_goalies.to_excel(writer, sheet_name="Goalkeepers", index=False)
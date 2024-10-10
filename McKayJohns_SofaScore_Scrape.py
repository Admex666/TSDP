import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# Gathering data from Chromedriver
options = webdriver.ChromeOptions()
options.set_capability('goog:loggingPrefs', {"performance": "ALL", "browser": "ALL"})

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

## Manipulating the chrome window
URL = "https://www.sofascore.com/hu/football/match/atletico-madrid-real-madrid/EgbsLgb#id:12437787"

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
    shotmap_response = driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
    shotmap_data = json.loads(shotmap_response['body'])['shotmap']
    print("Shotmap found")
except Exception as e:
    print("Error retrieving shotmap data:", e)
    

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



# Creating DataFrame
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

## Finalllyyy
plt.show()
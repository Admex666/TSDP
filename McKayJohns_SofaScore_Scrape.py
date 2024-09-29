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
URL = "https://www.sofascore.com/hu/football/match/aston-villa-ipswich-town/HsP#id:12436993"

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
    print(shotmap_data)
except Exception as e:
    print("Error retrieving shotmap data:", e)

driver.quit()


# Creating DataFrame
import pandas as pd
import numpy as np

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
df['Team'] = np.where(df['Team'] == True, 'Home', 'Away')
df['team_cumulative_xG'] = df.groupby('Team')['xG'].cumsum()

print(df)

# Displaying Cumulative xG by time
plot_x_home = df[df['Team'] == "Home"]['timeSeconds']
plot_y_home = df[df['Team'] == "Home"]['team_cumulative_xG']
plot_x_away = df[df['Team'] == "Away"]['timeSeconds']
plot_y_away = df[df['Team'] == "Away"]['team_cumulative_xG']

plt.plot(plot_x_home/60, plot_y_home, color='g', label="Home")
plt.plot(plot_x_away/60, plot_y_away, color='r', label="Away")
plt.scatter(
            df[df['Shot Type'] == "goal"].iloc[:,0]/60, 
            df[df['Shot Type'] == "goal"].iloc[:,9],
            label='Goals')


plt.xlim(df['timeSeconds'].min()/60, df['timeSeconds'].max()/60+0.05)
plt.ylim(df['team_cumulative_xG'].min(), df['team_cumulative_xG'].max()+0.05)

plt.xlabel('Minutes')
plt.ylabel('Expected goals')
plt.title('Cumulative expected goals')
plt.grid(axis='x')

plt.legend()

plt.show()

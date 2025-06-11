#%%
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Selenium beállítások
options = Options()
options.add_argument("--headless")  # Fej nélküli mód
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# WebDriver inicializálása
driver = webdriver.Chrome(options=options)

try:
    url = "https://fbref.com/en/comps/46/schedule/NB-I-Scores-and-Fixtures"
    driver.get(url)

    # Várakozás a táblázat betöltésére
    wait = WebDriverWait(driver, 20)
    table = wait.until(EC.presence_of_element_located((By.ID, "sched_2024-2025_46_1")))

    # Sorok kinyerése
    rows = table.find_elements(By.TAG_NAME, "tr")

    # Adatok tárolása
    data = []
    for row in rows[1:]:  # Első sor a fejléc
        cells = row.find_elements(By.TAG_NAME, "td")
        if not cells:
            continue  # Üres sor kihagyása

        row_data = [cell.text for cell in cells]

        # "Match Report" link kinyerése
        match_report_cell = row.find_elements(By.CLASS_NAME, "center")[0]
        link_element = match_report_cell.find_elements(By.TAG_NAME, "a")
        match_report_url = link_element[0].get_attribute("href") if link_element else None

        row_data.append(match_report_url)
        data.append(row_data)

    # Adatok DataFrame-be helyezése
    columns = [th.text for th in table.find_elements(By.TAG_NAME, "th")][1:]  # Fejléc oszlopnevek
    columns.append("Match_URL")
    
    df = pd.DataFrame(data)
    # Eredmény mentése CSV fájlba
    df.to_csv("TSDP/fbref/nb1_fixtures_2024_2025.csv", index=False)
    print("Adatok sikeresen elmentve a 'nb1_fixtures_2024_2025.csv' fájlba.")

finally:
    driver.quit()


#%%
df_raw = pd.read_csv("TSDP/fbref/nb1_fixtures_2024_2025.csv")
df = df_raw.copy()
df = df.iloc[:,[1, 3, 5, 4, 11]].copy()
df.columns = ['Date', 'Home', 'Away', 'Score', 'Match_URL']
df.dropna(subset='Home', inplace=True)

# Eredmény megjelenítése
df[['HomeGoals', 'AwayGoals']] = df.Score.str.split('–', expand=True)
df[['HomeGoals', 'AwayGoals']] = df[['HomeGoals', 'AwayGoals']].astype(int)
df_matches = df[['Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Match_URL']].copy()

df_matches['HomePoints'] = np.where(df_matches['HomeGoals'] > df_matches['AwayGoals'], 3,
                                    np.where(df_matches['HomeGoals'] == df_matches['AwayGoals'], 1, 0
                                             )
                                    )
df_matches['AwayPoints'] = np.where(df_matches['HomeGoals'] < df_matches['AwayGoals'], 3,
                                    np.where(df_matches['HomeGoals'] == df_matches['AwayGoals'], 1, 0
                                             )
                                    )
df_matches = df_matches.reset_index(drop=True)
print(df_matches.head())

#%% get all tables of league
import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import numpy as np
import time 
import random

path_matches = "TSDP/fbref/nb1_24_25_matches.csv"
df_matches = pd.read_csv(path_matches)

path_goals = "TSDP/fbref/nb1_24_25_goals.csv"
goals_season = pd.read_csv(path_goals)

#%%
for i in df_matches[50:].index:
    if pd.isna(df_matches.loc[i, 'HomePointsDropped']) and pd.isna(df_matches.loc[i, 'AwayPointsDropped']):
        if any(df_matches.loc[i, ['HomeGoals', 'AwayGoals']] == [0, 0]):
            print(f"{i}: One of the teams didn't score, no points dropped.")
            df_matches.loc[i, ['HomePointsDropped', 'AwayPointsDropped']] = 0, 0
        else:
            match_url = df_matches.loc[i, 'Match_URL']
            
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import pandas as pd
            import time
            import random
            
            # Selenium beállítások
            options = Options()
            #options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
    
            # WebDriver inicializálása
            driver = webdriver.Chrome(options=options)
            
            try:
                driver.get(match_url)
            
                # Várakozás az események betöltésére
                wait = WebDriverWait(driver, 20)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'event')))
            
                # Gól események keresése
                goal_events = driver.find_elements(By.CLASS_NAME, 'event')
            
                goals = []
                for event in goal_events:
                    # Ellenőrizzük, hogy van-e gól ikon
                    try:
                        goal_icon = event.find_element(By.CLASS_NAME, 'event_icon.goal')
                    except:
                        continue
            
                    # Időpont és eredmény kinyerése
                    try:
                        time_div = event.find_element(By.TAG_NAME, 'div')
                        goal_time = time_div.text.split('’')[0]
                        score = time_div.find_element(By.TAG_NAME, 'small').text
                    except:
                        continue
            
                    # Játékos nevének kinyerése
                    try:
                        player = event.find_element(By.TAG_NAME, 'a').text
                    except:
                        continue
            
                    # Csapat nevének kinyerése
                    try:
                        team_logo = event.find_element(By.CLASS_NAME, 'teamlogo')
                        team = team_logo.get_attribute('alt').replace(' Club Crest', '')
                    except:
                        continue
            
                    goals.append({
                        'time': goal_time,
                        'player': player,
                        'team': team,
                        'score': score
                    })
            
                # Eredmények megjelenítése
                df_goals = pd.DataFrame(goals)
                
                df_goals[['HomeGoals', 'AwayGoals']] = df_goals.score.str.split(':', expand=True).astype(int)
                df_goals[['match_id']] = i
                print(f'{i}/{len(df_matches)}')
                
                # home points dropped
                had_the_lead_home = any(df_goals['HomeGoals'] > df_goals['AwayGoals'])
                points_home = df_matches.loc[i, 'HomePoints']
                if had_the_lead_home and (points_home != 3):
                    points_dropped_home = 3 - points_home
                else:
                    points_dropped_home = 0
                # away points dropped
                had_the_lead_away = any(df_goals['AwayGoals'] > df_goals['HomeGoals'])
                points_away = df_matches.loc[i, 'AwayPoints']
                if had_the_lead_away and (points_away != 3):
                    points_dropped_away = 3 - points_away
                else:
                    points_dropped_away = 0
                    
                df_matches.loc[i, ['HomePointsDropped', 'AwayPointsDropped']] = points_dropped_home, points_dropped_away
                
                goals_season = pd.concat([goals_season, df_goals], ignore_index=True)
            
            except Exception as e:
                print(f"Hiba történt: {e}")
            
            finally:
                driver.quit()

#%% Points dropped per team
points_dropped_dict = {k:{} for k in df_matches.Home.values}

for team in points_dropped_dict.keys():
    for side in ['Home', 'Away']:
        points_dropped_dict[team][f'{side}PointsDropped'] = int(df_matches[df_matches[side] == team][f'{side}PointsDropped'].sum())
    points_dropped_dict[team]['PointsDropped'] = points_dropped_dict[team]['HomePointsDropped'] + points_dropped_dict[team]['AwayPointsDropped']

df_points_dropped = pd.DataFrame(points_dropped_dict).T
df_points_dropped.reset_index(inplace=True)
df_points_dropped.rename(columns={'index': 'Squad'}, inplace=True)
df_points_dropped = df_points_dropped.sort_values(by='PointsDropped', ascending=False).reset_index(drop=True)

#%% plot
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

background_color = '#3c3d3d'
mycolor = '#5ECB43'

font_path = 'TSDP/Athletic/Arvo-Regular.ttf'
my_font_path = 'TSDP/Athletic/Nexa-ExtraLight.ttf'
font_props = font_manager.FontProperties(fname=font_path)
my_font_props = font_manager.FontProperties(fname=my_font_path)

fig, ax = plt.subplots(figsize=(10,7))
fig.set_facecolor(background_color)
ax.patch.set_facecolor(background_color)

bars = ax.bar(x=df_points_dropped.Squad, 
              height=df_points_dropped.PointsDropped,
              color='red', zorder=2)

ax.bar_label(bars, labels=df_points_dropped.PointsDropped, 
             color='white', fontsize=11, padding=3)

ax.grid(lw=0.5, axis='y', zorder=1)
ax.tick_params(axis='x', rotation=90, color='white', labelcolor='white', labelsize=11)
ax.tick_params(axis='y', color='white', labelcolor='white', labelsize=12)
ax.set_yticks(range(0,21+1,3))
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

from highlight_text import htext
title = '<Points Dropped> from the lead'
htext.fig_text(0.5, 0.963, s=title,
               highlight_textprops=[{'color': 'red', 'weight': 'bold'}], 
               va='center', ha='center',
               fontsize=18, color="white")    


plt.text(5.3, 23.3, '2024-25 | OTP Bank Liga', color='white', fontsize=12, ha='center', va='center')
plt.text(10.4, 24, 'ADAM JAKUS', fontsize=18, color=mycolor, fontproperties=my_font_props, ha='center', va='center')

save_plot = True
file_name = '2025.06.06., points dropped NB1.png'

if save_plot:
    folder = 'C:/Users/Adam/Dropbox/TSDP_output/fbref/2025.06/'
    save_path = folder+file_name
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
else:
    plt.show()

#%% Search specific matches
chosen_team = 'Puskás'
mask_home = (df_matches.Home == chosen_team) & (df_matches['HomePointsDropped'] != 0) 
mask_away = (df_matches.Away == chosen_team) & (df_matches['AwayPointsDropped'] != 0)
df_pd_team = df_matches[mask_home+mask_away]

#%% save
df_matches.to_csv(path_matches, index=False)
goals_season.to_csv(path_goals, index=False)
path_pd = "TSDP/fbref/nb1_24_25_points_dropped.csv"
df_points_dropped.to_csv(path_pd, index=False)
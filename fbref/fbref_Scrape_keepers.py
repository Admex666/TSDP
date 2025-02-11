import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

URL = 'https://fbref.com/en/comps/9/keepers/Premier-League-Stats'
URL_adv = 'https://fbref.com/en/comps/9/keepersadv/Premier-League-Stats'

#%% creating a function for reading HTMLs
def fbref_read_html_upd(URL, table_id):
    # Set up the Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode (no GUI)
    
    chrome_version = '131.0.6778.86'  # Replace this with your Chrome version
    driver = webdriver.Chrome(service=Service(ChromeDriverManager(driver_version=chrome_version).install()))
    
    # Load the page
    driver.get(URL)
    
    try:
        # Wait for the table to be present in the DOM
        table = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, table_id))
        )
        
        # Get the HTML content of the table
        table_html = table.get_attribute('outerHTML')
        
        # Parse the table HTML into a DataFrame
        df = pd.read_html(table_html)[0]
        
        return [df]  # Return as a list to maintain compatibility with the original function
    finally:
        driver.quit()

# creating a function for transforming scraped data into proper dataframes
def fbref_to_dataframe(df):
    if (type(df) == list) & (len(df)==1):
        df = df[0]
    elif type(df) == pd.core.frame.DataFrame:
        df = df.dropna(subset=['Rk'])
    else:
        print('Unknown df type')
        
    if type(df) != pd.core.frame.DataFrame:
        print('Not a pandas dataframe')
    else:
        return df

def fbref_column_joiner(df):
    if type(df.columns.values[0]) != str:
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        return df
    else:
        return df

def fbref_format_column_names(df):
    for ncol in range(len(df.columns)):
        if 'Unnamed' in df.columns[ncol]:
            col_old_name = df.columns[ncol]
            col_new_name = df.columns[ncol].split('_')[3]
            df.rename(columns={col_old_name:col_new_name}, inplace=True)
        else:
            pass

def fbref_scrape(URL, table_id):
    df = fbref_column_joiner(fbref_to_dataframe(fbref_read_html_upd(URL, table_id)))
    return df

def fbref_season_to_next(season):
    season_list = season.split('-')
    season_list[0] = str(int(season_list[0])+1)
    season_list[1] = str(int(season_list[1])+1)
    new_season = '-'.join(season_list)
    return new_season

#%% importing dataframes
df_keepers_adv = fbref_scrape(URL_adv, 'stats_keeper_adv')
fbref_format_column_names(df_keepers_adv)
df_keepers = fbref_scrape(URL,'stats_keeper')
fbref_format_column_names(df_keepers)

#%% Formatting
df_keepers.drop(index=25, inplace=True)
df_keepers_adv.drop(index=25, inplace=True)
df_keepers = df_keepers.reset_index(drop=True)
df_keepers_adv = df_keepers_adv.reset_index(drop=True)

df_keepers_adv['player_short'] = None
df_keepers_adv.Player = df_keepers_adv.Player.str.strip()
for x in range(len(df_keepers_adv.Player)):
    if df_keepers_adv.Player[x].count(" ") == 1:
        keeper_name_list = df_keepers_adv.Player[x].split(" ")
        new_name = []
        new_name.append(keeper_name_list[0][0] + ".")
        new_name.append(keeper_name_list[-1])
        df_keepers_adv['player_short'][x] = " ".join(new_name)
    elif df_keepers_adv.Player[x].count(" ") == 0:
        df_keepers_adv['player_short'][x] = df_keepers_adv['Player'][x]
    elif df_keepers_adv.Player[x].count(" ") == 2:
        keeper_name_list = df_keepers_adv.Player[x].split(" ")
        new_name = []
        new_name.append(keeper_name_list[0][0] + ".")
        new_name.append(keeper_name_list[1][0] + ".")
        new_name.append(keeper_name_list[-1])
        df_keepers_adv['player_short'][x] = " ".join(new_name)
    else:
        df_keepers_adv['player_short'][x] = None
del x, keeper_name_list, new_name

df_toanalyze = pd.merge(df_keepers[['Player',
                                    'Performance_SoTA',
                                    'Performance_Save%']],
                        df_keepers_adv,
                        on='Player')
df_toanalyze = df_toanalyze[df_toanalyze['90s'].astype(float) > 1]
df_toanalyze.iloc[:,2] = df_toanalyze.iloc[:,2].astype(float)
df_toanalyze.iloc[:,9] = df_toanalyze.iloc[:,9].astype(float)
df_toanalyze.iloc[:,9:35] = df_toanalyze.iloc[:,9:35].astype(float)

#%%export to excel
df_toanalyze.to_excel(r'C:\TwitterSportsDataProject\fbref scrapes\fbref_keepers_adv.xlsx')

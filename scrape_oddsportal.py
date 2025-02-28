from selenium import webdriver 
from bs4 import BeautifulSoup
import pandas as pd

ccodes = ['ENG', 'ESP', 'GER', 'ITA', 'FRA']
link_dict = {'ENG': {'country': 'england', 'league':'premier-league'},
             'ESP': {'country': 'spain', 'league':'laliga'},
             'GER': {'country': 'germany', 'league':'bundesliga'},
             'ITA': {'country': 'italy', 'league':'serie-a'},
             'FRA': {'country': 'france', 'league':'ligue-1'},
             }

nr_of_matches = 10

match_links_list_merged = []
for countrycode in ccodes:
    country = link_dict.get(countrycode).get('country')
    league = link_dict.get(countrycode).get('league')
    url = f'https://www.oddsportal.com/football/{country}/{league}/'
    
    driver = webdriver.Chrome()
    driver.get(url)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    match_links = soup.find_all("a", href=True)
    
    # Filtering match links only
    match_links_list = []
    for link in match_links:
        if f"/football/{country}/{league}/" in link['href'] and link['href'].count('/') >= 5 and link['href'].count('-') >= 3:
            match_links_list.append(f"https://www.oddsportal.com{link['href']}")
    
    driver.close()
    
    match_links_list_merged.append(match_links_list)

#%%
df_all = pd.DataFrame()
for league_list in match_links_list_merged:
    for match_url in league_list[:nr_of_matches]:
        df_odds_all = pd.DataFrame()
        for btype in ['1X2', 'OU', 'BTTS']:
    
            bettype_dict = {'1X2': {'url':'#1X2;2',
                                    'html':'border-black-borders flex h-9 border-b border-l border-r text-xs',
                                    'odds_cols': [0, 2, 4, 6],
                                    'odds_colnames': ['bookie', '1', 'X', '2']
                                    },
                            'OU': {'url':'#over-under;2;2.50;0',
                                    'html':'border-black-borders flex h-9 border-b border-l border-r text-xs bg-gray-med_light border-black-borders border-b',
                                    'odds_cols': [0, 3, 5],
                                    'odds_colnames': ['bookie', 'Over', 'Under']
                                    },
                            'BTTS': {'url':'#bts;2',
                                     'html':'border-black-borders flex h-9 border-b border-l border-r text-xs',
                                     'odds_cols': [0, 2, 4],
                                     'odds_colnames': ['bookie', 'Yes', 'No']
                                     }
                            }
            
            btype_url = bettype_dict.get(btype).get('url')
            
            driver = webdriver.Chrome()
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import TimeoutException
            
            driver.get(match_url + btype_url)
            class_name = bettype_dict.get(btype).get('html')
            
            timeout = 7.5
            try:
                element_present = EC.presence_of_element_located((By.CLASS_NAME, class_name))
                WebDriverWait(driver, timeout).until(element_present)
            except TimeoutException:
                print("Timed out waiting for page to load")
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            driver.close()
            
            #%% Get data and manipulate
            e_teams = soup.find(attrs={'data-testid': 'game-participants'})
            team_home = e_teams.find('div', attrs={'data-testid': 'game-host'}).find('p').text
            team_away = e_teams.find('div', attrs={'data-testid': 'game-guest'}).find('p').text
            
            e_date = soup.find(attrs={'data-testid': 'game-time-item'})
            date_raw = e_date.find_all('p')[1].text
            date = pd.to_datetime(date_raw.strip(','), dayfirst=True)
            
            e_rowbet = soup.find_all(class_=class_name)
            
            df_all_text = pd.DataFrame()
            to_drop = []
            for i, elem in enumerate(e_rowbet):
                texts = elem.stripped_strings
                for c, text in enumerate(texts):
                    df_all_text.loc[i, c] = text
                    if (text == 'Odds movement') or (text == '-'):
                        to_drop.append(i)
            
            ncol = len(df_all_text.columns)
            for i in range(len(df_all_text)):
                if df_all_text.iloc[i, 1] != 'claim bonus':
                    df_all_text.iloc[i,2:ncol] =  df_all_text.iloc[i,1:(ncol-1)]
            df_all_text.drop(index=to_drop, inplace=True)
            
            #%% Create clean dataframe
            odds_cols = bettype_dict.get(btype).get('odds_cols')
            odds_colnames = bettype_dict.get(btype).get('odds_colnames') 
            
            globals()[f'df_odds_{btype}'] = df_all_text[odds_cols]
            globals()[f'df_odds_{btype}'].columns = odds_colnames
            globals()[f'df_odds_{btype}'].iloc[:,1:] = globals()[f'df_odds_{btype}'].iloc[:,1:].astype(float)
            #df_odds['house_edge%'] = (1/df_odds['1'] + 1/df_odds['X'] + 1/df_odds['2'] -1)*100
            globals()[f'df_odds_{btype}']['Home'] = team_home
            globals()[f'df_odds_{btype}']['Away'] = team_away
            globals()[f'df_odds_{btype}']['Date'] = date
            
        df_odds_all = pd.merge(df_odds_1X2, df_odds_BTTS,
                               how='outer', on=['bookie', 'Date','Home', 'Away'])
        df_odds_all = pd.merge(df_odds_all, df_odds_OU,
                               how='outer', on=['bookie', 'Date','Home', 'Away'])
            
        df_all = pd.concat([df_all, df_odds_all], ignore_index=True)
        
    df_all = df_all[['Date', 'Home', 'Away', 'bookie', '1', 'X', '2', 'Yes', 'No', 'Over', 'Under']]
    #df_odds_all[['1','X','2']].median()

#%%
path = r'ML_PL_new\modinput_odds.xlsx'
previous_table = pd.read_excel(path)

new_table = pd.concat([previous_table, df_all]).sort_values(by="Date").reset_index(drop=True)
new_table.drop_duplicates(subset=['Date', 'Home', 'Away', 'bookie'], inplace=True)

new_table.to_excel(path, index=False)


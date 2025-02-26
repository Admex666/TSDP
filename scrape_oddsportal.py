from selenium import webdriver 
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.oddsportal.com/football/england/premier-league/"
driver = webdriver.Chrome()
driver.get(url)

soup = BeautifulSoup(driver.page_source, 'html.parser')

match_links = soup.find_all("a", href=True)

# Filtering match links only
match_links_list = []
for link in match_links:
    if "/football/england/premier-league/" in link['href'] and link['href'].count('/') >= 5 and link['href'].count('-') >= 3:
        match_links_list.append(f"https://www.oddsportal.com{link['href']}")

driver.close()

#%%
for match_url in match_links_list[:1]:
    df_odds_all = pd.DataFrame()
    for btype in ['1X2', 'O/U', 'BTTS']:

        bettype_dict = {'1X2': {'url':'#1X2;2',
                                'html':'border-black-borders flex h-9 border-b border-l border-r text-xs',
                                'odds_cols': [0, 2, 4, 6],
                                'odds_colnames': ['bookie', '1', 'X', '2']
                                },
                        'O/U': {'url':'#over-under;2;2.50;0',
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
        
        df_odds = df_all_text[odds_cols]
        df_odds.columns = odds_colnames
        df_odds.iloc[:,1:] = df_odds.iloc[:,1:].astype(float)
        #df_odds['house_edge%'] = (1/df_odds['1'] + 1/df_odds['X'] + 1/df_odds['2'] -1)*100
        df_odds['bet_type'] = btype
        df_odds['Home'] = team_home
        df_odds['Away'] = team_away
        df_odds['Date'] = date_raw
        
        df_odds_all.columns
        df_odds_all = pd.concat([df_odds_all, df_odds]).reset_index(drop=True)
        
        df_odds_all_new = pd.DataFrame({'bookie':df_odds_all.bookie.unique()})
        df_odds_all_new[['Date', 'Home', 'Away']] = date_raw, team_home, team_away
        
        df_odds_all_new2 = pd.merge(df_odds_all_new, df_odds_all.drop(columns='bet_type'),
                                   how='outer', on=['bookie', 'Home', 'Away'])

"""        
        df_odds_all_new = pd.merge(df_odds_all, df_odds, how='outer', on=['Home','Away', 'bookie'])
        
df_odds_all_new = pd.DataFrame({'bookie':df_odds_all.bookie.unique(),
                                'Date':date_raw,
                                'Home',
                                'Away'}
                               )

    columns=['bookie', 'Date', 'Home', 'Away', 'bet_type',
                                    '1', 'X', '2',   'Over', 'Under', 'Yes', 'No'],
                           data=['a']*12)
"""    
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
    for btype in ['1X2', 'O/U', 'BTTS'][:1]:
        bettype_dict = {'1X2': '#1X2;2',
                        'O/U': '#over-under;2',
                        'BTTS': '#bts;2'}
        btype_url = bettype_dict.get(btype)
        
        driver = webdriver.Chrome()
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException
        
        driver.get(match_url + btype_url)
        timeout = 10
        # Várakozás egy adott elem megjelenésére
        try:
            element_present = EC.presence_of_element_located((By.CLASS_NAME, 'border-black-borders flex h-9 border-b border-l border-r text-xs'))
            WebDriverWait(driver, timeout).until(element_present)
        except TimeoutException:
            print("Timed out waiting for page to load")
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        asdf = soup.find_all(class_='border-black-borders flex h-9 border-b border-l border-r text-xs')
        
        teams = soup.find(attrs={'data-testid': 'game-participants'})
        team_home = teams.find('div', attrs={'data-testid': 'game-host'}).find('p').text
        team_away = teams.find('div', attrs={'data-testid': 'game-guest'}).find('p').text
        
        date_elems = soup.find(attrs={'data-testid': 'game-time-item'})
        date_raw = date_elems.find_all('p')[1].text
        
        df_all_text = pd.DataFrame()
        for i, unit in enumerate(asdf):
            texts = unit.stripped_strings
            for c, text in enumerate(texts):
                df_all_text.loc[i, c] = text
        
        for i in range(len(df_all_text)):
            if df_all_text.iloc[i, 1] != 'claim bonus':
                df_all_text.iloc[i,2:9] =  df_all_text.iloc[i,1:8]
        df_all_text.drop(index=17, inplace=True)
        
        
        df_odds = df_all_text[[0, 2, 4, 6]]
        df_odds.columns = ['bookie', '1', 'X', '2']
        df_odds.iloc[:,1:] = df_odds.iloc[:,1:].astype(float)
        df_odds['house_edge%'] = (1/df_odds['1'] + 1/df_odds['X'] + 1/df_odds['2'] -1)*100
        df_odds['bet_type'] = btype
        df_odds['Home'] = team_home
        df_odds['Away'] = team_away
        df_odds['Date'] = date_raw
        
        df_odds_all = pd.concat([df_odds_all, df_odds])
        
        driver.close()
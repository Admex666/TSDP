import requests
from bs4 import BeautifulSoup
import pandas as pd

url_acca = "https://footystats.org/predictions/accumulator-tips"

headers_acc = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response_acc = requests.get(url_acca, headers=headers_acc)

if response_acc.status_code != 200:
    print(f"Failed to retrieve the page. Status code: {response_acc.status_code}")

soup_acc = BeautifulSoup(response_acc.content, 'html.parser')

fixture_items_acc = soup_acc.find_all(class_="row cf acca-date")

if not fixture_items_acc:
    print("No fixture items found. The page structure might have changed.")

data_acc = []

for item in fixture_items_acc:
    acca_elem = item.find(class_="row cf acca-tip whiteBG bbox br4")

    acca = acca_elem.text.strip() if acca_elem else "N/A"
    
    data_acc.append(acca)
del acca

df = pd.DataFrame({'other':[""]*len(data_acc),
                   'matches':[""]*len(data_acc)
                   })

for acca_nr in range(len(data_acc)):
    df['other'][acca_nr] = data_acc[acca_nr].split('Tip & Stats\nOdds\n\n')[0]
    df['matches'][acca_nr] = data_acc[acca_nr].split('Tip & Stats\nOdds\n\n')[1]
del acca_nr, data_acc

for x in range(len(df)):
    if "\nIn this" in df.iloc[x,1]:
        df.iloc[x,1] = df.iloc[x,1].split("\nIn this")[0]
del x

#%% Creating the clean dataframe
df_clean_acca = pd.DataFrame({'acca_name':[],
                         'match_name':[],
                         'bet_type':[],
                         'real_odds':[],
                         'calc_odds':[]})

for x in range(len(df.matches)):
    acca_name = df.other[x].split("Odds")[0].strip()
    df_temp_acca = pd.DataFrame({'acca_name':[acca_name],
                          'match_name':[''],
                          'bet_type':['Accumulator'],
                          'real_odds':[1],
                          'calc_odds':[1]
                          })
    df_clean_acca = pd.concat([df_clean_acca, df_temp_acca], ignore_index=True)
    
    match_list = df.matches[x].split('\n')
    y = 0
    while y < len(match_list):
        if match_list[y] == "":
            del match_list[y]
        y += 1
    z = 0
    while z < len(match_list):
        if match_list[z+1].split('%')[0][-2:] == '00':
            bet_type = match_list[z+1].split('%')[0][:-3]
        else:
            bet_type = match_list[z+1].split('%')[0][:-2]
        calc_odds = float(match_list[z+1].split('%')[1].split('(')[1].split(')')[0])
        
        df_temp_acca = pd.DataFrame({'acca_name':[''],
                                     'match_name':[match_list[z]],
                                     'bet_type':[bet_type],
                                     'real_odds':[float(match_list[z+2])],
                                     'calc_odds': [calc_odds]
                                     })
        df_clean_acca = pd.concat([df_clean_acca, df_temp_acca], ignore_index=True)
        z += 3
del x, y, z, acca_name, match_list, df_temp_acca, df, bet_type, calc_odds

row = 0
while row <= len(df_clean_acca):
    for oddsss in ['real_odds', 'calc_odds']:
        if df_clean_acca.bet_type[row] == "Accumulator":
            row_under = row+1
            while (df_clean_acca.bet_type[row_under] != "Accumulator"):
                df_clean_acca[oddsss][row] = (df_clean_acca[oddsss][row])*(df_clean_acca[oddsss][row_under])
                row_under += 1
        else:
            pass
    row += 1

#%%
for t in range(row+1, row_under):
    df_clean_acca['calc_odds'][row] = (df_clean_acca['calc_odds'][row])*(df_clean_acca['calc_odds'][t])

del row, row_under, oddsss, t

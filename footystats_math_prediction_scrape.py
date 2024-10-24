import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date

def footystats_scrape_math_predictions():
    url = "https://footystats.org/predictions/mathematical"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Print the entire HTML content for debugging
    print(soup.prettify())
    
    fixture_items = soup.find_all(class_="fixture-item")
    
    if not fixture_items:
        print("No fixture items found. The page structure might have changed.")
        return None
    
    data = []
    
    for item in fixture_items:
        match_name_elem = item.find(class_="match-name")
        bet_items_elem = item.find(class_="bet-items")

        match_name = match_name_elem.text.strip() if match_name_elem else "N/A"
        bet_items = bet_items_elem.text.strip() if bet_items_elem else "N/A"
        
        data.append({
            "match_name": match_name,
            "bet_items": bet_items,
        })
    
    df = pd.DataFrame(data)
    
    return df

# Run the scraper and print the result
df = footystats_scrape_math_predictions()
if df is not None and not df.empty:
    print('Dataframe was found.')
else:
    print("Failed to scrape the data or no data was found.")
    
#%% Splitting probability
temp_odds = df.bet_items.str.split('Implied Odds', expand=True)
temp_odds.rename(columns={0:'implied_odds', 1:'real_odds'}, inplace=True)

temp_odds.real_odds = temp_odds.real_odds.str.split('Real Odds')
for x in range(len(temp_odds.real_odds)):
    temp_odds.real_odds[x] = float(temp_odds.real_odds[x][0])
del x

temp_odds.implied_odds = temp_odds.implied_odds.str.strip()
temp_odds['prob'] = ' '
temp_odds['bet_type'] = 0
temp_odds['implied_odds_new'] = 0


for z in range(len(temp_odds.implied_odds)):
    if temp_odds.implied_odds[z][0:3] == '100':
        temp_odds['prob'][z] = float(temp_odds.implied_odds[z][0:3])/100
        temp_odds.implied_odds[z] = temp_odds.implied_odds[z][5:]
    else:
        temp_odds['prob'][z] = float(temp_odds.implied_odds[z][0:2])/100
        temp_odds.implied_odds[z] = temp_odds.implied_odds[z][4:]
del z

#%% Splitting bet_type and implied_odds
for y in range(len(temp_odds.implied_odds)):
    if 1/temp_odds.prob[y] in range(0,10):
        number_cut = -1
    else:
        if temp_odds.implied_odds[y].count('.') == 2:
            if len(temp_odds.implied_odds[y].split('.')[-1]) == 2:
                number_cut = -4
            elif len(temp_odds.implied_odds[y].split('.')[-1]) == 1:
                number_cut = -3
            else:
                temp_odds['bet_type'][y] = 'ERROR'
                temp_odds.implied_odds_new[y] = 'ERROR'
        elif temp_odds.implied_odds[y][-4:].count('.') == 1:
            if len(temp_odds.implied_odds[y].split('.')[-1]) == 2:
                number_cut = -4
            elif len(temp_odds.implied_odds[y].split('.')[-1]) == 1:
                number_cut = -3
            else:
                temp_odds['bet_type'][y] = 'ERROR'
                temp_odds.implied_odds_new[y] = 'ERROR'
    
    temp_odds['bet_type'][y] = temp_odds.implied_odds[y][:number_cut]
    temp_odds.implied_odds_new[y] = float(temp_odds.implied_odds[y][number_cut:])
del y, number_cut

#%% Creating and formatting clean dataframe
df['id'] = range(len(df.match_name))
temp_odds['id'] = range(len(temp_odds.prob))
df_clean = pd.merge(df[['id', 'match_name']], 
                    temp_odds[['id','bet_type','real_odds','implied_odds_new']], 
                    on='id')
df_clean.rename(columns={'implied_odds_new':'calc_odds'}, inplace=True)
df_clean['real_prob'] = 1/df_clean.real_odds
df_clean['calc_prob'] = 1/df_clean.calc_odds

df_clean['bet_unit1'] = 1
df_clean['bet_unit2'] = 1/(df_clean.real_odds-1)
df_clean['bet_unit3'] = 0
for x in range(len(df_clean.real_prob)):
    if df_clean.real_prob[x] > 0.65:
        df_clean['bet_unit3'][x] = 1.75
    elif df_clean.real_prob[x] > 0.4:
        df_clean['bet_unit3'][x] = 1
    elif df_clean.real_prob[x] > 0.2:
        df_clean['bet_unit3'][x] = 0.5
    else:
        df_clean['bet_unit3'][x] = 0.1
del x

df_clean.drop(columns='id', inplace=True)
df_clean['date'] = date.today()
df_clean = df_clean.iloc[:,[9,0,1,2,3,4,5,6,7,8]]
df_clean.columns
#%% Writing data to excel
import pandas as pd
from openpyxl import load_workbook

xlsx = 'C:/TwitterSportsDataProject/footystats/footystats_math_preds.xlsx'

# Load the existing workbook
book = load_workbook(xlsx)

# Get the active sheet
sheet = book.active

# Find the first empty row
###first_empty_row = sheet.max_row
first_empty_row = None
for row in range(1, sheet.max_row+1):
    if all(cell.value is None for cell in sheet[row]):
        first_empty_row = row -1
        break

# If no empty row is found, append at the end
if first_empty_row is None:
    first_empty_row = sheet.max_row

# Append the data
with pd.ExcelWriter(xlsx, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    df_clean.to_excel(writer,
                      sheet_name=sheet.title,
                      index=False,
                      header=False,  # Don't write the header again
                      startrow=first_empty_row)
book.close()    
print("Data successfully appended to the Excel file.")

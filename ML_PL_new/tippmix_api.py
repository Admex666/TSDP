import requests
import json
import pandas as pd
from datetime import datetime, timedelta, time
import numpy as np

url = 'https://api.tippmix.hu/event'
response = requests.get(url)

if response.status_code != 200:
    print(f"Hiba történt: {response.status_code}")


data = response.json()
matches = data['data']

#%%
today_date = datetime.today().date()
span = timedelta(days=10)

# Filter matches
leagues_dict = {'ESP1': 1596, 'ENG1': 1486, 'POR1': 1462, 'ITA1': 1385,
                'FRA1': 951, 'GER1': 517, 'NED1': 1040, 'BEL1': 425}
leagues_dict_inv = {v:k for k, v in leagues_dict.items()}

matches_f = []
for match in matches:
    match_date_iso = datetime.fromisoformat(match['eventDate'])
    match_date = match_date_iso.date()
    filter_ = (
        (match['sportId'] == 1) and
        ( match['competitionId'] in (list(leagues_dict.values())) ) and 
        (match_date <= today_date+span)
        )
    if filter_:
        matches_f.append(match)

# Find odds of filtered matches
odds = []
for match_f in matches_f:
    match_odds = {}
    match_date_iso = datetime.fromisoformat(match_f['eventDate'])
    match_odds['Date'] = datetime.combine(match_date_iso.date(), match_date_iso.time())
    match_odds['Home'] = match_f['eventParticipants'][0]['participantName']
    match_odds['Away'] = match_f['eventParticipants'][1]['participantName']
    match_odds['league_name'] = leagues_dict_inv[match_f['competitionId']]
    
    found_1X2, found_Double, found_o_u = False, False, False
    for market in match_f['markets']:
        # Get 1X2 odds
        if market['marketName'] == '1X2':
            match_odds['H_odds'] = market['outcomes'][0]['fixedOdds']
            match_odds['D_odds'] = market['outcomes'][1]['fixedOdds']
            match_odds['A_odds'] = market['outcomes'][2]['fixedOdds']
            found_1X2 = True
        else:
            pass

        # Get Double chance odds
        if market['marketName'] == 'Kétesély':
            match_odds['Double_1X_odds'] = market['outcomes'][0]['fixedOdds']
            match_odds['Double_12_odds'] = market['outcomes'][1]['fixedOdds']
            match_odds['Double_X2_odds'] = market['outcomes'][2]['fixedOdds']
            found_Double = True
        else:
            pass
        
        # O/U odds
        if market['marketName'] == 'Gólszám 2,5':
            match_odds['Under_odds'] = market['outcomes'][0]['fixedOdds']
            match_odds['Over_odds'] = market['outcomes'][1]['fixedOdds']
            found_o_u = True
        else:
            pass
        
    if found_1X2 or found_Double:
        odds.append(match_odds)

df_odds = pd.DataFrame(odds)

#%% Update excel
path = 'ML_PL_new/modinput_odds_tx.xlsx'
odds_tx_prev = pd.read_excel(path)
odds_tx_combined = pd.concat([odds_tx_prev, df_odds], ignore_index=True)
odds_tx_new = odds_tx_combined.drop_duplicates(subset=['Date', 'Home', 'Away'],
                                               keep='first').reset_index(drop=True)

odds_tx_new.to_excel(path, index=False)
print(f'Appended data with {len(odds_tx_new) - len(odds_tx_prev)} games.')

#%% Make tippmix predictions
# Import fuzzy matching xlsx
path_fuzz = 'ML_PL_new/fuzz_teams.xlsx'
fuzz_teams_og = pd.read_excel(path_fuzz, sheet_name='cities')
# Import previous predictions of model
input_path = 'ML_PL_new/predictions.xlsx'
df_predprobs = pd.read_excel(input_path, sheet_name='pred_probabilities')

today = datetime.today()
#end_date = datetime.combine((today + span), time(23,59))
mask_dates = (df_predprobs.Date > today) #& (df_predprobs.Date <= end_date)
df_actual = df_predprobs[mask_dates]
df_actual = df_actual[['Date', 'HomeTeam', 'AwayTeam', 'H_gNB_prob', 'D_gNB_prob', 'A_gNB_prob']]
df_actual['Double_1X_prob'] = df_actual.H_gNB_prob+df_actual.D_gNB_prob
df_actual['Double_12_prob'] = df_actual.H_gNB_prob+df_actual.A_gNB_prob
df_actual['Double_X2_prob'] = df_actual.D_gNB_prob+df_actual.A_gNB_prob

for i, row in df_odds.iterrows():
    fdcouk_home = fuzz_teams_og[fuzz_teams_og.Team_tippmix == row['Home']].Team_fdcouk.values[0] if row['Home'] in fuzz_teams_og.Team_tippmix.unique() else None
    fdcouk_away = fuzz_teams_og[fuzz_teams_og.Team_tippmix == row['Away']].Team_fdcouk.values[0] if row['Away'] in fuzz_teams_og.Team_tippmix.unique() else None
    df_odds.loc[i, ['HomeTeam', 'AwayTeam']] = fdcouk_home, fdcouk_away

df_actual_odds = pd.merge(df_actual, 
                          df_odds[['HomeTeam', 'AwayTeam', 'H_odds', 'D_odds', 'A_odds', 
                                   'Double_1X_odds', 'Double_12_odds', 'Double_X2_odds']],
                          on=['HomeTeam', 'AwayTeam'],
                          how='inner')

def calc_bet_size_propo(bankroll, odds_bookie, prob_fair):
    # proportional = ( (myprob-prob) * bankroll / (odds-1))
    prob_bookie = 1/odds_bookie
    bet_size = (prob_fair - prob_bookie) * bankroll / (odds_bookie-1) /7 if prob_fair > prob_bookie else 0
    return bet_size

for i in range(len(df_actual_odds)):
    for out in ['H', 'D', 'A', 'Double_1X', 'Double_12', 'Double_X2']:
        probs = df_actual_odds[[col for col in df_actual_odds.columns if (out+'_' in col) and ('prob' in col)]].loc[i].values[0]
        odds = df_actual_odds[[col for col in df_actual_odds.columns if (out+'_' in col) and ('odds' in col)]].loc[i].values[0]
        
        df_actual_odds.loc[i, f'{out}_bet'] = round(calc_bet_size_propo(10000, odds, probs),0)

print(f'{len(df_odds) - len(df_actual_odds)} games not converted from tippmix.')    
missing_rows = []
for i, row in df_odds.iterrows():
    if (row['HomeTeam'] not in df_actual_odds.HomeTeam.unique()) or (row['AwayTeam'] not in df_actual_odds.AwayTeam.unique()):
        missing_rows.append(row)
    
df_missing = pd.DataFrame(missing_rows)
    
#%% Predictions To Excel
output_tx_path = 'ML_PL_new/actual_tx.xlsx'
df_actual_odds.to_excel(output_tx_path, index=False)

#%% Fuzz_teams fuzzy matching
path_fuzz = 'ML_PL_new/fuzz_teams.xlsx'
fuzz_teams_og = pd.read_excel(path_fuzz, sheet_name='cities')
# Only None values to modify
fuzz_teams = fuzz_teams_og[pd.isna(fuzz_teams_og.Team_tippmix)]
fuzz_teams['ratio'] = 0

from fuzzywuzzy import fuzz

for i_fuzz, row_fuzz in fuzz_teams.iterrows():
    best_ratio1, best_ratio2 = 0, 0
    for i_t, row_t in df_odds.iterrows():
        team_tippmix1 = row_t['Home']
        team_tippmix2 = row_t['Away']
        country_code = row_t['league_name'][:3]
        
        if row_fuzz['Country'] == country_code:
            fteam1, fteam2, fteam3 = row_fuzz[['Team_fdcouk', 'Team_fbref', 'Team_odds']]
            
            for teamnr in ['1', '2']:
                for teamcompnr in ['1', '2', '3']:
                    globals()[f'r{teamnr}{teamcompnr}'] = fuzz.ratio(globals()[f'team_tippmix{teamnr}'], globals()[f'fteam{teamcompnr}']) if pd.isna(globals()[f'fteam{teamcompnr}']) == False else 0 
                
                ratios = [r11, r12, r13] if teamnr == '1' else [r21, r22, r23]
                globals()[f'ratio{teamnr}'] = np.array(ratios).mean()
                if globals()[f'ratio{teamnr}'] > globals()[f'best_ratio{teamnr}']:
                    globals()[f'best_ratio{teamnr}'] = globals()[f'ratio{teamnr}']
                    globals()[f'best_index{teamnr}'] = i_fuzz
            
            #print(f"TX: {team_tippmix1}; fdcouk: {fuzz_teams.loc[best_index1, 'Team_fdcouk']}; ratio: {best_ratio1}")
            #print(f"TX: {team_tippmix2}; fdcouk: {fuzz_teams.loc[best_index2, 'Team_fdcouk']}; ratio: {best_ratio2}")
            if best_ratio1 > fuzz_teams.loc[best_index1, 'ratio']:
                final_team1 = team_tippmix1 
                final_ratio1 = best_ratio1
            else:
                final_team1 = fuzz_teams.loc[best_index1, 'Team_tippmix']
                final_ratio1 = fuzz_teams.loc[best_index1, 'ratio']
                
            if best_ratio2 > fuzz_teams.loc[best_index2, 'ratio']:
                final_team2 = team_tippmix2
                final_ratio2 = best_ratio2
            else:
                final_team2 = fuzz_teams.loc[best_index2, 'Team_tippmix']
                final_ratio2 = fuzz_teams.loc[best_index2, 'ratio']
                
            print(final_team1, best_index1)
            fuzz_teams.loc[best_index1, ['Team_tippmix', 'ratio']] = final_team1, final_ratio1
            fuzz_teams.loc[best_index2, ['Team_tippmix', 'ratio']] = final_team2, final_ratio2
            
        else:
            continue
    
#%% Modify fuzz_teams
with pd.ExcelWriter(path_fuzz) as writer:
    fuzz_teams_og.to_excel(writer, sheet_name='cities', index=False)
    
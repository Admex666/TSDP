import pandas as pd
from fbref import fbref_module as fbref
from datetime import datetime

fuzz_teams = pd.read_excel('ML_PL_new/fuzz_teams.xlsx')

preds_path = 'ML_PL_new/predictions.xlsx'
preds_sheets = ['predictions', 'pred_probabilities']
# Get new predictions
df_preds = pd.read_excel(preds_path, sheet_name=preds_sheets[0])
df_predprobs = pd.read_excel(preds_path, sheet_name=preds_sheets[1])

output_path = 'ML_PL_new/paperbets.xlsx'
output_sheets = ['bet_results', 'profits']
# Get evaluated predictions
xlsx_prev1 = pd.read_excel(output_path, sheet_name=output_sheets[0])

df_preds[['HomeGoals', 'AwayGoals']] = None

# Find and delete evaluated from df_preds
for i, row in xlsx_prev1.iterrows():
    evaluated = str(row['FTR_result']) != 'nan'
    mask = (df_preds.HomeTeam == row['HomeTeam']) & (df_preds.AwayTeam == row['AwayTeam']) & (df_preds.Date == row['Date']) & (evaluated == True)
    row_preds = df_preds[mask]
    if row_preds.empty == False:
        df_preds.drop(index=row_preds.index, inplace=True)

# Only check matches that weren't played
today = datetime.today()
df_preds_played = df_preds[df_preds.Date <= today].reset_index(drop=True)

#%%
for countrycode in df_preds_played.Country.unique():
    comp_id, league = fbref.team_dict_get(countrycode)
    url_fixtures = f'https://fbref.com/en/comps/{comp_id}/schedule/{league}-Scores-and-Fixtures'
    df_fixtures = fbref.scrape(url_fixtures, f'sched_2024-2025_{comp_id}_1')
    # Set datetime format
    df_fixtures.drop(index=df_fixtures[df_fixtures.Wk=='Wk'].index, inplace=True)
    df_fixtures.Date = pd.to_datetime(df_fixtures.Date)
    # only played games
    #df_fixtures = df_fixtures[df_fixtures.Date <= datetime.today()]
    
    for i, fdcouk_home in enumerate(df_preds_played.HomeTeam):
        if df_preds_played.loc[i, 'HomeGoals'] == None:
            fbref_home = fuzz_teams.loc[fuzz_teams.Team_fdcouk == fdcouk_home,'Team_fbref'].iloc[0]
            mask = (df_fixtures.Date == df_preds_played.loc[i, 'Date'].normalize()) & (df_fixtures.Home == fbref_home)
            if df_fixtures.loc[mask, 'Score'].empty:
                goals_home, goals_away = None, None
            else:
                score = df_fixtures.loc[mask, 'Score'].iloc[0]
                if str(score) != 'nan':
                    goals_home, goals_away = score.split('–')
                else:
                    goals_home, goals_away = None, None
                
            df_preds_played.loc[i,['HomeGoals', 'AwayGoals']] = goals_home, goals_away
                        
#%% Check results
df_preds_played[['HomeGoals', 'AwayGoals']] = df_preds_played[['HomeGoals', 'AwayGoals']].astype(float) # handle NoneTypes
df_preds_played['FTR_result'] = None
df_preds_played['O/U2.5_result'] = None

for i in range(len(df_preds_played)):
    goals_home = df_preds_played.loc[i, 'HomeGoals']
    goals_away = df_preds_played.loc[i, 'AwayGoals']
    if (str(goals_home) or str(goals_away)) == 'nan':
        pass
    else:
        # Full time result
        ftr = 'H' if goals_home > goals_away else 'D' if goals_home == goals_away else 'A' if goals_home < goals_away else None
        df_preds_played.loc[i, 'FTR_result'] = ftr
        # Over/under result
        o_u = 'Over' if (goals_home+goals_away) > 2.5 else 'Under' if (goals_home+goals_away) < 2.5 else None
        df_preds_played.loc[i, 'O/U2.5_result'] = o_u

# Calculate profits
for btype in ['FTR', 'O/U2.5']:
    for m_short in ['gNB', 'RF', 'DT', 'KNN']:
        df_preds_played[f'{btype}_{m_short}_profit'] = 0
        for i, outcome in enumerate(df_preds_played[f'{btype}_result']):
            outcome_pred = df_preds_played.loc[i,f'{btype}_{m_short}']
            
            if outcome != None:
                win = outcome == outcome_pred
                stake = df_preds_played.loc[i, f'{outcome_pred}_{m_short}_bet']
                if win:
                    odds = df_preds_played.loc[i, f'{outcome_pred}_odds']
                    df_preds_played.loc[i, f'{btype}_{m_short}_profit'] += stake*odds - stake
                else:
                    df_preds_played.loc[i, f'{btype}_{m_short}_profit'] += -stake

#%% To excel
xlsx_prev1 = xlsx_prev1.dropna()
df_preds_played_clean = df_preds_played[df_preds_played.FTR_result.isin(['H', 'D', 'A'])]

# Add new
xlsx_prev1_new = pd.concat([xlsx_prev1,df_preds_played_clean], ignore_index=True)
# Drop duplicates
xlsx_prev1_new = xlsx_prev1_new.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam']).sort_values(by='Date').reset_index(drop=True)

# Summarize profits
profits = pd.concat([xlsx_prev1_new.iloc[:,-8:].sum(),
                     xlsx_prev1_new.iloc[:,-8:].sum() / xlsx_prev1_new.FTR_result.count()],
                    axis=1)

profits.columns = ['Total', 'Average']

with pd.ExcelWriter(output_path) as writer:
    xlsx_prev1_new.to_excel(writer, sheet_name=output_sheets[0], index=False)
    profits.to_excel(writer, sheet_name=output_sheets[1], index=False)
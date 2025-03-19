import pandas as pd
from fbref import fbref_module as fbref
from datetime import datetime
import matplotlib.pyplot as plt

fuzz_teams = pd.read_excel('ML_PL_new/fuzz_teams.xlsx')

preds_path = 'ML_PL_new/predictions.xlsx'
preds_sheets = ['predictions', 'pred_probabilities']
# Get new predictions
df_preds = pd.read_excel(preds_path, sheet_name=preds_sheets[0])
df_predprobs = pd.read_excel(preds_path, sheet_name=preds_sheets[1])

output_path = 'ML_PL_new/paperbets.xlsx'
output_sheets = ['bet_results_pred', 'bet_results_predprob', 'profits']
# Get evaluated predictions
xlsx_preds = pd.read_excel(output_path, sheet_name=output_sheets[0])
xlsx_predprobs = pd.read_excel(output_path, sheet_name=output_sheets[1])

df_preds[['HomeGoals', 'AwayGoals']] = None

# Find and delete evaluated from df_preds
for dfname in ['_preds', '_predprobs']:
    for i, row in globals()[f'xlsx{dfname}'].iterrows():
        evaluated = str(row['FTR_result']) != 'nan'
        mask = (globals()[f'df{dfname}'].HomeTeam == row['HomeTeam']) & (globals()[f'df{dfname}'].AwayTeam == row['AwayTeam']) & (globals()[f'df{dfname}'].Date == row['Date']) & (evaluated == True)
        row_preds = globals()[f'df{dfname}'][mask]
        if row_preds.empty == False:
            globals()[f'df{dfname}'].drop(index=row_preds.index, inplace=True)

# Only check matches that weren't played
today = datetime.today()
df_preds_played = df_preds[df_preds.Date <= today].reset_index(drop=True)
df_predprobs_played = df_predprobs[df_predprobs.Date <= today].reset_index(drop=True)

#%% scrape fbref for scores
for countrycode in df_preds_played.Country.unique():
    comp_id, league = fbref.team_dict_get(countrycode)
    url_fixtures = f'https://fbref.com/en/comps/{comp_id}/schedule/{league}-Scores-and-Fixtures'
    df_fixtures = fbref.scrape(url_fixtures, f'sched_2024-2025_{comp_id}_1')
    # Set datetime format
    df_fixtures.drop(index=df_fixtures[df_fixtures.Wk=='Wk'].index, inplace=True)
    df_fixtures.Date = pd.to_datetime(df_fixtures.Date)
        
    for i, fdcouk_home in enumerate(df_preds_played.HomeTeam):
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

                        
df_predprobs_played = pd.merge(df_predprobs_played, 
                               df_preds_played[['Date', 'HomeTeam', 'FTR_result', 'O/U2.5_result']], 
                               on=['Date', 'HomeTeam'], how='left')

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

for btype in ['FTR', 'O/U2.5']:
    for m_short in ['gNB', 'RF', 'DT', 'KNN']:
        df_predprobs_played[f'{btype}_{m_short}_profit'] = 0
        for i, outcome in enumerate(df_predprobs_played[f'{btype}_result']):
            if str(outcome) == 'nan' or outcome == None:
                pass
            else:
                outcome_list = ['H', 'D', 'A'] if btype == 'FTR' else ['Over', 'Under']
                bets_won = df_predprobs_played.loc[i, f'{outcome}_{m_short}_bet']
                odds = df_predprobs_played.loc[i, f'{outcome}_odds']
                
                outcome_list.remove(outcome)
                bets_lost = df_predprobs_played.loc[i, [f'{col}_{m_short}_bet' for col in outcome_list]].sum()
                
                
                df_predprobs_played.loc[i, f'{btype}_{m_short}_profit'] = (bets_won*odds - bets_won) - bets_lost

#%% Prepare for export
# Import previous
xlsx_preds = xlsx_preds.dropna()
xlsx_predprobs = xlsx_predprobs.dropna()

df_preds_played_clean = df_preds_played[df_preds_played.FTR_result.isin(['H', 'D', 'A'])]
df_predprobs_played_clean = df_predprobs_played[df_predprobs_played.FTR_result.isin(['H', 'D', 'A'])]

# Add new
xlsx_preds_new = pd.concat([xlsx_preds,df_preds_played_clean], ignore_index=True)
xlsx_predprobs_new = pd.concat([xlsx_predprobs,df_predprobs_played_clean], ignore_index=True)
# Drop duplicates
xlsx_preds_new = xlsx_preds_new.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam']).sort_values(by='Date').reset_index(drop=True)
xlsx_predprobs_new = xlsx_predprobs_new.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam']).sort_values(by='Date').reset_index(drop=True)

# Summarize profits
profit_cols = [col for col in xlsx_preds_new.columns if ('_profit' in col) & ('cumsum' not in col)]
profits = pd.concat([xlsx_preds_new.loc[:,profit_cols].sum(),
                     xlsx_preds_new.loc[:,profit_cols].sum() / xlsx_preds_new.FTR_result.count(),
                     xlsx_predprobs_new.loc[:,profit_cols].sum(),
                     xlsx_predprobs_new.loc[:,profit_cols].sum() / xlsx_predprobs_new.FTR_result.count(),
                     ],
                    axis=1)
profits.columns = ['Total_preds', 'Average_preds', 'Total_predprobs', 'Average_predprobs']

# Change of profit
print(f'\n Changes of profit ({len(df_preds_played_clean)} games):\n \n', df_preds_played_clean.loc[:,profit_cols].sum())
print(f'\n Total profits ({len(xlsx_preds_new)} games):\n \n', xlsx_preds_new.loc[:,profit_cols].sum())
print(f'\n Changes of profit ({len(df_predprobs_played_clean)} games):\n \n', df_predprobs_played_clean.loc[:,profit_cols].sum())
print(f'\n Total profits ({len(xlsx_predprobs_new)} games):\n \n', xlsx_predprobs_new.loc[:,profit_cols].sum())

for dfname in ['preds', 'predprobs']:
    # Cumulated profits
    for btype in ['FTR', 'O/U2.5']:
        for m_short in ['gNB', 'RF', 'DT', 'KNN']:
            globals()[f'xlsx_{dfname}_new'][f'{btype}_{m_short}_profit_cumsum'] = globals()[f'xlsx_{dfname}_new'][f'{btype}_{m_short}_profit'].cumsum()

    # Plot the cumulated profits
    plt.figure(figsize=(10,6))
    plt.plot(globals()[f'xlsx_{dfname}_new'].iloc[:,-8:], label=globals()[f'xlsx_{dfname}_new'].columns[-8:])
    plt.title(f'Profits by matches, {dfname}', fontsize=18) 
    plt.xlabel('Matches')
    plt.ylabel('Profit')
    plt.legend()
    plt.axhline(y=0, linestyle='--', color='grey')
    plt.show()

#%% To excel
with pd.ExcelWriter(output_path) as writer:
    xlsx_preds_new.to_excel(writer, sheet_name=output_sheets[0], index=False)
    xlsx_predprobs_new.to_excel(writer, sheet_name=output_sheets[1], index=False)
    profits.to_excel(writer, sheet_name=output_sheets[2], index=True)
import pandas as pd
from fbref import fbref_module as fbref
from datetime import datetime

fuzz_teams = pd.read_excel('ML_PL_new/fuzz_teams.xlsx')

preds_path = r'C:\Users\Ádám\Dropbox\TSDP_output\PL ML model\ML_predictions.xlsx'
preds_sheets = ['predictions', 'pred_probabilities']

df_preds = pd.read_excel(preds_path, sheet_name=preds_sheets[0])
df_predprobs = pd.read_excel(preds_path, sheet_name=preds_sheets[1])

df_preds[['HomeGoals', 'AwayGoals']] = None

#%%
for countrycode in ['ENG', 'ESP', 'GER', 'ITA', 'FRA']:
    comp_id, league = fbref.team_dict_get(countrycode)
    url_fixtures = f'https://fbref.com/en/comps/{comp_id}/schedule/{league}-Scores-and-Fixtures'
    df_fixtures = fbref.scrape(url_fixtures, f'sched_2024-2025_{comp_id}_1')
    # Set datetime format
    df_fixtures.drop(index=df_fixtures[df_fixtures.Wk=='Wk'].index, inplace=True)
    df_fixtures.Date = pd.to_datetime(df_fixtures.Date)
    # only played games
    #df_fixtures = df_fixtures[df_fixtures.Date <= datetime.today()]
    
    for i, fdcouk_home in enumerate(df_preds.HomeTeam):
        if df_preds.loc[i, 'HomeGoals'] == None:
            fbref_home = fuzz_teams.loc[fuzz_teams.Team_fdcouk == fdcouk_home,'Team_fbref'].iloc[0]
            mask = (df_fixtures.Date == df_preds.loc[i, 'Date'].normalize()) & (df_fixtures.Home == fbref_home)
            score = df_fixtures.loc[mask, 'Score'].iloc[0]
            
            if str(score) != 'nan':
                goals_home, goals_away = score.split('–')
            else:
                goals_home, goals_away = None, None
                
            df_preds.loc[i,['HomeGoals', 'AwayGoals']] = goals_home, goals_away
                        
#%% Check results
df_preds[['HomeGoals', 'AwayGoals']] = df_preds[['HomeGoals', 'AwayGoals']].astype(float) # handle NoneTypes
df_preds['FTR_result'] = None
df_preds['O/U2.5_result'] = None

for i in range(len(df_preds)):
    goals_home = df_preds.loc[i, 'HomeGoals']
    goals_away = df_preds.loc[i, 'AwayGoals']
    if (str(goals_home) or str(goals_away)) == 'nan':
        pass
    else:
        # Full time result
        ftr = 'H' if goals_home > goals_away else 'D' if goals_home == goals_away else 'A' if goals_home < goals_away else None
        df_preds.loc[i, 'FTR_result'] = ftr
        # Over/under result
        o_u = 'Over' if (goals_home+goals_away) > 2.5 else 'Under' if (goals_home+goals_away) < 2.5 else None
        df_preds.loc[i, 'O/U2.5_result'] = o_u

# Calculate profits
for btype in ['FTR', 'O/U2.5']:
    for m_short in ['gNB', 'RF', 'DT', 'KNN']:
        df_preds[f'{btype}_{m_short}_profit'] = 0
        for i, outcome in enumerate(df_preds[f'{btype}_result']):
            outcome_pred = df_preds.loc[i,f'{btype}_{m_short}']
            
            if outcome != None:
                win = outcome == outcome_pred
                stake = df_preds.loc[i, f'{outcome_pred}_{m_short}_bet']
                if win:
                    odds = df_preds.loc[i, f'{outcome_pred}_odds']
                    df_preds.loc[i, f'{btype}_{m_short}_profit'] += stake*odds - stake
                else:
                    df_preds.loc[i, f'{btype}_{m_short}_profit'] += -stake

# Summarize profits
profits = pd.concat([df_preds.iloc[:,-8:].sum(),
                     df_preds.iloc[:,-8:].sum() / df_preds.FTR_result.count()],
                    axis=1)

profits.columns = ['Total', 'Average']

#%% To excel
with pd.ExcelWriter('ML_PL_new/paperbets.xlsx') as writer:
    df_preds.to_excel(writer, sheet_name='bet_results')
    profits.to_excel(writer, sheet_name='profits')
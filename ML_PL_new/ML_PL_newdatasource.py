import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Loading data
path = r'C:\Users\Ádám\Downloads\E0.csv'
df = pd.read_csv(path)
# Only needed columns
needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
df = df[needed_cols]
# create BTTS and O2,5 labels
df['BTTS'] = np.where((df.FTHG!=0)&(df.FTAG!=0),'Yes','No')
df['O/U2.5'] = np.where(df.FTHG+df.FTAG>2.5,'Over','Under')

#%% Transforming data
def df_to_model_input(df):  
    # Writing it out
    df.rename(columns={
        'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals',
        'HS': 'HomeShots', 'AS': 'AwayShots',
        'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget',
        'HC': 'HomeCorners', 'AC': 'AwayCorners',
        'HY': 'HomeYellows', 'AY': 'AwayYellows',
        'FTR': 'FullTimeResult'
    }, inplace=True)
    
    # Date form
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.sort_values('Date', inplace=True)
    
    # Stats list for rolling averages
    stats = ['Goals', 'Shots', 'ShotsOnTarget', 'Corners', 'Yellows']
    home_stats = ['Home' + stat for stat in stats]
    away_stats = ['Away' + stat for stat in stats]
    
    # Rolling average calculations for each team
    rolling_features = []
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        team_home = df[df['HomeTeam'] == team].copy()
        team_away = df[df['AwayTeam'] == team].copy()
    
        for stat, home_stat, away_stat in zip(stats, home_stats, away_stats):
            team_home[stat + '_Home_RAvg'] = team_home[home_stat].shift(1).rolling(window=3).mean()
            team_away[stat + '_Away_RAvg'] = team_away[away_stat].shift(1).rolling(window=3).mean()
    
        rolling_features.append(pd.concat([team_home, team_away]))
    
    # Summing rolling averages
    rolling_df = pd.concat(rolling_features).sort_values(['Date', 'HomeTeam'])
    
    final_df = df[['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 'BTTS', 'O/U2.5']].copy()
    
    # Home team rolling average merge
    home_rolling = rolling_df[['Date', 'HomeTeam'] + [stat + '_Home_RAvg' for stat in stats]]
    home_rolling.columns = ['Date', 'HomeTeam'] + [stat + '_Home_RAvg' for stat in stats]
    final_df = final_df.merge(home_rolling, on=['Date', 'HomeTeam'])
    
    # Away team rolling average merge
    away_rolling = rolling_df[['Date', 'AwayTeam'] + [stat + '_Away_RAvg' for stat in stats]]
    away_rolling.columns = ['Date', 'AwayTeam'] + [stat + '_Away_RAvg' for stat in stats]
    final_df = final_df.merge(away_rolling, on=['Date', 'AwayTeam'], how='left')
    
    # Final data structure
    model_input = final_df.copy().dropna().reset_index(drop=True)
    return model_input

model_input = df_to_model_input(df)

#%% Build ML model
df_accs = pd.DataFrame()

for btype in ['FullTimeResult', 'BTTS', 'O/U2.5']:
    x = model_input.iloc[:,6:]
    y = model_input.loc[:, btype]
    for m in ['GaussianNB', 'RandomForestClassifier', 'DecisionTreeClassifier']:
        acc_list = []
        for n in range(170, 220):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=n)
            
            GaussianNB()
            model = globals()[f"{m}"]()
            model.fit(x_train, y_train)
            
    # Evaluation
            y_pred = model.predict(x_test)
            accuracy= accuracy_score(y_pred, y_test)
            acc_list.append(accuracy)
        df_accs[f'{m}_{btype}'] = acc_list
    
accs_describe = df_accs.describe().iloc[1:, :]
df_accs.boxplot(rot=90)
    
#%% Getting the fresh data for predictions
import pandas as pd
path_pred = path
df_pred = pd.read_csv(path_pred)

#%% Transform
import numpy as np

needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
df_pred = df_pred[needed_cols]
# create BTTS and O2,5 labels
df_pred['BTTS'] = np.where((df_pred.FTHG!=0)&(df_pred.FTAG!=0),'Yes','No')
df_pred['O/U2.5'] = np.where(df_pred.FTHG+df_pred.FTAG>2.5,'Over','Under')

teams = np.sort(df_pred.HomeTeam.unique())

# Add current matches
select_date = '03/01/2025'

#%% Create next round's pairings
df_current = df_pred.copy()
df_current = df_current.iloc[:10]
df_current.Date = select_date
df_current.iloc[:,3:-3] = 0
df_current[['FTR','BTTS','O/U2.5']] = 'none'

# select matches from teams list
df_current.loc[0, ['HomeTeam', 'AwayTeam']] = [teams[15], teams[4]]
df_current.loc[1, ['HomeTeam', 'AwayTeam']] = [teams[17], teams[3]]
df_current.loc[2, ['HomeTeam', 'AwayTeam']] = [teams[16], teams[9]]
df_current.loc[3, ['HomeTeam', 'AwayTeam']] = [teams[11], teams[8]]
df_current.loc[4, ['HomeTeam', 'AwayTeam']] = [teams[13], teams[6]]
df_current.loc[5, ['HomeTeam', 'AwayTeam']] = [teams[14], teams[7]]
df_current.loc[6, ['HomeTeam', 'AwayTeam']] = [teams[10], teams[2]]
df_current.loc[7, ['HomeTeam', 'AwayTeam']] = [teams[12], teams[1]]
df_current.loc[8, ['HomeTeam', 'AwayTeam']] = [teams[19], teams[0]]
df_current.loc[9, ['HomeTeam', 'AwayTeam']] = [teams[18], teams[5]]

df_all = pd.concat([df_pred,df_current], ignore_index=True)

model_input_pred = df_to_model_input(df_all)
model_input_pred = model_input_pred.iloc[-10:,:].reset_index(drop=True)

#%% To excel

model_input_pred.to_excel(r'C:\Users\Adam\.Data files\ML_PL_new\model_input_predict.xlsx', index=False)

import pandas as pd
import numpy as np

def df_to_model_input(df):  
    # Writing it out
    df.rename(columns={
        'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals',
        'HS': 'HomeShots', 'AS': 'AwayShots',
        'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget',
        'HC': 'HomeCorners', 'AC': 'AwayCorners',
        'HY': 'HomeYellows', 'AY': 'AwayYellows',
    }, inplace=True)
    
    # Date form
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.sort_values('Date', inplace=True)
    
    # Calculate points
    df['HomePoints'] = np.where(df.FTR == 'H', 3, np.where(df.FTR=='D', 1, 0))
    df['AwayPoints'] = np.where(df.FTR == 'A', 3, np.where(df.FTR=='D', 1, 0))
    
    # Calculate goal differences
    df['HomeGD'] = df.HomeGoals - df.AwayGoals
    df['AwayGD'] = df.AwayGoals - df.HomeGoals
    
    # Stats list for rolling averages
    stats = ['Goals', 'Shots', 'ShotsOnTarget', 'Corners', 'Yellows', 'Points', 'GD']
    home_stats = ['Home' + stat for stat in stats]
    away_stats = ['Away' + stat for stat in stats]
    
    # Rolling average calculations for each team
    rolling_features = []
    ppm_season_dict = {}
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        team_home = df[df['HomeTeam'] == team].copy()
        team_away = df[df['AwayTeam'] == team].copy()
        
        ppm_season_dict[team] = {'Home':team_home.HomePoints.mean(),
                                 'Away':team_away.AwayPoints.mean()}
    
        for stat, home_stat, away_stat in zip(stats, home_stats, away_stats):
            team_home[stat + '_Home_RAvg'] = team_home[home_stat].shift(1).rolling(window=3).mean()
            team_away[stat + '_Away_RAvg'] = team_away[away_stat].shift(1).rolling(window=3).mean()
    
        rolling_features.append(pd.concat([team_home, team_away]))
    
    # Summing rolling averages
    rolling_df = pd.concat(rolling_features).sort_values(['Date', 'HomeTeam'])
    
    final_df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'BTTS', 'O/U2.5']].copy()
    
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
    
    # Adding new variables
    model_input[['HomePPM', 'AwayPPM']] = None, None 
    for i, row in model_input.iterrows():
        home_ppm = ppm_season_dict.get(row['HomeTeam']).get('Home')
        away_ppm = ppm_season_dict.get(row['AwayTeam']).get('Away')
        model_input.loc[i, ['HomePPM', 'AwayPPM']] = home_ppm, away_ppm
    
    return model_input
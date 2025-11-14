import pandas as pd
import numpy as np
from ML_PL_new.historical_weather_api import get_coordinates, get_hourly_weather

def df_to_model_input(df, weather=False):  
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
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        team_home = df[df['HomeTeam'] == team].copy()
        team_away = df[df['AwayTeam'] == team].copy()
    
        for stat, home_stat, away_stat in zip(stats, home_stats, away_stats):
            team_home[stat + '_Home_RAvg'] = team_home[home_stat].shift(1).rolling(window=3).mean()
            team_away[stat + '_Away_RAvg'] = team_away[away_stat].shift(1).rolling(window=3).mean()
        
        # calculate and add ppm variables
        team_home[['HomePPM']] = float(0)
        for i, row in team_home.iterrows():
            date = row['Date']
            mask = team_home.Date<date
            ppm = team_home.loc[mask, 'HomePoints'].mean()
            # add ppm value
            team_home.loc[i, 'HomePPM'] = ppm
            
        team_away[['AwayPPM']] = float(0)
        for i, row in team_away.iterrows():
            date = row['Date']
            mask = team_away.Date<date
            ppm = team_away.loc[mask, 'AwayPoints'].mean()
            # add ppm value
            team_away.loc[i, 'AwayPPM'] = ppm
        
        rolling_features.append(pd.concat([team_home, team_away]))
    
    # Summing rolling averages
    rolling_df = pd.concat(rolling_features).sort_values(['Date', 'HomeTeam'])
    
    final_df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'BTTS', 'O/U2.5']].copy()
    
    # Home team rolling average merge
    home_rolling = rolling_df[['Date', 'HomeTeam'] + [stat + '_Home_RAvg' for stat in stats]  + ['HomePPM']]
    home_rolling.columns = ['Date', 'HomeTeam'] + [stat + '_Home_RAvg' for stat in stats]  + ['HomePPM']
    final_df = final_df.merge(home_rolling, on=['Date', 'HomeTeam'])
    
    # Away team rolling average merge
    away_rolling = rolling_df[['Date', 'AwayTeam'] + [stat + '_Away_RAvg' for stat in stats] + ['AwayPPM']]
    away_rolling.columns = ['Date', 'AwayTeam'] + [stat + '_Away_RAvg' for stat in stats]  + ['AwayPPM']
    final_df = final_df.merge(away_rolling, on=['Date', 'AwayTeam'], how='left')
    
    # Final data structure
    model_input = final_df.copy().dropna().reset_index(drop=True)
    
    # Adding weather data
    """
    fuzz_teams_all = pd.read_excel('ML_PL_new/fuzz_teams.xlsx', sheet_name='cities')
    ## Create a data table for each city
    date_min, date_max = model_input.Date.min().strftime('%Y-%m-%d'), model_input.Date.max().strftime('%Y-%m-%d')
    weathers_historic = pd.DataFrame()
    for team in model_input.HomeTeam.unique():
        mask = fuzz_teams_all.Team_fdcouk == team
        [city, postal_code] = fuzz_teams_all.loc[mask, ['City', 'Postal_Code']].iloc[0]
        lati, longi = get_coordinates(city, postal_code)
        weather_team = get_hourly_weather(lati, longi, date_min, date_max)
        weather_team['Team'] = team
        
        weathers_historic = pd.concat([weathers_historic, weather_team])
    weathers_historic.to_excel('ML_PL_new/weathers_ENG_22_24.xlsx', index=False)
    """
    if weather:
        weathers_historic = pd.read_excel('ML_PL_new/weathers_ENG_22_24.xlsx')
        model_input[['temp_celsius', 'wind_speed_kmh', 'weathercode']] = float(0)
        for i, row in model_input.iterrows():
            date = row['Date']
            #date = date.strftime('%Y-%m-%d')
            team = row['HomeTeam']
            mask = (weathers_historic.Team == team) & (date == weathers_historic['date'].dt.normalize())
            date_weather = weathers_historic[mask]
            temp_mean, wind_mean = date_weather[['temp_celsius', 'wind_speed_kmh']].mean()
            weather_mode = date_weather['weathercode'].mode().iloc[0]
            
            model_input.loc[i, ['temp_celsius', 'wind_speed_kmh', 'weathercode']] = temp_mean, wind_mean, weather_mode
        
    return model_input
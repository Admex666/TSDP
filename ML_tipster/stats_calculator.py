# stats_calculator.py
import pandas as pd
from datetime import datetime

def get_last5_stats(team_name, df_league):
    """Utolsó 5 mérkőzés statisztikáinak számítása"""
    home_matches = df_league[df_league['HomeTeam'] == team_name].copy()
    away_matches = df_league[df_league['AwayTeam'] == team_name].copy()
    
    # Statisztikák számítása
    home_matches['Points'] = home_matches.apply(
        lambda x: 3 if x['FTHG'] > x['FTAG'] else (1 if x['FTHG'] == x['FTAG'] else 0), axis=1
    )
    home_matches['Goals_For'] = home_matches['FTHG']
    home_matches['Goals_Against'] = home_matches['FTAG']
    
    away_matches['Points'] = away_matches.apply(
        lambda x: 3 if x['FTAG'] > x['FTHG'] else (1 if x['FTAG'] == x['FTHG'] else 0), axis=1
    )
    away_matches['Goals_For'] = away_matches['FTAG']
    away_matches['Goals_Against'] = away_matches['FTHG']
    
    all_matches = pd.concat([home_matches, away_matches])
    
    if len(all_matches) == 0:
        return 0, 0, 0, 999
    
    all_matches['Date'] = pd.to_datetime(all_matches['Date'], dayfirst=True)
    all_matches = all_matches.sort_values('Date', ascending=False)
    last_5 = all_matches.head(5)
    
    avg_points = last_5['Points'].mean()
    avg_goals_for = last_5['Goals_For'].mean()
    avg_goals_against = last_5['Goals_Against'].mean()
    
    latest_date = last_5['Date'].iloc[0]
    days_since = (datetime.now() - latest_date).days
    
    return round(avg_points, 1), round(avg_goals_for, 1), round(avg_goals_against, 1), days_since
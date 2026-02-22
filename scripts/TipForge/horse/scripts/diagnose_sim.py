import pandas as pd

df = pd.read_csv('data/simulation_results.csv')

# Bookmaker margin (overround)
df['implied_prob'] = 1 / df['market_odds']
race_overround = df.groupby('race_id')['implied_prob'].sum()
print(f'Avg bookmaker overround: {race_overround.mean():.3f} ({(race_overround.mean()-1)*100:.1f}% margin)')

# Model calibration check
wins = df.groupby(pd.cut(df['prob_norm'], bins=5))['win'].mean()
print(f'\nCalibration check (prob_norm bin -> actual win rate):')
print(wins.to_string())

# Average market odds
print(f'\nAvg market odds of WINNERS : {df[df["win"]==1]["market_odds"].mean():.2f}')
print(f'Avg market odds of LOSERS  : {df[df["win"]==0]["market_odds"].mean():.2f}')
print(f'\nValue bets hit rate: {df[df["is_value"]]["win"].mean():.3f}')
print(f'Avg market odds on value bets: {df[df["is_value"]]["market_odds"].mean():.2f}')

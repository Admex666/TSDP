#%% 1. Fetch data
import pandas as pd
from fbref import fbref_module as fbref
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

#df_raw = fbref.get_all_player_data('ESP', year='2024-2025')
#df_raw.to_excel('laligaplayers24-25.xlsx', index=False)
df_raw = pd.read_excel('laligaplayers24-25.xlsx')

#%%
cols_basic = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']
metrics = {'offense': ['Per 90 Minutes_Gls' , 'Per 90 Minutes_xG' ,'Standard_G/Sh',
                       'Standard_G/SoT', 'Standard_SoT/90'],
           'creativity': ['Per 90 Minutes_Ast', 'Per 90 Minutes_xAG', 'KP', 'SCA_SCA90',
                          'GCA_GCA90'],
           'progression': ['Progression_PrgP', 'Progression_PrgC', 'Progression_PrgR',
                           'Carries_PrgDist', 'Carries_PrgC'],
           'activity': ['Touches_Att 3rd', 'Carries_Carries', 'Receiving_Rec',
                        'Receiving_PrgR'],
           'defense': ['Tackles_Tkl', 'Performance_Int', 'Blocks_Blocks', 
                       'Performance_Recov'],
           'errors': ['Carries_Dis', 'Carries_Mis', 'Err']
           }
all_metrics = [item for sublist in metrics.values() for item in sublist]

df = df_raw[cols_basic+all_metrics]
df.fillna(0, inplace=True)
df = df[df['90s'] != 0]
df[['Age', 'Born']] = df[['Age', 'Born']].astype(float)
# make it per 90
df90 = df.copy()
for metric in all_metrics:
    if ('90' in metric) or ('%' in metric):
        pass
    else:
        df90[metric] = df90[metric] / df90['90s']

standard_positions = ['GK', 'DF', 'MF', 'FW']

# A Pos oszlopban van pl. "DF, MF" vagy "FW"
# Szétbontjuk vessző mentén, majd explode a lista
df90['Pos_list'] = df90['Pos'].str.split(',\s*')

# Explode -> minden pozícióhoz külön sor
df_pos_exploded = df90.explode('Pos_list').rename(columns={'Pos_list': 'Pos_single'})

# Csak azokat a sorokat tartjuk meg, amiknek Pos_single eleme a standard_positions
df_pos_exploded = df_pos_exploded[df_pos_exploded['Pos_single'].isin(standard_positions)]

# aggregate to teams
def weighted_team_avg(df, group_col, value_cols, weight_col='90s'):
    return df.groupby(group_col).apply(
        lambda g: np.average(g[value_cols], weights=g[weight_col], axis=0)
    ).apply(pd.Series)

def weighted_team_pos_avg(df, group_cols, value_cols, weight_col='90s'):
    return df.groupby(group_cols).apply(
        lambda g: np.average(g[value_cols], weights=g[weight_col], axis=0)
    ).apply(pd.Series)

team_aggregates = weighted_team_avg(df90, 'Squad', all_metrics)
team_aggregates.columns = all_metrics  # újra megnevezzük
team_aggregates = team_aggregates.reset_index()

team_pos_aggregates = weighted_team_pos_avg(
    df_pos_exploded,
    group_cols=['Squad', 'Pos_single'],
    value_cols=all_metrics
)

team_pos_aggregates.columns = all_metrics
team_pos_aggregates = team_pos_aggregates.reset_index()

# Ha szeretnéd, csinálhatunk belőle pivot táblát is a könnyebb áttekinthetőséghez:
team_pos_pivot = team_pos_aggregates.pivot(index='Squad', columns='Pos_single')

#%% Player relative profiles (%, Z-score)
## Merge with team aggregates
df90 = df90.merge(team_aggregates, on='Squad', suffixes=('', '_team_avg'))

## Percentages for team metrics
for metric in all_metrics:
    df90[f'{metric}_pct_of_team'] = (df90[metric] / df90[f'{metric}_team_avg']) * 100

## std for team metrics
team_std = df90.groupby('Squad')[all_metrics].std().reset_index()
team_std = team_std.rename(columns={col: col + '_team_std' for col in all_metrics})
df90 = df90.merge(team_std, on='Squad')

## Z-score for players
for metric in all_metrics:
    std_col = f'{metric}_team_std'
    team_avg_col = f'{metric}_team_avg'
    z_col = f'{metric}_zscore'
    
    df90[z_col] = (df90[metric] - df90[team_avg_col]) / df90[std_col].replace(0, np.nan)
    df90[z_col] = df90[z_col].fillna(0)

## Grouped Z-score averages
for k in metrics.keys():
    metrics_of_group = metrics[k].copy()
    for n in range(len(metrics_of_group)):
        metrics_of_group[n] = metrics_of_group[n]+'_zscore'
    df90[f'{k}_median_zscore'] = df90[metrics_of_group].median(axis=1)
    print(f'\n Top players in {k}:')
    print(df90.loc[df90['90s'] >= 5,['Player', 'Squad', 'Pos', '90s', f'{k}_median_zscore']].sort_values(f'{k}_median_zscore', ascending=False).head(10))
    
#%% Distance to team
from sklearn.preprocessing import StandardScaler

# 1. Válassz csapatot
team_name = 'Barcelona'
pos = 'FW'

# 2. Csapat játékosainak adatai
team_players = df90[(df90['Squad'] == team_name) & (df90['Pos'] == pos)]

# 3. Kiválasztott mutatók, pl. z-score-ok alapján hasonlóság
# (vagy bármely más mutató, de z-score normalizált adatok előnyösek)
selected_metrics = [col for col in df90.columns if col.endswith('_zscore')]

# 4. Csapat profilja = csapat játékosainak átlaga ezekből a mutatókból
team_profile = team_players[selected_metrics].mean(axis=0).values.reshape(1, -1)

# 5. Különítsd el a nem-csapat játékosokat
other_players = df90[df90['Squad'] != team_name]

# 6. Más játékosok mutatói
other_profiles = other_players[selected_metrics].values

# 7. Számold ki a távolságokat a csapat profil és a többi játékos között
distances = euclidean_distances(team_profile, other_profiles).flatten()

# 8. Add hozzá a távolságokat a dataframe-hez
other_players = other_players.copy()
other_players['distance_to_team'] = distances

# 9. Rendezés növekvő távolság szerint - legkisebb a legjobb hasonlóság
closest_players = other_players.sort_values('distance_to_team').head(10)

# 10. Kiírás vagy további feldolgozás
print(f"Top 10 hasonló játékos a csapathoz ({team_name}) képest, akik nem a csapat játékosai:")
print(closest_players[['Player', 'Squad', 'distance_to_team']])

#%% Distance to player
player_name = 'Tamás Nikitscher'
selected_player = df90[df90['Player'] ==player_name]
zscore_cols = [col for col in df90.columns if col.endswith('_zscore')]
candidates = df90[df90['Player'] != player_name]

target_vector = selected_player[zscore_cols].values
candidate_vectors = candidates[zscore_cols].values

distances = euclidean_distances(candidate_vectors, target_vector).flatten()
candidates['zscore_distance'] = distances

similar_players = candidates.sort_values('zscore_distance')
print(f'Closest players to {player_name}:')
print(similar_players[['Player', 'Squad', 'Pos', 'zscore_distance']].head(10))

#%% Radar chart
import matplotlib.pyplot as plt
import numpy as np

def radar_chart(player_name, df, metrics):
    # Kiválasztjuk a játékos sorát
    player = df[df['Player'] == player_name].iloc[0]

    # Az értékeket készítjük elő (itt z-score értékeket használjuk)
    values = [player[f'{m}_zscore'] for m in metrics]
    values += values[:1]  # kör bezárása

    # Szöveg címkék
    labels = metrics
    labels += labels[:1]

    # Szög számítása
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    # Plot létrehozása
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_title(f'Radar chart for {player_name}')
    ax.grid(True)
    plt.show()

#metrics_for_radar = ['Per 90 Minutes_Gls', 'Per 90 Minutes_Ast', 'KP', 'SCA_SCA90', 'Tackles_Tkl']
metrics_for_radar = [f'{k}_median' for k in metrics.keys()]
radar_chart(player_name, df90, metrics_for_radar)


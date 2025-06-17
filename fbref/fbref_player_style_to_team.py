#%% import
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df_raw = pd.read_excel('TSDP/laligaplayers24-25.xlsx')

df_wys_raw = pd.read_excel('TSDP/Wys_NB_I_players_stats_20250604.xlsx')
df_wys_raw['90s'] = df_wys_raw['Minutes played'] / 90
df_wys_raw.rename(columns={'Team within selected timeframe': 'Squad', 
                           'Birth country': 'Nation', 'Position': 'Pos_exact'},
                  inplace=True)
"""
all_positions = []
for i in df_wys_raw.index:
    poses = df_wys_raw.loc[i, 'Pos_exact']
    pos_list = poses.split(',')
    for pos in pos_list:
        pos = pos.strip()
        if pos not in all_positions:
            all_positions.append(pos)
"""

for i in df_wys_raw.index:
    pos_first = df_wys_raw.loc[i, 'Pos_exact'].split(',')[0]
    if pos_first == 'GK':
        df_wys_raw.loc[i, 'Pos'] = 'GK'
    elif pos_first in ['CB', 'LB', 'RB', 'LWB', 'RWB', 'LCB', 'RCB']:
        df_wys_raw.loc[i, 'Pos'] = 'DF'
    elif pos_first in ['RCMF', 'LCMF', 'CMF', 'RDMF', 'LDMF', 'DMF', 'AMF', 'LAMF', 'RAMF']:
        df_wys_raw.loc[i, 'Pos'] = 'MF'
    elif pos_first in ['CF', 'LWF', 'RWF', 'LW', 'RW']:
        df_wys_raw.loc[i, 'Pos'] = 'FW'
    else:
        df_wys_raw.loc[i, 'Pos'] = None

df_wys_raw.fillna(0, inplace=True)

#%% Settings, functions
#----------------------#
#       SETTINGS       #
#----------------------#

settings = {
    # For fbref
    'fbref': {
        'df': df_raw,
        'metrics': {
                    'offense': ['Per 90 Minutes_Gls', 'Per 90 Minutes_xG', 'Standard_G/Sh', 'Standard_G/SoT', 'Standard_SoT/90'],
                    'creativity': ['Per 90 Minutes_Ast', 'Per 90 Minutes_xAG', 'KP', 'SCA_SCA90', 'GCA_GCA90'],
                    'progression': ['Progression_PrgP', 'Progression_PrgC', 'Progression_PrgR', 'Carries_PrgDist', 'Carries_PrgC'],
                    'activity': ['Touches_Att 3rd', 'Carries_Carries', 'Receiving_Rec', 'Receiving_PrgR'],
                    'defense': ['Tackles_Tkl', 'Performance_Int', 'Blocks_Blocks', 'Performance_Recov'],
                    'errors': ['Carries_Dis', 'Carries_Mis', 'Err']
                    },
        'cols_basic': ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']
        },
    # For wyscout
    'wyscout':{
        'df': df_wys_raw,
        'metrics': {
                    'offense': ['Goals', 'xG', 'Successful attacking actions per 90', 'Goal conversion, %', 'Shots on target, %'],
                    'creativity': ['Assists', 'xA', 'Key passes per 90', 'Second assists per 90', 'Accurate smart passes, %'],
                    'progression': ['Progressive passes per 90', 'Dribbles per 90', 'Progressive runs per 90', 'Deep completions per 90'],
                    'activity': ['Passes to final third per 90', 'Touches in box per 90', 'Received passes per 90'],
                    'defense': ['Successful defensive actions per 90', 'Interceptions per 90', 'Shots blocked per 90']
                    },
        'cols_basic': ['Player', 'Nation', 'Pos', 'Squad', 'Age', '90s']
        }
    }

platform = 'wyscout'

df = settings[platform]['df']
cols_basic = settings[platform]['cols_basic']
metrics = settings[platform]['metrics']
numeric_cols = [col for col in df.drop(columns=cols_basic) if (' played' not in col) & 
                (col != 'Market value') & ('Playing Time_' not in col) & 
                (col != 'Height') & (col != 'Weight') &
                ((df[col].dtype == 'float64') | (df[col].dtype == 'int64'))]

all_metrics = [item for sublist in metrics.values() for item in sublist]

#----------------------#
#      FUNCTIONS       #
#----------------------#

def create_df90(df, numeric_cols):
    df = df.fillna(0)
    df90 = df.copy()
    for col in numeric_cols:
        if ('90' in col) or ('%' in col) or ('G/S' in col) or ('Playing Time' in col):
            continue
        else:
            df90[col] = df90[col] / df90['90s']
    return df90

def weighted_team_avg(df, group_col, value_cols, weight_col='90s'):
    team_aggregates = df.groupby(group_col).apply(lambda g: np.average(g[value_cols], weights=g[weight_col], axis=0)).apply(pd.Series).reset_index()
    original_first_column = [team_aggregates.columns[0]]
    team_aggregates.columns = original_first_column + value_cols
    return team_aggregates
    
def weighted_team_pos_avg(df, group_cols, value_cols, weight_col='90s'):
    team_pos_aggregates = df.groupby(group_cols).apply(lambda g: np.average(g[value_cols], weights=g[weight_col], axis=0)).apply(pd.Series).reset_index()
    original_first_columns = list(team_pos_aggregates.columns[0:2])
    team_pos_aggregates.columns = original_first_columns + value_cols
    return team_pos_aggregates

def add_team_and_positional_comparisons(df, all_metrics, matches_at_least=5):
    df = df.copy()

    # --- Csapatátlag és Z-score számítás ---
    team_avg = df.groupby('Squad')[all_metrics].mean().reset_index()
    team_std = df.groupby('Squad')[all_metrics].std().reset_index()

    team_avg = team_avg.rename(columns={col: col + '_team_avg' for col in all_metrics})
    team_std = team_std.rename(columns={col: col + '_team_std' for col in all_metrics})

    df = df.merge(team_avg, on='Squad')
    df = df.merge(team_std, on='Squad')

    df = df[df['90s'] > matches_at_least]

    for col in all_metrics:
        avg_col = f'{col}_team_avg'
        std_col = f'{col}_team_std'
        z_col = f'{col}_team_zscore'
        df[z_col] = (df[col] - df[avg_col]) / df[std_col].replace(0, np.nan)
        df[z_col] = df[z_col].fillna(0)

    # --- Pozíciós Z-score számítás ---
    pos_means = df.groupby('Pos')[all_metrics].mean()
    pos_stds = df.groupby('Pos')[all_metrics].std()

    for col in all_metrics:
        df[f'{col}_pos_comp_zscore'] = df.apply(
            lambda x: (x[col] - pos_means.loc[x['Pos'], col]) / pos_stds.loc[x['Pos'], col]
            if x['Pos'] in pos_means.index and pos_stds.loc[x['Pos'], col] != 0 else 0,
            axis=1
        )

    # --- Kategóriák medián z-score-ja ---
    for ztype in ['_team_zscore', '_pos_comp_zscore']:
        for k, metric_list in metrics.items():    
            z_cols = [f"{m}{ztype}" for m in metric_list]
            df[f'{k}_median{ztype}'] = df[z_cols].median(axis=1)

    return df

def find_similar_players_to_team(df, team_name, pos, selected_metrics, metric='euclidean', n_neighbors=20, matches_at_least=5):
    df_filtered = df[(df['90s'] > matches_at_least)].reset_index(drop=True).copy()

    if pos:
        df_filtered = df_filtered[df_filtered['Pos'] == pos].reset_index(drop=True)

    team_players = df_filtered[df_filtered['Squad'] == team_name]
    if team_players.empty:
        raise ValueError(f"Nincs játékos a(z) {team_name} csapatból a kiválasztott pozícióban: {pos}")

    team_profile = team_players[selected_metrics].mean(axis=0).values.reshape(1, -1)
    other_players = df_filtered[df_filtered['Squad'] != team_name].copy()
    other_profiles = other_players[selected_metrics].values

    if metric == 'euclidean':
        distances = euclidean_distances(team_profile, other_profiles).flatten()
        other_players['distance_to_team'] = distances
        return other_players.sort_values('distance_to_team').head(n_neighbors)

    elif metric == 'cosine':
        similarities = cosine_similarity(team_profile, other_profiles).flatten()
        other_players['cosine_similarity_to_team'] = similarities
        return other_players.sort_values('cosine_similarity_to_team', ascending=False).head(n_neighbors)

    elif metric == 'knn':
        # Skálázás
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_filtered[selected_metrics])

        # PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(scaled_data)

        # KNN
        knn = NearestNeighbors(n_neighbors=n_neighbors+1)  # +1 saját maga miatt
        knn.fit(X_pca)

        # Team profil PCA térbe vetítve (átlagolt érték)
        team_index = df_filtered[df_filtered['Squad'] == team_name].index
        team_pca_vector = X_pca[team_index].mean(axis=0).reshape(1, -1)

        distances, indices = knn.kneighbors(team_pca_vector)

        # Nearest neighbors lekérdezése
        similar_players = df_filtered.iloc[indices[0]].copy()
        similar_players = similar_players[similar_players['Squad'] != team_name]  # saját csapat kizárása
        return similar_players.head(n_neighbors)

    else:
        raise ValueError(f"Ismeretlen metrika: '{metric}'. Használható: 'euclidean', 'cosine', 'knn'")


def find_similar_players_to_player(df, player_id, selected_metrics, metric='euclidean', matches_at_least=5, pos=None):
    df_filtered = df[df['90s'] > matches_at_least].reset_index(drop=False).copy()

    if player_id not in df_filtered['index']:
        raise ValueError(f"Nincs ilyen játékos (id: {player_id}) a szűrt adatokban.")

    selected_player = df_filtered.loc[df_filtered['index'] == player_id]
    candidates = df_filtered.loc[df_filtered['index'] != player_id]

    target_vector = selected_player[selected_metrics].values
    candidate_vectors = candidates[selected_metrics].values

    if metric == 'euclidean':
        distances = euclidean_distances(candidate_vectors, target_vector).flatten()
        candidates['distance_to_player'] = distances
        return candidates.sort_values('distance_to_player').head(20)

    elif metric == 'cosine':
        similarities = cosine_similarity(target_vector, candidate_vectors).flatten()
        candidates['cosine_similarity_to_player'] = similarities
        return candidates.sort_values('cosine_similarity_to_player', ascending=False).head(20)

    elif metric == 'knn':
        # Skálázás
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_filtered[selected_metrics])

        # PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(scaled_data)

        # KNN
        knn = NearestNeighbors(n_neighbors=21)
        knn.fit(X_pca)

        player_idx = df_filtered.index.get_loc(player_id)
        player_vector = X_pca[player_idx].reshape(1, -1)

        distances, indices = knn.kneighbors(player_vector)

        similar_players = df_filtered.iloc[indices[0]].copy()
        similar_players = similar_players.drop(index=player_id, errors='ignore')
        return similar_players.head(20)

    else:
        raise ValueError(f"Ismeretlen metrika: '{metric}'. Használható: 'euclidean', 'cosine', 'knn'")


def assign_player_archetypes(df, all_metrics, n_clusters=4):
    df = df.copy()
    df['player_type'] = None

    for pos in df['Pos'].unique():
        pos_df = df[df['Pos'] == pos]
        if len(pos_df) < n_clusters:
            continue  # Ne próbáljunk klasztert képezni túl kevés játékosból
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pos_df[all_metrics])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(scaled)

        df.loc[pos_df.index, 'player_type'] = clusters

    return df

#----------------------#
#      PIPELINE        #
#----------------------#

def analyze_player_similarity(df, cols_basic, numeric_cols, all_metrics, matches_at_least=5, team_name=None, pos=None, player_id=None):
    df90 = create_df90(df, numeric_cols)

    team_aggregates = weighted_team_avg(df90, 'Squad', all_metrics)
    df90 = add_team_and_positional_comparisons(df90, all_metrics, matches_at_least=matches_at_least)
    df90 = assign_player_archetypes(df90, all_metrics, n_clusters=4)
    
    results = {}
    results['df90'] = df90.copy()
    
    for size_type in ['player', 'team']:
        for metric in ['euclidean', 'cosine', 'knn']:
            for ztype in ['_team', '_pos_comp']:
                zscore_cols = [col for col in df90.columns if col.endswith(f'{ztype}_zscore')]
                
                if pd.notna(player_id) and (size_type == 'player'):
                    results[f'similar_to_player_{metric}{ztype}'] = find_similar_players_to_player(
                        df90, player_id, zscore_cols, metric=metric, matches_at_least=matches_at_least, pos=pos
                        )
                
                if team_name and pos and (size_type == 'team'):
                    results[f'similar_to_team_{metric}{ztype}'] = find_similar_players_to_team(
                        df90,  team_name, pos, zscore_cols, metric=metric
                        )

    return results

#%% Execution
matches_at_least = 5
team_name = 'Paksi FC'
pos = 'DF'
player_id = 47


results = analyze_player_similarity(
    df,
    cols_basic, numeric_cols, all_metrics,
    matches_at_least,
    team_name,
    pos,
    player_id   
)

df90 = results['df90'].copy()

# Eredmények megtekintés
for k in results.keys():
    print('\n\n')
    if 'player' not in k:
        if 'euclidean' in k:
            print(k)
            print(results[k][['Player', 'Squad', 'Pos', 'distance_to_team']])
        
        elif 'cosine' in k:
            print(k)
            print(results[k][['Player', 'Squad', 'Pos', 'cosine_similarity_to_team']])
        
        elif 'knn' in k:
            print(k)
            print(results[k][['Player', 'Squad', 'Pos']])
    else:
        print(f"\nLeginkább hasonló játékosok {df90.loc[player_id, 'Player']}-hoz:")
        print(k)
        print(results[k][['Player', 'Squad', 'Pos']])
        
#%% Viz functions
import matplotlib.pyplot as plt
import numpy as np
from mlpsoccer.radar_module import radar

# Egyszerű matplotlib radar chart egy játékosról
def plot_radar_single_player(df, player_id, metrics):
    player = df.loc[player_id, :]
    player_name = player['Player']
    values = [player[f'{m}_zscore'] for m in metrics]
    values += values[:1]
    
    labels = metrics + [metrics[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_title(f'Radar chart for {player_name}', fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# mplsoccer radar chart: két játékos összehasonlítása
def plot_mplsoccer_player_comparison(df, player1_id, player2_id, params, reversed_list=None, pretty_names=None, league_name='', matches_at_least=5, only_pos=False, use_zscore=True,
                                     save_folder=False, save_name=False, save=False):
    reversed_list = reversed_list if reversed_list else []
    pretty_names = pretty_names if pretty_names else params

    # értékek és stat range meghatározása
    if use_zscore:
        param_cols = [f'{p}_zscore' for p in params]
    else:
        param_cols = params
    
    if only_pos:
        low = df[df.Pos == only_pos][param_cols].min().tolist()
        high = df[df.Pos == only_pos][param_cols].max().tolist()
        pos = only_pos
    else:
        low = df[param_cols].min().tolist()
        high = df[param_cols].max().tolist()
        pos = 'All position'
    
    p1 = df.loc[player1_id, :]
    p2 = df.loc[player2_id, :]
    
    nr1_name, nr1_squad = p1[['Player', 'Squad']]
    nr2_name, nr2_squad = p2[['Player', 'Squad']]

    nr1_values = p1[param_cols].tolist()
    nr2_values = p2[param_cols].tolist()

    radar(pretty_names, low, high, reversed_list, 
          nr1_name, nr1_squad, nr1_values, 
          nr2_name, nr2_squad, nr2_values, 
          league_name, pos, matches_at_least,
          save_folder=save_folder, save_name=save_name, save=save)


# Csapat radar chart pozícióval vagy anélkül
def plot_mplsoccer_team_comparison(df, team1, team2, params, position=None, reversed_list=None, pretty_names=None, league_name='',
                                   save_folder=False, save_name=False, save=False):
    reversed_list = reversed_list if reversed_list else []
    pretty_names = pretty_names if pretty_names else params
    
    if position:
        mask1 = (df['Squad'] == team1) & (df['Pos'].str.contains(position))
        mask2 = (df['Squad'] == team2) & (df['Pos'].str.contains(position))
    else:
        mask1 = (df['Squad'] == team1)
        mask2 = (df['Squad'] == team2)

    values1 = df.loc[mask1, params].values[0].tolist()
    values2 = df.loc[mask2, params].values[0].tolist()

    low = df[params].min().tolist()
    high = df[params].max().tolist()
    
    nr1_squad,  nr2_squad = '', ''

    radar(pretty_names, low, high, reversed_list, 
          team1, nr1_squad, values1, 
          team2, nr2_squad, values2, 
          league_name, position, matches_at_least,
          save_folder=save_folder, save_name=save_name, save=save)

#%% Viz execution
plot_radar_single_player(df90, 47,
                         ['offense_median_pos_comp', 'creativity_median_pos_comp', 
                          'progression_median_pos_comp', 'activity_median_pos_comp',
                          'defense_median_pos_comp'])

#%% Két játékos összehasonlítása zscore alapján
plot_mplsoccer_player_comparison(df90, 47, 226,
                                 params=(metrics['activity']+metrics['defense']+metrics['progression']),
                                 reversed_list=['errors'],
                                 league_name='OTP Bank liga',
                                 only_pos='DF',
                                 use_zscore=False,
                                 save_folder=r'C:\Users\Adam\Dropbox\TSDP_output\fbref\2025.06', 
                                 save_name='2025.06.17., Gartenmann-Katona L.', 
                                 save=True)

#%% Csapat összehasonlítás egy pozícióban
team_pos_aggregates = weighted_team_pos_avg(df90, ['Squad', 'Pos'], all_metrics)

plot_mplsoccer_team_comparison(team_pos_aggregates, 'Paksi FC', 'Debrecen',
                               params=metrics['activity'] + metrics['progression'] + metrics['defense'],
                               position='DF',
                               league_name='OTP Bank liga',
                               save_folder=False, save_name=False, save=False)

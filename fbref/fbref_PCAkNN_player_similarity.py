#%% 1. Fetch data
import pandas as pd
from TSDP.fbref import fbref_module as fbref
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

matches_at_least = 5
#df_raw = fbref.get_all_player_data('ESP', year='2024-2025')
#df_raw.to_excel('laligaplayers24-25.xlsx', index=False)
df_raw = pd.read_excel('TSDP/laligaplayers24-25.xlsx')

df_wys_raw = pd.read_excel('TSDP/Wys_NB_I_players_stats_20250604.xlsx')
df_wys_raw['90s'] = df_wys_raw['Minutes played'] / 90
df_wys_raw.rename(columns={'Team within selected timeframe': 'Squad', 
                           'Birth country': 'Nation', 'Position': 'Pos'},
                  inplace=True)

#%% Function
def PCAplayer(df90, cols_basic, numeric_cols, cols_chosen, player_index, matches_at_least):
    df90 = df90[df90['90s'] > matches_at_least].reset_index().copy()
    # --- 4. Skálázás (StandardScaler) ---
    X = df90[cols_chosen]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 5. PCA ---
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # --- 6. kNN (K legközelebbi szomszéd) ---
    knn = NearestNeighbors(n_neighbors=11)  # 1 saját magát is tartalmazza
    knn.fit(X_pca)
    
    # --- 7. Megadott játékos keresése ---
    player_index_new = df90[df90['index']==player_index].index[0]
    player_name = df90.at[player_index_new, 'Player']
    distances, indices = knn.kneighbors([X_pca[player_index_new]])
    
    # --- 8. Hasonló játékosok kiírása ---
    print(f"Hasonló játékosok {player_name}-hez:")
    for idx in indices[0][1:]:  # Első saját maga
        print('')
        print(f"{df90.loc[idx, 'Player']} ({df90.loc[idx, 'Squad']}, {df90.loc[idx, 'Pos']}), ID: {df90.loc[idx, 'index']}")
    
    # --- 9. (opcionális) 2D-s vizualizáció ---
    pca_2d = PCA(n_components=2).fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_2d[:, 0], pca_2d[:, 1], alpha=0.6)
    plt.scatter(pca_2d[player_index, 0], pca_2d[player_index, 1], color='red', label=player_name)
    for idx in indices[0][1:]:
        plt.scatter(pca_2d[idx, 0], pca_2d[idx, 1], color='green')
    plt.legend()
    plt.title("PCA tér - Hasonló játékosok vizualizálása")
    plt.show()

def create_df90(df_raw):
    df = df_raw.copy()
    # --- 3. Hiányzó adatok kezelése ---
    df = df.fillna(0)
    
    df90 = df.reset_index(drop=True).copy()
    for col in numeric_cols:
        if ('90' in col) or ('%' in col) or ('G/S' in col):
            pass
        else:
            df90[col] = df90[col] / df90['90s']
    
    return df90

#%% Execution: Data for fbref df
cols_basic = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s'] 
numeric_cols = [col for col in df_raw.drop(columns=cols_basic) if ('Playing Time' not in col) & (df_raw[col].dtype == 'float64')]
cols_chosen = ['Tackles_Tkl', 'Performance_Int', 'Blocks_Blocks', 
               'Performance_Recov', 'Progression_PrgP', 'Progression_PrgC', 
               'Progression_PrgR', 'Carries_PrgDist', 'Carries_PrgC']
player_index = 101
df90 = create_df90(df_raw)

#%% Execution: Data for wyscout df
cols_basic = ['Player', 'Nation', 'Pos', 'Squad', 'Age', '90s'] 
numeric_cols = [col for col in df_wys_raw.drop(columns=cols_basic) if (' played' not in col) & 
                ('Market val' not in col) & (col != 'Height') & (col != 'Weight') &
                ((df_wys_raw[col].dtype == 'float64') | (df_wys_raw[col].dtype == 'int64'))]

cols_chosen = numeric_cols.copy()
cols_chosen = ['Successful defensive actions per 90', 'Defensive duels per 90',
               'Successful attacking actions per 90', 'xG per 90', 'Crosses per 90',
               'Shots per 90', 'Dribbles per 90', 'Offensive duels per 90', 
               'Progressive runs per 90', 'Fouls suffered per 90',
               'xA per 90', 'Smart passes per 90', 'Key passes per 90',
               ]
player_index = 2
df90 = create_df90(df_wys_raw)

PCAplayer(df90, cols_basic, numeric_cols, cols_chosen, player_index, matches_at_least)

#%% 
from mlpsoccer.radar_module import radar
params = cols_chosen.copy()
low = []
high = []
for param in params:
    low.append(df90.loc[:,param].min())
    high.append(df90.loc[:,param].max())
reversed_list = []

params_pretty = params.copy()
#params_pretty = ['Goals per 90', 'xG per 90', 'Goals per Shot']

# player one
nr1_id = player_index
nr1_name = df90.at[nr1_id, 'Player']
nr1_squad = df90.loc[nr1_id, 'Squad']
nr1_values = df90.loc[nr1_id, params].tolist()
# player two
nr2_id = 359
nr2_name = df90.at[nr2_id, 'Player']
nr2_squad = df90.loc[nr2_id, 'Squad']
nr2_values = df90.loc[nr2_id, params].tolist()
# other infos:
league_name, pos = 'OTP Bank liga', 'All position'

radar(params_pretty, low, high, reversed_list, 
      nr1_name, nr1_squad, nr1_values, 
      nr2_name, nr2_squad, nr2_values, 
      league_name, pos, matches_at_least,
      save=False, save_folder='C:/Users/Adam/Dropbox/TSDP_output/fbref/2025.06',
      save_name='2025.06.01, Álvarez-Olmo')
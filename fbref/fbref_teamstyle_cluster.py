#%% imports
import pandas as pd
from scipy.stats import rankdata

#%% fetch data
fetch = False
if fetch:
    from fbref.fbref_module import get_all_team_data_huv
    
    countrycode = 'ENG'
    season = '2023-2024'
    df = get_all_team_data_huv(countrycode, season)
else:
    df = pd.read_csv("fbref/Big5_teamdata.csv")

print(df.head)

#%%
def assign_quantile_category(series, labels=None):
    # Alapértelmezett kategóriák, ha nem adunk meg
    if labels is None:
        labels = ['Very Low', 'Low', 'High', 'Very High']
    quantiles = series.quantile([0.25, 0.5, 0.75]).values

    def categorize(x):
        if x <= quantiles[0]:
            return labels[0]
        elif x <= quantiles[1]:
            return labels[1]
        elif x <= quantiles[2]:
            return labels[2]
        else:
            return labels[3]

    return series.apply(categorize)

def classify_in_possession(row):
    instructions = []
    # Shoot on sight vs Work ball into box
    if row['Standard_Dist'] > df['Standard_Dist'].median():
        instructions.append('Shoot on sight')
    else:
        instructions.append('Work ball into box')

    # Wing play vs Through middle
    if row['Crosses_Opp'] > df['Crosses_Opp'].quantile(0.75):
        instructions.append('Focus play down the flanks')
    elif row['Passes_Thr'] > df['Passes_Thr'].quantile(0.75):
        instructions.append('Play through the middle')

    return ', '.join(instructions)

def classify_in_transition(row):
    instructions = []
    # Counter vs Hold shape
    if (row['Poss'] < df['Poss'].median()) and \
       (row['Progression_PrgC'] > df['Progression_PrgC'].median()):
        instructions.append('Counter')
    else:
        instructions.append('Hold shape')

    return ', '.join(instructions)

def classify_out_of_possession(row):
    instructions = []
    # High press / Mid block / Low block
    if row['Tackles_Att 3rd'] > df['Tackles_Att 3rd'].quantile(0.75):
        instructions.append('High press')
    elif row['Tackles_Mid 3rd'] > df['Tackles_Mid 3rd'].quantile(0.5):
        instructions.append('Mid block')
    else:
        instructions.append('Low block')

    # Get stuck in / Stay on feet
    if row['Performance_Fls'] > df['Performance_Fls'].quantile(0.75):
        instructions.append('Get stuck in')
    else:
        instructions.append('Stay on feet')

    return ', '.join(instructions)

def percentile_scores(series):
    """0–1 skálás percentilis értékek számítása."""
    return rankdata(series, method='average') / len(series)

def complex_mentality_core(df):
    # Biztosítsuk számként a fontos változókat
    for col in ['Poss', 'Standard_Sh/90', 'Passes_AvgLen', 'Crosses_Opp',
                'Tackles_Att 3rd', 'Tackles_Mid 3rd', 'Tackles_Def 3rd',
                'Progression_PrgC', 'Performance_Fls']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Percentilis skálás értékek
    df['Poss_pct'] = percentile_scores(df['Poss'])
    df['Shots_pct'] = percentile_scores(df['Standard_Sh/90'])
    df['Crosses_pct'] = percentile_scores(df['Crosses_Opp'])
    df['PassLen_pct'] = percentile_scores(df['Passes_AvgLen'])

    df['HighPress_pct'] = percentile_scores(df['Tackles_Att 3rd'])
    df['MidBlock_pct'] = percentile_scores(df['Tackles_Mid 3rd'])
    df['LowBlock_pct'] = percentile_scores(df['Tackles_Def 3rd'])
    df['Progression_pct'] = percentile_scores(df['Progression_PrgC'])
    df['Fouls_pct'] = percentile_scores(df['Performance_Fls'])

    # Mentality pontszám
    df['mentality_score'] = df['Poss_pct'] * 0.6 + df['Shots_pct'] * 0.4
    df['mentality_score'] = df['mentality_score'] * 100  # skála 0–100

    # Mentality kategória kvantilek alapján
    df['mentality_category'] = assign_quantile_category(
        df['mentality_score'],
        labels=['Very Defensive', 'Defensive', 'Attacking', 'Very Attacking']
    )

    # Core style logika folytonos mutatókra
    def core_style(row):
        if (row['Poss_pct'] > 0.6) and (row['PassLen_pct'] < 0.4):
            return 'Possession'
        elif row['Crosses_pct'] > 0.75:
            return 'Wing Play'
        elif (row['Poss_pct'] < 0.4) and (row['PassLen_pct'] > 0.6):
            return 'Direct Counter'
        else:
            return 'Balanced'

    df['core_style'] = df.apply(core_style, axis=1)

    # IP / IT / OOP címkék maradnak, de a percentilis score-okra is lehetne alapozni
    df['in_possession'] = df.apply(classify_in_possession, axis=1)
    df['in_transition'] = df.apply(classify_in_transition, axis=1)
    df['out_of_possession'] = df.apply(classify_out_of_possession, axis=1)

    return df[['Squad', 'Poss', 'Poss_pct', 'Standard_Sh/90', 'Shots_pct',
               'Passes_AvgLen', 'PassLen_pct', 'Crosses_Opp', 'Crosses_pct',
               'HighPress_pct', 'MidBlock_pct', 'LowBlock_pct',
               'Progression_pct', 'Fouls_pct',
               'mentality_score', 'mentality_category', 'core_style', 
               'in_possession', 'in_transition', 'out_of_possession']]

# Használat:
df_result = complex_mentality_core(df)
print(df_result)

#%% KMeans klaszterezés
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def kmeans_tactical_clusters(df, n_clusters=5):
    # Csak percentilis oszlopok
    features = [
        'Poss_pct', 'Shots_pct', 'PassLen_pct', 'Crosses_pct',
        'HighPress_pct', 'MidBlock_pct', 'LowBlock_pct',
        'Progression_pct', 'Fouls_pct'
    ]
    
    X = df[features].copy()

    # Itt a skála már 0–1, de biztos ami biztos, standardizálunk
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans klaszterezés
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[:, 'tactical_cluster'] = kmeans.fit_predict(X_scaled)

    return df, kmeans

# Használat:
df_result, kmeans_model = kmeans_tactical_clusters(df_result, n_clusters=5)
print(df_result[['Squad', 'tactical_cluster']])

#%% Hasonló csapat kereső
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# 1. Hasonló csapat kereső
def find_similar_teams(df, team_name, top_n=3):
    pct_cols = [c for c in df.columns if c.endswith('_pct')]
    team_vec = df.loc[df['Squad'] == team_name, pct_cols].values[0]

    # Abszolút különbségek
    df['abs_dist_sum'] = df[pct_cols].apply(lambda row: np.abs(row - team_vec).sum(), axis=1)
    df['abs_dist_median'] = df[pct_cols].apply(lambda row: np.median(np.abs(row - team_vec)), axis=1)

    # Eltávolítjuk önmagát
    result_sum = df[df['Squad'] != team_name].sort_values('abs_dist_sum').head(top_n)
    result_median = df[df['Squad'] != team_name].sort_values('abs_dist_median').head(top_n)

    return result_sum[['Squad', 'abs_dist_sum']], result_median[['Squad', 'abs_dist_median']]

# használat
closest_sum, closest_median = find_similar_teams(df_result, "Barcelona", top_n=7)
print("Legközelebbi (összesített távolság):")
print(closest_sum)
print("Legközelebbi (medián távolság):")
print(closest_median)


#%% 2. One-hot encoding + korrelációs elemzés
def expand_instruction_columns(df, instruction_cols):
    """
    instruction_cols: lista azokról az oszlopokról, amikben vesszővel elválasztott instrukciók vannak
    Minden instrukció külön boolean oszlop lesz prefixszel.
    """
    for col in instruction_cols:
        prefix = col[:2].upper()  # IP / IT / OO...
        # Egyedi instrukciók kigyűjtése
        unique_instr = set()
        df.loc[:, col] = df[col].fillna("")
        for instr_list in df[col]:
            for instr in [i.strip() for i in instr_list.split(",") if i.strip()]:
                unique_instr.add(instr)

        # Dummy oszlopok létrehozása
        for instr in sorted(unique_instr):
            dummy_col = f"{prefix}_{instr}"
            df.loc[:,dummy_col] = df.loc[:,col].apply(lambda x: instr in x)

    return df


def categorical_correlation_analysis(df, categorical_cols, method='pearson', plot=True):
    """
    A categorical_cols tartalmazhat már a dummy-oszlopokat is!
    """
    # One-hot encoding (ha szükséges)
    df_dummies = pd.get_dummies(df[categorical_cols], drop_first=False) \
                    if not all(df[categorical_cols].nunique() <= 2) else df[categorical_cols]

    # Korrelációs mátrix
    corr_matrix = df_dummies.corr(method=method)

    if plot:
        # Hőtérkép
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
        plt.title(f"{method.capitalize()} correlation heatmap (categorical vars)")
        plt.show()

        # Network graph
        G = nx.Graph()
        for col in corr_matrix.columns:
            G.add_node(col)
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                weight = corr_matrix.iloc[i, j]
                if abs(weight) > 0.2:
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=weight)

        pos = nx.spring_layout(G, seed=42)
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=[abs(w)*3 for w in weights])
        nx.draw_networkx_labels(G, pos, font_size=9)
        plt.title(f"Network graph (|{method}| > 0.5)")
        plt.axis('off')
        plt.show()

    return corr_matrix


# ===== Példa használat =====
# 1) Bővítjük a df-et dummy instr oszlopokkal
df_expanded = expand_instruction_columns(df_result, ['in_possession', 'in_transition', 'out_of_possession'])

# 2) Korrelációs elemzés az új oszlopokra + egyéb kategóriákra
categorical_cols = (
    ['mentality_category', 'core_style'] + 
    [c for c in df_expanded.columns if c.startswith(('IP_', 'IT_', 'OO_'))]
)
corr_mat = categorical_correlation_analysis(df_expanded, categorical_cols, method='spearman')

#%%
def expand_and_dummify(df, col_name):
    """
    Szétbontja a stringben összefűzött értékeket (', ' alapján),
    létrehozza a külön oszlopokat dummy-ként,
    majd visszaadja a dummy DataFrame-et.
    """
    # Szétválasztjuk az értékeket egy listába minden sorban
    exploded = df[col_name].str.split(', ')
    
    # Készítünk egy új DataFrame-et az egyedi dummy oszlopokkal
    # Minden elem kap egy saját oszlopot
    dummies = pd.get_dummies(exploded.apply(pd.Series).stack()).groupby(level=0).max()
    
    return dummies

# Kategóriás oszlopokat dummykra bontjuk külön-külön (pl. IP, IT, OOP)
ip_dummies = df['in_possession'].str.get_dummies(sep=', ')
it_dummies = df['in_transition'].str.get_dummies(sep=', ')
oop_dummies = df['out_of_possession'].str.get_dummies(sep=', ')

# Egyesítjük a numerikus adatokat és a dummy változókat egy DataFrame-be
df_for_corr = pd.concat([ip_dummies, it_dummies, oop_dummies], axis=1)

# Ekkor már csak számok vannak benne, így számolható a korreláció:
corr_mat = df_for_corr.corr(method='spearman')

# Ezt már vizualizálhatod pl. heatmap vagy network graph-ként
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 12))
sns.heatmap(corr_mat, cmap='coolwarm', center=0, linewidths=0.3)
plt.title('Spearman correlation matrix with expanded IP, IT, OOP features')
plt.show()

#%% Dummy korrelációk scouting interpretációval (inverz párok kiszűrve)

def interpret_dummy_correlations(corr_mat, threshold=0.4):
    """
    Erős dummy korrelációk kiemelése és scout-szintű interpretáció.
    Inverz párokat (pl. Shoot on sight vs Work ball into box) nem listáz.
    threshold: minimális abszolút korreláció (pl. 0.4)
    """
    findings = []
    
    # --- Inverz (kölcsönösen kizáró) párok előre definiálva ---
    inverse_groups = {
        "IP": ["Shoot on sight", "Work ball into box",
               "Focus play down the flanks", "Play through the middle"],
        "IT": ["Counter-press", "Regroup", "Counter", "Hold shape"],
        "OOP": ["High press", "Mid block", "Low block",
                "Get stuck in", "Stay on feet"]
    }
    
    # Egy segédfüggvény: benne vannak-e ugyanabban a kizáró csoportban?
    def is_inverse_pair(c1, c2):
        for group in inverse_groups.values():
            if any(g in c1 for g in group) and any(g in c2 for g in group):
                return True
        return False

    # --- Erős korrelációk kigyűjtése ---
    for i in range(len(corr_mat.columns)):
        for j in range(i+1, len(corr_mat.columns)):
            col1, col2 = corr_mat.columns[i], corr_mat.columns[j]
            corr = corr_mat.iloc[i, j]
            
            # Inverz párok átugrása
            if is_inverse_pair(col1, col2):
                continue

            if abs(corr) >= threshold:
                direction = "gyakran együtt jár" if corr > 0 else "ritkán fordul elő együtt"
                findings.append((col1, col2, corr, direction))
    
    # --- Szöveges riport ---
    print(f"\n📊 Scout interpretáció (|ρ| ≥ {threshold}, inverz párok kihagyva):\n")
    for f in findings:
        col1, col2, corr, direction = f
        if corr > 0:
            print(f"➡️ Ha egy csapat '{col1}' stílussal játszik, akkor általában '{col2}' is jellemző. (ρ={corr:.2f})")
        else:
            print(f"⚖️ A '{col1}' és '{col2}' inkább alternatívák: ha az egyik van, a másik kevésbé. (ρ={corr:.2f})")
    
    print("\n✅ Használat: Ez alapján scout előre jelezheti, "
          "hogy ha egy ellenfél pressingel, akkor várhatóan milyen további helyzetekre kell készülni "
          "(pl. kontrák, szélső játék, direkt stílus).")
    
    return findings

# Példahasználat:
dummy_corr_findings = interpret_dummy_correlations(corr_mat, threshold=0.3)

#%% Szomszédsági gráf + központiság elemzés dummy korrelációkra

import networkx as nx
import matplotlib.pyplot as plt

def build_style_network(corr_mat, threshold=0.4):
    """
    Szomszédsági gráf építése dummy korrelációkból.
    threshold: minimális abszolút korreláció (pl. 0.4)
    """
    G = nx.Graph()
    
    # Csomópontok
    for col in corr_mat.columns:
        G.add_node(col)
    
    # Élek hozzáadása (threshold alapján)
    for i in range(len(corr_mat.columns)):
        for j in range(i+1, len(corr_mat.columns)):
            col1, col2 = corr_mat.columns[i], corr_mat.columns[j]
            corr = corr_mat.iloc[i, j]
            
            if abs(corr) >= threshold:
                G.add_edge(col1, col2, weight=abs(corr), sign=np.sign(corr))
    
    return G

def analyze_style_network(G):
    """
    Központiság mutatók és modularitás elemzés
    """
    # Betweenness centrality
    betw = nx.betweenness_centrality(G, weight='weight')
    # Degree centrality
    deg = nx.degree_centrality(G)
    # Eigenvector centrality (fontosság „beágyazottság” alapján)
    eig = nx.eigenvector_centrality_numpy(G, weight='weight')
    
    # Közösségdetektálás (Louvain ha van, különben greedy modularity)
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    
    results = {
        'betweenness': betw,
        'degree': deg,
        'eigenvector': eig,
        'communities': communities
    }
    return results

def plot_style_network(G, centrality, title="Stílus-korrelációs hálózat"):
    """
    Stílusok hálózati vizualizációja központiság szerint színezve
    """
    pos = nx.spring_layout(G, seed=42, k=0.5)
    plt.figure(figsize=(10, 8))
    
    # csomópontméret központiság alapján
    node_sizes = [5000 * centrality[n] for n in G.nodes]
    node_colors = [centrality[n] for n in G.nodes]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=[d['weight']*2 for _,_,d in G.edges(data=True)],
                           edge_color='gray', alpha=0.6)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

# Példahasználat
G_styles = build_style_network(corr_mat, threshold=0.25)
results = analyze_style_network(G_styles)

print("\n📌 Legfontosabb stíluselemek (betweenness centrality alapján):")
print(sorted(results['betweenness'].items(), key=lambda x: -x[1])[:5])

print("\n📌 Közösségek a játékmodellekben:")
for i, comm in enumerate(results['communities']):
    print(f"  Modul {i+1}: {list(comm)}")

# Plot network
plot_style_network(G_styles, results['betweenness'], title="Stílus-hálózat (Betweenness Centrality)")


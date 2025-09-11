#%% import
import pandas as pd
import numpy as np
from fbref.fbref_module import get_all_player_data

df = get_all_player_data("ENG", year='2024-2025')

#%%
role_index_map = {
    'GK': {
        'Sweeper_Keeper': {
            'Sweeper_#OPA/90': 1.3,
            'Sweeper_AvgDist': 1.0,
            'Passes_Att (GK)': 0.6,
            'Passes_Launch%': -0.4   # alacsony launch% előny
        },
        'Shot_Stoper': {
            'Performance_Save%': 1.0,
            'Performance_CS%': 0.8,
            'Performance_Saves': 0.6,
            'Expected_PSxG+/-': 0.7
        },
        'Ball_Playing_GK': {
            'Launched_Cmp': 0.4,
            'Passes_Thr': 0.8,
            'Passes_Att (GK)': 0.8,
            'Goal Kicks_AvgLen': -0.4,  # inkább rövid játék
            'Total_PrgDist': 1.0
        }
    },
    'DF': {
        'Ball_Playing_CB': {
            'Total_Cmp%': 0.6,
            'Progression_PrgP': 0.8,
            'Carries_PrgDist': 0.7,
            'Pass Types_TB': 0.5
        },
        'No_Nonsense_CB': {
            'Clr': 1.0,
            'Blocks_Blocks': 0.8,
            'Aerial Duels_Won%': 0.7,
            'Total_Cmp%': -0.4
        },
        'Stopper_CB': {
            'Tackles_Tkl': 1.1,
            'Tkl+Int': 0.8,
            'Challenges_Tkl%': 0.7,
            'Err': -0.6,
        },
        'Libero_CB': {
            'Carries_PrgC': 0.8,
            'Carries_PrgDist': 0.7,
            'Progression_PrgP': 0.7,
            'Progression_PrgC': 0.5,
            'Touches_Mid 3rd': 0.5
        },
        'Inverted_FB': {
            'Total_Cmp%': 0.6,
            'Progression_PrgP': 0.7,
            'Carries_PrgDist': 0.6,
            'Progression_PrgC': 0.5,
            'Touches_Mid 3rd': 0.5
        },
        'Attacking_WB': {
            'CrsPA': 0.7,
            'Carries_PrgC': 0.7,
            'Expected_xAG': 0.6,
            'KP': 0.6,
            'Touches_Att 3rd': 0.6
        },
        'Defensive_FB': {
            'Tackles_Tkl': 0.7,
            'Tkl+Int': 0.8,
            'Challenges_Tkl%': 0.6,
            'Clr': 0.5
        },
        'Complete_WB': {
            'CrsPA': 0.6,
            'Carries_PrgDist': 0.6,
            'Tkl+Int': 0.9,
            'Progression_PrgP': 0.6,
            'Touches_Att 3rd': 0.7
        }
    },
    'MF': {
        'Deep_Lying_Playmaker': {
            'Progression_PrgP': 0.8,
            'Total_Cmp%': 0.7,
            'Expected_xAG': 0.6,
            'KP': 0.6
        },
        'Box_to_Box': {
            'Tkl+Int': 0.7,
            'Progression_PrgC': 0.6,
            'Progression_PrgP': 0.6,
            'Per 90 Minutes_G+A': 0.5,
            'Touches_Def 3rd': 0.4,
            'Touches_Att 3rd': 0.4
        },
        'Ball_Winning_MF': {
            'Tkl+Int': 0.9,
            'Tackles_Def 3rd': 0.6,
            'Challenges_Tkl%': 0.6,
            'Err': -0.5
        },
        'Mezzala': {
            'Carries_PrgC': 0.6,
            'Carries_1/3': 0.5,
            'KP': 0.6,
            'Per 90 Minutes_Ast': 0.5
        },
        'Advanced_Playmaker': {
            'Expected_xAG': 0.8,
            'KP': 0.8,
            'SCA_SCA90': 0.7,
            'Pass Types_TB': 0.6
        },
        'Anchor_Man': {
            'Tkl+Int': 0.7,
            'Tackles_Def 3rd': 0.7,
            'Err': -0.6,
            'Progression_PrgP': -0.6
        },
        'Carrilero': {
            'Touches_Mid 3rd': 0.6,
            'Total_Cmp%': 0.6,
            'Progression_PrgP': 0.5,
            'Per 90 Minutes_Ast': 0.3
        },
        'Shadow_Striker': {
            'Per 90 Minutes_Gls': 0.8,
            'Expected_npxG': 0.7,
            'Touches_Att Pen': 0.6,
            'SCA_SCA90': 0.5,
            'Standard_Sh': 0.5
        }
    },
    'FW': {
        'Inverted_Winger': {
            'Take-Ons_Succ': 1.1,
            'Carries_PrgC': 0.9,
            'Per 90 Minutes_Gls': 0.6,
            'Expected_npxG': 0.6
        },
        'Winger': {
            'Pass Types_Crs': 1.1,
            'CrsPA': 0.7,
            'KP': 0.6,
            'Carries_PrgDist': 0.6
        },
        'Inside_Forward': {
            'Per 90 Minutes_Gls': 1.0,
            'Expected_npxG': 1.0,
            'Take-Ons_Succ': 0.7,
            'Carries_1/3': 0.8
        },
        'Wide_Playmaker': {
            'KP': 1.0,
            'Expected_xAG': 0.8,
            'SCA_SCA90': 1.0,
            'Pass Types_TB': 0.6
        },
        'Raumdeuter': {
            'Touches_Att 3rd': 0.7,
            'Touches_Att Pen': 0.6,
            'Per 90 Minutes_Gls': 0.6,
            'Carries_PrgC': -0.8  # kevés labdaérintés
        },
        'Poacher': {
            'Per 90 Minutes_Gls': 0.9,
            'Expected_npxG': 0.8,
            'Touches_Att Pen': 0.7,
            'KP': -0.5
        },
        'Target_Man': {
            'Aerial Duels_Won%': 0.8,
            'Performance_PKwon': 0.5,
            'Per 90 Minutes_Gls': 0.5,
            'Ast': 0.4
        },
        'Pressing_Forward': {
            'Tkl+Int': 0.7,
            'Performance_PKwON': 0.5,
            'Per 90 Minutes_G+A': 0.5,
            'Performance_Fls': 0.4
        },
        'Deep_Lying_Forward': {
            'Ast': 0.7,
            'Expected_xAG': 0.7,
            'KP': 0.8,
            'Total_Cmp%': 0.6
        },
        'Complete_Forward': {
            'Per 90 Minutes_Gls': 0.7,
            'Ast': 0.6,
            'Tkl+Int': 0.5,
            'Take-Ons_Succ': 0.5
        },
        'False_9': {
            'Expected_xAG': 0.7,
            'KP': 0.7,
            'SCA_SCA90': 0.7,
            'Per 90 Minutes_Gls': 0.4,
            'Touches_Mid 3rd': 0.5
        }
    }
}

pos_map = {
    'GK': 'GK',
    'DF': 'DF',
    'MF': 'MF',
    'FW': 'FW'
}

# 1. Szűrés (legalább 5 mérkőzés / 5*90 perc)
df_filtered = df[df['90s'] >= 5].copy()

# 2. Kivonjuk az összes feature-t, ami szerepel a role_index_map-ben
def get_all_features(role_index_map):
    features = set()
    for pos_dict in role_index_map.values():
        for role, weights in pos_dict.items():
            features.update(weights.keys())
    return list(features)

all_features = get_all_features(role_index_map)
df_filtered['MainPos'] = df_filtered['Pos'].str[:2]

# Biztonság: csak olyan oszlopokat vegyünk, amik tényleg vannak a df-ben
available_features = [f for f in all_features if f in df_filtered.columns]

# 3. Percentile normalizálás 0–1 közé - groupby pozíció szerint
for feat in available_features:
    df_filtered[feat + "_pct"] = df_filtered.groupby("MainPos")[feat].rank(pct=True)


# 4. Index számító függvény
def calculate_role_index(player_row, role_weights):
    values = []
    weights = []
    for feat, w in role_weights.items():
        feat_col = feat + "_pct"
        if feat_col in player_row:
            val = player_row[feat_col]
            values.append(val * w)  # súlyozott percentil érték
            weights.append(abs(w))
    if len(values) == 0:
        return np.nan
    return np.sum(values) / np.sum(weights)  # normalizált súlyozott átlag

# 5. Végigszámoljuk minden szerepkörre
for idx, row in df_filtered.iterrows():
    player_pos = None
    for key in pos_map:
        if key in row['Pos']:   # pl. "DF,MF" is lehet
            player_pos = pos_map[key]
            break
    if player_pos is None:
        continue  # nincs ismert poszt
    
    # Csak a releváns szerepekre számolunk
    for role, weights in role_index_map[player_pos].items():
        col_name = f"Index_{role}"
        df_filtered.loc[idx, col_name] = calculate_role_index(row, weights)
        
# ✅ Eredmény: df_filtered tartalmazza az Index_* oszlopokat minden szerepkörre

#%% correlations
import matplotlib.pyplot as plt

def plot_role_corr(df, pos_key, role_dict):
    pos_df = df[df['Pos'].str.contains(pos_key)]
    
    # kiválasztjuk az index oszlopokat
    role_cols = [f"Index_{role}" for role in role_dict.keys()]
    available_cols = [c for c in role_cols if c in pos_df.columns]
    
    if len(available_cols) < 2:
        print(f"Nincs elég adat {pos_key}-hoz")
        return
    
    corr = pos_df[available_cols].corr()
    
    plt.figure(figsize=(8,6))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), [c.replace("Index_","") for c in corr.columns], rotation=90)
    plt.yticks(range(len(corr.index)), [c.replace("Index_","") for c in corr.index])
    plt.title(f"Correlation matrix – {pos_key}")
    plt.show()

# Használat minden posztra:
for pos_key, role_dict in role_index_map.items():
    plot_role_corr(df_filtered, pos_key, role_dict)


#%% find best for role
role_to_check = 'Attacking_WB'
df_filtered[['Player', 'Squad', 'Pos', f'Index_{role_to_check}']].sort_values(by=f'Index_{role_to_check}', ascending=False).head(10)

#%% find best role for player
index_cols = [col for col in df_filtered.columns if 'Index_' in col]

player_id = 281
best_value = 0
best_index = 'None'
for col in index_cols:
    value = df_filtered.at[player_id, col]
    if pd.notna(value):
        if best_value < value:
            best_value = value
            best_index = col
print(best_index, best_value)

#%% Guardiola-féle 4-3-3 szerepkör mapping
guardiola_433_roles = {
    "GK": "Sweeper_Keeper",
    "RB": "Inverted_FB",
    "RCB": "Ball_Playing_CB",
    "LCB": "Ball_Playing_CB",
    "LB": "Inverted_FB",
    "CDM": "Deep_Lying_Playmaker",
    "RCM": "Mezzala",
    "LCM": "Box_to_Box",
    "RW": "Inverted_Winger",
    "LW": "Inside_Forward",
    "ST": "False_9"
}

def select_best_eleven_unique(df, formation_roles, n=1):
    """
    df: df_filtered játékos statisztikákkal
    formation_roles: dict {poszt: role}
    n: posztonként hány játékost hozzunk
    """
    best_eleven = {}
    selected_players = set()  # már kiválasztott játékosok

    for pos, role in formation_roles.items():
        col_name = f"Index_{role}"
        if col_name not in df.columns:
            print(f"⚠️ {col_name} nincs a dataframe-ben!")
            continue

        # Szűrés: csak releváns poszton játszók
        if pos in ["GK"]:
            candidates = df[df["MainPos"] == "GK"]
        elif pos in ["RB","RCB","LCB","LB"]:
            candidates = df[df["MainPos"] == "DF"]
        elif pos in ["CDM","RCM","LCM"]:
            candidates = df[df["MainPos"] == "MF"]
        elif pos in ["RW","LW","ST"]:
            candidates = df[df["MainPos"] == "FW"]
        else:
            candidates = df.copy()

        # Kizárjuk azokat, akik már kaptak helyet
        candidates = candidates[~candidates["Player"].isin(selected_players)]

        # Kiválasztás index alapján
        top_players = (
            candidates
            .sort_values(col_name, ascending=False)
            .head(n)[["Player","Squad","Pos",col_name]]
        )

        # Frissítjük a selected_players listát
        selected_players.update(top_players["Player"].tolist())

        best_eleven[pos] = {
            "role": role,
            "players": top_players.to_dict(orient="records")
        }

    return best_eleven

# Használat
best_team_unique = select_best_eleven_unique(df_filtered, guardiola_433_roles, n=1)

# Kiíratás
for pos, info in best_team_unique.items():
    role = info["role"]
    players = info["players"]
    print(f"{pos} ({role}):")
    for p in players:
        print(f"  {p['Player']} – {p['Squad']} – {p['Pos']} – {p[f'Index_{role}']:.2f}")
    print()

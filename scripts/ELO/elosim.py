import pandas as pd
import requests
from io import StringIO

url = 'http://api.clubelo.com/2025-06-22'

try:
    response = requests.get(url)
    response.raise_for_status()  # Hibát dob, ha a válasz státusza nem 200 OK
    
    # Kiírjuk a válasz tartalmát
    print("Válasz státusz kód:", response.status_code)
    print("\nVálasz tartalma:")
    rtext = response.text
    print(rtext)
    
except requests.exceptions.HTTPError as http_err:
    print(f'HTTP hiba történt: {http_err}')
except requests.exceptions.RequestException as err:
    print(f'Hiba történt a kérés során: {err}')
    
df = pd.read_csv(StringIO(rtext))

df = df.sort_values(by="Elo", ascending=False).reset_index(drop=True)
df['Rank'] = df.index + 1

#%% Calculate ELO probs
import math
import random

# Qualifier games
qualifiers = {
    'Champions Path': [
        {
            'matches': (('Flora Tallinn', 'RFS'), ('Dinamo Tbilisi', 'Malmoe')),
            'seeded': []
        },
        {
            'matches': (('Zalgiris Vilnius', 'Hamrun'), ('Dynamo Kyiv',)),
            'seeded': ['Dynamo Kyiv']
        },
        {
            'matches': (('Paphos',), ('M Tel Aviv',)),
            'seeded': ['M Tel Aviv']
        },
        {
            'matches': (('Vikingur', 'Lincoln'), ('Crvena Zvezda',)),
            'seeded': ['Crvena Zvezda']
        },
        {
            'matches': (('Noah', 'Buducnost Podgorica'), ('Ferencvaros',)),
            'seeded': ['Ferencvaros']
        },
        {
            'matches': (('Lech',), ('Egnatia', 'Breidablik')),
            'seeded': ['Lech']
        },
        {
            'matches': (('FC Kobenhavn',), ('Drita', 'Differdang')),
            'seeded': ['FC Kobenhavn']
        },
        {
            'matches': (('Rijeka',), ('Razgrad', 'Dinamo Minsk')),
            'seeded': ['Razgrad', 'Dinamo Minsk']
        },
        {
            'matches': (('The New Saints', 'Shkendija'), ('Steaua', 'Inter Club d\'Escaldes')),
            'seeded': ['Steaua', 'Inter Club d\'Escaldes']
        },
        {
            'matches': (('Slovan Bratislava',), ('SS Virtus', 'Zrinjski Mostar')),
            'seeded': ['Slovan Bratislava']
        },
        {
            'matches': (('Shelbourne', 'Linfield'), ('Karabakh Agdam',)),
            'seeded': ['Karabakh Agdam']
        },
        {
            'matches': (('Kuopio', 'Milsami Orhei'), ('Olimpija Ljubljana', 'FK Astana')),
            'seeded': ['Olimpija Ljubljana', 'FK Astana']
        }
    ],
    'League Path': [
        {
            'matches': (('Brann',), ('Salzburg',)),
            'seeded': ['Salzburg']
        },
        {
            'matches': (('Viktoria Plzen',), ('Servette',)),
            'seeded': ['Viktoria Plzen']
        },
        {
            'matches': (('Rangers',), ('Panathinaikos',)),
            'seeded': ['Rangers']
        }
    ]
}

def calculate_win_probabilities(elo1, elo2):
    """
    Kiszámolja a győzelem valószínűségeket két ELO érték alapján
    
    Args:
        elo1: Az első csapat ELO pontszáma
        elo2: A második csapat ELO pontszáma
    
    Returns:
        Tuple (prob1, prob2) ahol:
        - prob1: Az első csapat győzelmi valószínűsége
        - prob2: A második csapat győzelmi valószínűsége
    """
    # ELO különbség
    elo_diff = elo1 - elo2
    
    # Az első csapat győzelmi valószínűsége (logisztikus függvény)
    prob1 = 1 / (1 + math.pow(10, -elo_diff / 400))
    
    # A második csapat győzelmi valószínűsége
    prob2 = 1 - prob1
    
    return prob1, prob2

# ELO szótár létrehozása a DataFrame-ből
def create_elo_dict(df):
    elo_dict = {}
    for _, row in df.iterrows():
        # Kisbetűsítés és szóközök eltávolítása a jobb illesztésért
        key = row['Club'].lower().replace(' ', '')
        elo_dict[key] = row['Elo']
    return elo_dict

# Csapatnév normalizálása
def normalize_team_name(team_name):
    return team_name.lower().replace(' ', '').replace('-', '').replace('.', '').replace('\'', '')

# Mérkőzés szimulációja
def simulate_match(team1, team2, elo_dict):
    # Csapatnevek normalizálása
    norm_team1 = normalize_team_name(team1)
    norm_team2 = normalize_team_name(team2)
    
    # ELO értékek lekérése (alapértelmezett 1000 ha nem található)
    elo1 = elo_dict.get(norm_team1, 1000)
    elo2 = elo_dict.get(norm_team2, 1000)
    
    # Győzelem valószínűségek kiszámolása
    prob1, _ = calculate_win_probabilities(elo1, elo2)
    return team1 if random.random() < prob1 else team2

# Node szimuláció (rekurzív)
def simulate_node(node, elo_dict):
    if isinstance(node, str):
        return node
    
    if isinstance(node, tuple):
        if len(node) == 1:
            return simulate_node(node[0], elo_dict)
        else:
            team1_node = simulate_node(node[0], elo_dict)
            team2_node = simulate_node(node[1], elo_dict)
            
            # Ha mindkét részeredmény string, akkor mérkőzés szimuláció
            if isinstance(team1_node, str) and isinstance(team2_node, str):
                return simulate_match(team1_node, team2_node, elo_dict)
            else:
                # Hibakezelés, ha nem megfelelő típusú az eredmény
                raise ValueError(f"Érvénytelen node típus: {type(team1_node)}, {type(team2_node)}")
    
    # Hibakezelés más típusok esetén
    raise ValueError(f"Érvénytelen node típus: {type(node)}")

def extract_teams_from_node(node):
    teams = []
    if isinstance(node, str):
        teams.append(node)
    elif isinstance(node, tuple):
        for item in node:
            teams.extend(extract_teams_from_node(item))
    return teams

# Összes csapat gyűjtése a qualifiers-ből
def get_all_teams(qualifiers):
    teams = set()
    
    def extract_teams(node):
        if isinstance(node, str):
            teams.add(node)
        elif isinstance(node, tuple):
            for item in node:
                extract_teams(item)
    
    for path, groups in qualifiers.items():
        for group in groups:
            match = group['matches']
            extract_teams(match)
    
    return teams

def pair_seeded_vs_unseeded(seeded, unseeded):
    """
    Kiemelt vs nem kiemelt csapatok véletlenszerű párosítása.
    """
    random.shuffle(seeded)
    random.shuffle(unseeded)
    pairings = []
    for s, u in zip(seeded, unseeded):
        pairings.append((s, u))
    return pairings


# Fő szimulációs folyamat
def run_simulation(qualifiers, df, num_simulations=10000):
    # ELO szótár létrehozása
    elo_dict = create_elo_dict(df)
    
    # Összes csapat azonosítása
    all_teams = get_all_teams(qualifiers)
    
    # Számlálók inicializálása
    counters = {team: 0 for team in all_teams}
    
    # Szimulációk futtatása
    for _ in range(num_simulations):
        # Győztesek halmaza ebben a szimulációban
        qualified_teams = set()
        
        # Minden ágon (Champions Path, League Path) végigmegyünk
        for path, groups in qualifiers.items():
            # Az útvonal összes csapata
            all_path_teams = []
            for group in groups:
                all_path_teams.extend(extract_teams_from_node(group))
            
            # 1. kör szimulációja
            round1_winners = []
            for group in groups:
                winner = simulate_node(group['matches'], elo_dict)
                round1_winners.append(winner)
            
            # 2. kör: Felosztás seeded/unseeded alapján
            round2_seeded = sorted(round1_winners, key=lambda t: elo_dict.get(normalize_team_name(t), 1000), reverse=True)[:len(round1_winners)//2]
            round2_unseeded = [t for t in round1_winners if t not in round2_seeded]
            
            round2_matchups = pair_seeded_vs_unseeded(round2_seeded, round2_unseeded)

            # 2. kör szimulációja
            round2_winners = []
            for matchup in round2_matchups:
                winner = simulate_match(matchup[0], matchup[1], elo_dict)
                round2_winners.append(winner)
            
            # 3. kör: véletlenszerű párosítás
            random.shuffle(round2_winners)
            round3_matchups = []
            for i in range(0, len(round2_winners), 2):
                if i+1 < len(round2_winners):
                    round3_matchups.append((round2_winners[i], round2_winners[i+1]))
                else:
                    qualified_teams.add(round2_winners[i])
            
            # 3. kör szimulációja
            round3_winners = []
            for matchup in round3_matchups:
                winner = simulate_match(matchup[0], matchup[1], elo_dict)
                round3_winners.append(winner)
            
            # 4. kör: véletlenszerű párosítás
            random.shuffle(round3_winners)
            round4_matchups = []
            for i in range(0, len(round3_winners), 2):
                if i+1 < len(round3_winners):
                    round4_matchups.append((round3_winners[i], round3_winners[i+1]))
                else:
                    qualified_teams.add(round3_winners[i])
            
            # 4. kör szimulációja és továbbjutók hozzáadása
            for matchup in round4_matchups:
                winner = simulate_match(matchup[0], matchup[1], elo_dict)
                qualified_teams.add(winner)
        
        # Győztesek számolása
        for team in qualified_teams:
            counters[team] += 1
    
    # Eredmények összesítése
    results = []
    for team, count in counters.items():
        probability = count / num_simulations
        results.append({
            'Team': team,
            'Qualification Probability': probability,
            'Simulation Count': count
        })
    
    return pd.DataFrame(results)

# Szimuláció futtatása
results_df = run_simulation(qualifiers, df, num_simulations=10000)

# Eredmények rendezése és megjelenítése
results_df = results_df.sort_values(by='Qualification Probability', ascending=False)
print(results_df.head(10))

#%%
import matplotlib.pyplot as plt
# Rendezés a valószínűség szerint
sorted_df = results_df.sort_values('Qualification Probability', ascending=False)[0:15]
sorted_df = sorted_df.sort_values('Qualification Probability', ascending=True)

# Ábra beállítás
plt.figure(figsize=(7, 8))
bars = plt.barh(sorted_df['Team'], sorted_df['Qualification Probability'], color='#5ECB43')
plt.xlabel('Qualification Probability')
plt.title('Champions League Qualification Probability - 10,000 ELO Simulations')

# Valószínűségi értékek kiírása a sávok mellé
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
             f'{width:.2%}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(r'C:\Users\Adam\Dropbox\TSDP_output\Random data\qualification_probabilities.png', dpi=300)
plt.show()

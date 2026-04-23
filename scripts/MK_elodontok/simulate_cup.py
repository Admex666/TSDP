import sys
import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def get_rating(leagues):
    if not isinstance(leagues, str):
        return 200
    leagues = leagues.lower()
    # NB I / First Division
    if any(x in leagues for x in ['fizz', 'nb i', 'champions', 'europa', 'nbi']):
        if 'nb ii' not in leagues or 'nb i ' in leagues: # avoid false positive with NB II
             return 1000
    # NB II
    if 'nb ii' in leagues or 'nbii' in leagues:
        return 700
    # NB III
    if 'nb iii' in leagues or 'nbiii' in leagues:
        return 500
    # County / BLSZ
    if any(x in leagues for x in ['oszt', 'blsz', 'megye', 'területi']):
        return 300
    # Default (usually small cup participants)
    return 200

def win_probability(rating_a, rating_b):
    # Elo-style probability
    return 1 / (1 + 10 ** (-(rating_a - rating_b) / 400))

def simulate_tournament(bracket, team_ratings):
    # bracket is a list of rounds, each round is a list of matches (team1, team2)
    current_teams = []
    # Round 0: 64 teams in 32 matches
    for m in bracket[0]:
        current_teams.append(m[0])
        current_teams.append(m[1])
        
    winners = current_teams
    
    # Progress through rounds
    for r in range(5): # 32->16->8->4->2->1
        next_winners = []
        for i in range(0, len(winners), 2):
            t1 = winners[i]
            t2 = winners[i+1]
            p1 = win_probability(team_ratings.get(t1, 200), team_ratings.get(t2, 200))
            if random.random() < p1:
                next_winners.append(t1)
            else:
                next_winners.append(t2)
        winners = next_winners
        
    return winners[0], winners # Return winner and final participants

def main():
    # 1. Load data
    matches_df = pd.read_csv("magyar_kupa_matches_sofascore.csv")
    teams_df = pd.read_csv("magyar_kupa_teams_leagues.csv")
    
    # 2. Assign ratings
    team_ratings = {}
    rating_counts = {}
    for _, row in teams_df.iterrows():
        r = get_rating(row['leagues'])
        team_ratings[row['team_name']] = r
        rating_counts[r] = rating_counts.get(r, 0) + 1
    
    print(f"Rating distribution: {rating_counts}")
        
    # 3. Reconstruct Bracket
    # We assume the CSV rows for Round 3 are the base of the bracket
    r3_matches = matches_df[matches_df['round'] == 'Round 3']
    if len(r3_matches) != 32:
        print(f"Warning: Found {len(r3_matches)} matches for Round 3, expected 32.")
        
    # The order in the CSV/Tree is important.
    # We'll use the order of matches in Round 3 as the initial bracket
    initial_bracket = []
    for _, row in r3_matches.iterrows():
        initial_bracket.append((row['home_team'], row['away_team']))
        
    # 4. Monte Carlo Simulation
    num_simulations = 10000
    final_counts = {} # How many times a team reached the final
    
    all_team_names = pd.concat([r3_matches['home_team'], r3_matches['away_team']]).unique()
    for team in all_team_names:
        final_counts[team] = 0
        
    print(f"Running {num_simulations} simulations...")
    for _ in tqdm(range(num_simulations)):
        # Simulate until the semi-finals are done to see who is in the final
        # Or just simulate the whole thing and count the last two
        
        winners = [list(m) for m in initial_bracket] # Start with pairs
        
        # We need to simulate 5 rounds to get the 2 finalists
        # R3 -> R32 (16 matches)
        # R32 -> R16 (8 matches)
        # R16 -> QF (4 matches)
        # QF -> SF (2 matches) -> These are the finalists
        
        current_round_winners = []
        # Round 3 (32 matches)
        for t1, t2 in initial_bracket:
            p1 = win_probability(team_ratings.get(t1, 200), team_ratings.get(t2, 200))
            current_round_winners.append(t1 if random.random() < p1 else t2)
            
        # Round of 32 (16 matches)
        next_round_winners = []
        for i in range(0, len(current_round_winners), 2):
            t1, t2 = current_round_winners[i], current_round_winners[i+1]
            p1 = win_probability(team_ratings.get(t1, 200), team_ratings.get(t2, 200))
            next_round_winners.append(t1 if random.random() < p1 else t2)
        current_round_winners = next_round_winners
        
        # Round of 16 (8 matches)
        next_round_winners = []
        for i in range(0, len(current_round_winners), 2):
            t1, t2 = current_round_winners[i], current_round_winners[i+1]
            p1 = win_probability(team_ratings.get(t1, 200), team_ratings.get(t2, 200))
            next_round_winners.append(t1 if random.random() < p1 else t2)
        current_round_winners = next_round_winners
        
        # Quarterfinals (4 matches)
        next_round_winners = []
        for i in range(0, len(current_round_winners), 2):
            t1, t2 = current_round_winners[i], current_round_winners[i+1]
            p1 = win_probability(team_ratings.get(t1, 200), team_ratings.get(t2, 200))
            next_round_winners.append(t1 if random.random() < p1 else t2)
        current_round_winners = next_round_winners
        
        # Semifinals (2 matches) -> The winners reach the final
        for i in range(0, len(current_round_winners), 2):
            t1, t2 = current_round_winners[i], current_round_winners[i+1]
            p1 = win_probability(team_ratings.get(t1, 200), team_ratings.get(t2, 200))
            finalist = t1 if random.random() < p1 else t2
            final_counts[finalist] += 1
            
    # 5. Results
    results = []
    for team, count in final_counts.items():
        results.append({
            'Team': team,
            'League': teams_df[teams_df['team_name'] == team]['leagues'].values[0] if team in teams_df['team_name'].values else 'Unknown',
            'Final_Prob': count / num_simulations
        })
        
    df_results = pd.DataFrame(results).sort_values('Final_Prob', ascending=False)
    
    output_file = "magyar_kupa_final_probabilities.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {output_file}")
    print("\nTop 10 teams by Final Probability:")
    print(df_results.head(10).to_string(index=False))

if __name__ == "__main__":
    main()

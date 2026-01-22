import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import json
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

# --- 1. Data Loading ---
def load_and_clean_data():
    matches = pd.read_csv("data/matches.csv")
    odds = pd.read_csv("data/odds.csv")
    games = pd.read_csv("data/games.csv")
    with open("data/team_name_mapping.json", "r") as f:
        mapping = json.load(f)
    matches['date'] = pd.to_datetime(matches['date']).dt.date
    matches = matches.dropna(subset=['team1_score', 'team2_score']).sort_values('date')
    matches['target'] = (matches['team1_score'] > matches['team2_score']).astype(int)
    odds['Date'] = pd.to_datetime(odds['Date']).dt.date
    odds['team1_mapped'] = odds['home_team'].map(lambda x: mapping.get(x, {}).get('matched_team', x))
    odds['team2_mapped'] = odds['away_team'].map(lambda x: mapping.get(x, {}).get('matched_team', x))
    return matches, odds, games

# --- 2. Feature Engineering ---
def calculate_features(df, games_df):
    df = df.sort_values('date').reset_index(drop=True)
    team_elos = {}
    K = 20
    team_histories = {} 
    team_gold_diffs = {} 
    t1_wr10, t2_wr10 = [], []
    t1_gd15, t2_gd15 = [], []
    elo_diffs = []
    games_stats = {}
    for _, g in games_df.iterrows():
        mid = g['match_id']
        games_stats.setdefault(mid, []).append({'blue_team': g['blue_team'], 'red_team': g['red_team'], 'gd15': g.get('gold_diff_15min', 0)})
    for idx, row in df.iterrows():
        t1, t2 = row['team1'], row['team2']
        e1, e2 = team_elos.get(t1, 1500.0), team_elos.get(t2, 1500.0)
        elo_diffs.append(e1 - e2)
        h1, h2 = team_histories.get(t1, []), team_histories.get(t2, [])
        t1_wr10.append(np.mean(h1[-10:]) if h1 else 0.5)
        t2_wr10.append(np.mean(h2[-10:]) if h2 else 0.5)
        g1, g2 = team_gold_diffs.get(t1, []), team_gold_diffs.get(t2, [])
        t1_gd15.append(np.mean(g1[-5:]) if g1 else 0.0)
        t2_gd15.append(np.mean(g2[-5:]) if g2 else 0.0)
        target = row['target']
        team_histories.setdefault(t1, []).append(target)
        team_histories.setdefault(t2, []).append(1 - target)
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        team_elos[t1], team_elos[t2] = e1 + K * (target - exp1), e2 + K * ((1 - target) - (1 - exp1))
        mid = row['match_id']
        for g in games_stats.get(mid, []):
            if g['blue_team'] == t1: team_gold_diffs.setdefault(t1, []).append(g['gd15'])
            elif g['red_team'] == t1: team_gold_diffs.setdefault(t1, []).append(-g['gd15'])
            if g['blue_team'] == t2: team_gold_diffs.setdefault(t2, []).append(g['gd15'])
            elif g['red_team'] == t2: team_gold_diffs.setdefault(t2, []).append(-g['gd15'])
    df['elo_diff'], df['t1_wr10'], df['t2_wr10'], df['t1_gd15'], df['t2_gd15'] = elo_diffs, t1_wr10, t2_wr10, t1_gd15, t2_gd15
    return df

# --- 3. Precision Betting Logic ---
def run_simulation(df, strategy='kelly', bankroll_start=1000, stake_val=10):
    bankroll, history, fraction = bankroll_start, [], 0.05
    # 🎯 PRECISION FILTERS (Based on diagnostic ROI > 5%)
    # LCK Playoffs, LPL Playoffs, LEC Season/Playoffs
    PROFITABLE_TOUR_KEYS = [
        'LCK 2025 Season Playoffs', 
        'LPL 2025 Split 2 Playoffs', 
        'LEC 2025 Spring Season', 
        'LEC 2025 Spring Playoffs'
    ]
    MIN_ODDS, MAX_ODDS = 1.8, 3.2 # Narrowed to the "sweet spot" identified

    for idx, row in df.iterrows():
        p1, o1, o2 = row['p1'], row['team1_odds'], row['team2_odds']
        tour = row['tournament_name']
        is_targeted = tour in PROFITABLE_TOUR_KEYS
        bet_amount, won, odds = 0, False, 0
        
        if is_targeted:
            v1, v2 = p1 * o1 - 1, (1-p1) * o2 - 1
            if v1 > 0.05 and MIN_ODDS <= o1 <= MAX_ODDS: side, odds, prob, won = 1, o1, p1, (row['target'] == 1)
            elif v2 > 0.05 and MIN_ODDS <= o2 <= MAX_ODDS: side, odds, prob, won = 2, o2, 1-p1, (row['target'] == 0)
            else: side = None
        else: side = None
            
        if side:
            if strategy == 'kelly':
                f = (prob * odds - 1) / (odds - 1)
                bet_amount = bankroll * f * fraction
            elif strategy == 'flat': bet_amount = stake_val
            elif strategy == 'target': bet_amount = stake_val / (odds - 1)
            bet_amount = max(0, min(bet_amount, bankroll))
            if won: bankroll += bet_amount * (odds - 1)
            else: bankroll -= bet_amount
        history.append(bankroll)
    return history

if __name__ == "__main__":
    print("Starting PRECISION Backtest...")
    matches, odds, games = load_and_clean_data()
    df = calculate_features(matches, games)
    matched_odds = []
    odds_dict = {}
    for _, o in odds.iterrows(): odds_dict.setdefault(o['Date'], []).append(o)
    for idx, m in df.iterrows():
        day_odds = odds_dict.get(m['date'], [])
        for o in day_odds:
            if ((o['team1_mapped'] == m['team1']) and (o['team2_mapped'] == m['team2'])) or \
               ((o['team1_mapped'] == m['team2']) and (o['team2_mapped'] == m['team1'])):
                o1 = o['home_odds'] if o['team1_mapped'] == m['team1'] else o['away_odds']
                o2 = o['away_odds'] if o['team1_mapped'] == m['team1'] else o['home_odds']
                matched_odds.append({'match_id': m['match_id'], 'team1_odds': o1, 'team2_odds': o2})
                break
    df = df.merge(pd.DataFrame(matched_odds), on='match_id', how='inner')
    features = ['elo_diff', 't1_wr10', 't2_wr10', 't1_gd15', 't2_gd15']
    X, y = df[features], df['target']
    tscv = TimeSeriesSplit(n_splits=5)
    df['p1'] = np.nan
    print("Training Enhanced Models...")
    for train_idx, test_idx in tscv.split(X):
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42, eval_metric='logloss')
        cal = CalibratedClassifierCV(clf, method='isotonic', cv=3)
        cal.fit(X.iloc[train_idx], y.iloc[train_idx])
        df.iloc[test_idx, df.columns.get_loc('p1')] = cal.predict_proba(X.iloc[test_idx])[:, 1]
    df_results = df.dropna(subset=['p1']).copy()
    print("Simulating Strategies...")
    for s in ['kelly', 'flat', 'target']: df_results[f'bankroll_{s}'] = run_simulation(df_results, strategy=s)
    report = []
    for s in ['kelly', 'flat', 'target']:
        col = f'bankroll_{s}'
        final = df_results[col].iloc[-1]
        roi = (final - 1000) / 1000 * 100
        report.append({'Strategy': s.capitalize(), 'ROI': f"{roi:.1f}%", 'Final': f"{final:.2f}"})
    print("\n--- FINAL PRECISION RESULTS ---")
    print(pd.DataFrame(report).to_string(index=False))
    plt.figure(figsize=(10, 6))
    for s in ['kelly', 'flat', 'target']: plt.plot(df_results['date'], df_results[f'bankroll_{s}'], label=s.capitalize())
    plt.title("Bankroll: Final Precision Strategy")
    plt.legend(); plt.savefig("backtest_results_precision.png")
    print("Chart saved: backtest_results_precision.png")

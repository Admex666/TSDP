import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from backtest_engine import load_and_clean_data, calculate_features
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

def run_diagnostics():
    print("🚀 Loading data for diagnostics...")
    matches, odds, games = load_and_clean_data()
    df = calculate_features(matches, games)
    
    # Merge odds
    matched_odds = []
    odds_dict = {}
    for _, o in odds.iterrows():
        odds_dict.setdefault(o['Date'], []).append(o)
        
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
    
    print("🧠 Re-training Models...")
    for train_idx, test_idx in tscv.split(X):
        clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
        cal = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
        cal.fit(X.iloc[train_idx], y.iloc[train_idx])
        df.iloc[test_idx, df.columns.get_loc('p1')] = cal.predict_proba(X.iloc[test_idx])[:, 1]
        
    results = df.dropna(subset=['p1']).copy()
    
    # Calculate EV for both sides
    results['ev1'] = results['p1'] * results['team1_odds'] - 1
    results['ev2'] = (1 - results['p1']) * results['team2_odds'] - 1
    
    # Prepare diagnostic dataframe
    diag_rows = []
    for idx, row in results.iterrows():
        # Side 1
        diag_rows.append({
            'match_id': row['match_id'],
            'tournament': row['tournament_name'],
            'odds': row['team1_odds'],
            'ev': row['ev1'],
            'won': (row['target'] == 1),
            'prob': row['p1']
        })
        # Side 2
        diag_rows.append({
            'match_id': row['match_id'],
            'tournament': row['tournament_name'],
            'odds': row['team2_odds'],
            'ev': row['ev2'],
            'won': (row['target'] == 0),
            'prob': 1 - row['p1']
        })
    
    diag_df = pd.DataFrame(diag_rows)
    diag_df['profit'] = np.where(diag_df['won'], diag_df['odds'] - 1, -1)
    
    print("\n--- Diagnostic: ROI by EV Threshold ---")
    ev_thresholds = np.arange(-0.1, 0.3, 0.05)
    ev_results = []
    for t in ev_thresholds:
        subset = diag_df[diag_df['ev'] > t]
        if len(subset) > 0:
            roi = subset['profit'].sum() / len(subset) * 100
            ev_results.append({'Threshold': t, 'ROI': roi, 'Bets': len(subset)})
    print(pd.DataFrame(ev_results).to_string(index=False))
    
    print("\n--- Diagnostic: ROI by Odds Range ---")
    bins = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    diag_df['odds_bin'] = pd.cut(diag_df['odds'], bins=bins)
    # Only look at positive EV bets for odds analysis
    odds_roi = diag_df[diag_df['ev'] > 0.05].groupby('odds_bin')['profit'].agg(['sum', 'count'])
    odds_roi['ROI'] = odds_roi['sum'] / odds_roi['count'] * 100
    print(odds_roi[['ROI', 'count']])
    
    print("\n--- Diagnostic: ROI by Tournament (EV > 0.05) ---")
    tour_roi = diag_df[diag_df['ev'] > 0.05].groupby('tournament')['profit'].agg(['sum', 'count'])
    tour_roi['ROI'] = tour_roi['sum'] / tour_roi['count'] * 100
    print(tour_roi[tour_roi['count'] > 5].sort_values('ROI', ascending=False).head(10))

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=pd.DataFrame(ev_results), x='Threshold', y='ROI')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("ROI vs EV Threshold")
    plt.savefig("diagnostic_ev_roi.png")
    
if __name__ == "__main__":
    run_diagnostics()

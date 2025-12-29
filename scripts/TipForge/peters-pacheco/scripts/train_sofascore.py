import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Add src path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

# We can reuse the model classes if they exist, or implement robustly here.
# For simplicity and transparency, let's implement the logic explicitly here first 
# to ensure it fits the new SofaScore data structure perfectly.

def calculate_match_outcome_probabilities(home_goals_avg, away_goals_avg, max_goals=10):
    """
    Converts expected goals (lambda) to Win/Draw/Loss probabilities using Poisson distribution.
    """
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    
    # Outer product to get matrix of score probabilities
    score_matrix = np.outer(team_pred[0], team_pred[1])
    
    # Sum standard diagonals for W/D/L
    home_win = np.sum(np.tril(score_matrix, -1))
    draw = np.sum(np.diag(score_matrix))
    away_win = np.sum(np.triu(score_matrix, 1))
    
    return home_win, draw, away_win

def get_implied_probabilities(odds_home, odds_draw, odds_away):
    """
    Converts betting odds to probabilities (removing bookmaker margin).
    """
    if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
        return None, None, None
        
    p_h = 1 / odds_home
    p_d = 1 / odds_draw
    p_a = 1 / odds_away
    
    margin = p_h + p_d + p_a
    
    # Normalize
    return p_h / margin, p_d / margin, p_a / margin

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

def train_and_backtest():
    print("--- SofaScore Model Training & Backtesting (Multi-Model) ---")
    
    # 1. Load Data
    data_path = Path("data/sofascore/features_sofascore.csv")
    df = pd.read_csv(data_path)
    
    # Filter
    df = df[df['home_history_completeness'] > 0.8]
    df = df[df['away_history_completeness'] > 0.8]
    print(f"Dataset Size: {len(df)}")
    
    feature_cols = [c for c in df.columns if 'avg_' in c]
    
    # --- Feature Selection ---
    # Analyze correlations
    full_dataset = df[feature_cols].copy()
    full_dataset['target_diff'] = df['target_home_goals'] - df['target_away_goals']
    corr = full_dataset.corr()['target_diff'].sort_values(ascending=False)
    print("\nFeature Correlations with Goal Difference:")
    print(corr.head(5))
    print(corr.tail(5))
    
    # Select 'Offensive' features + Rating
    # We drop tackles/interceptions based on hypothesis they are noisy/defensive
    selected_features = [c for c in feature_cols if any(x in c for x in ['rating', 'goals', 'xg', 'shots', 'pass_acc', 'creativity'])]
    print(f"\nPerforming Feature Selection...")
    print(f"Original: {len(feature_cols)} -> Selected: {len(selected_features)}")
    print(f"Dropped: {set(feature_cols) - set(selected_features)}")
    
    full_X = df[selected_features]
    full_y_home = df['target_home_goals']
    full_y_away = df['target_away_goals']
    
    # 2. Setup Models
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import brier_score_loss
    
    models = {
        'SVR': lambda: make_pipeline(SimpleImputer(strategy='constant', fill_value=0), StandardScaler(), SVR(C=1.0, epsilon=0.2)),
        'Ridge': lambda: make_pipeline(SimpleImputer(strategy='constant', fill_value=0), StandardScaler(), Ridge(alpha=1.0)),
        'GBM': lambda: make_pipeline(SimpleImputer(strategy='constant', fill_value=0), GradientBoostingRegressor(learning_rate=0.05, max_depth=3, random_state=42))
    }
    
    strategies = {
        'Kelly (Frac)': {'threshold': 0.0, 'kelly': True} # Simple Kelly implementation
    }
    
    # 3. Backtest Loop
    summary = []
    events_dir = Path("data/sofascore/processed/events")
    start_index = 50
    bet_size = 10
    
    for model_name, model_factory in models.items():
        print(f"\nTraining {model_name}...")
        
        # Reset Strategies for this model
        strat_balances = {s: 1000.0 for s in strategies}
        strat_bets = {s: 0 for s in strategies}
        strat_wins = {s: 0 for s in strategies}
        
        model_home = model_factory()
        model_away = model_factory()
        
        preds_correct = 0
        preds_total = 0
        predictions_prob = []
        actuals = []
        
        # Iterative update (Simulating real season)
        # To save time in loop, we can batch re-train every N matches? 
        # But for correctness, let's keep retrain every match or every round. 
        # Refactor: To speed up, we RETRAIN every 10 matches. 
        
        for i in range(start_index, len(df)):
            # Retrain Periodically
            if i % 10 == 0 or i == start_index:
                X_train = full_X.iloc[:i]
                y_home_train = full_y_home.iloc[:i]
                y_away_train = full_y_away.iloc[:i]
                model_home.fit(X_train, y_home_train)
                model_away.fit(X_train, y_away_train)
            
            # Predict
            X_test = full_X.iloc[i:i+1]
            pred_h = max(0.05, model_home.predict(X_test)[0])
            pred_a = max(0.05, model_away.predict(X_test)[0])
            
            prob_h, prob_d, prob_a = calculate_match_outcome_probabilities(pred_h, pred_a)
            
            # Outcome
            actual_home = full_y_home.iloc[i]
            actual_away = full_y_away.iloc[i]
            match_id = df.iloc[i]['match_id']
            
            # Accuracy
            model_outcome = '1' if prob_h > prob_a and prob_h > prob_d else ('2' if prob_a > prob_h and prob_a > prob_d else 'X')
            actual_outcome = '1' if actual_home > actual_away else ('2' if actual_home < actual_away else 'X')
            if model_outcome == actual_outcome: preds_correct += 1
            preds_total += 1
            
            # Store for calibration metrics
            predictions_prob.append((prob_h, prob_d, prob_a))
            actuals.append(actual_outcome)
            
            # Betting
            odds_file = events_dir / str(match_id) / "odds.csv"
            odds_h, odds_d, odds_a = np.nan, np.nan, np.nan
            if odds_file.exists():
                try:
                    df_odds = pd.read_csv(odds_file)
                    row_1 = df_odds[df_odds['name'] == '1']
                    row_x = df_odds[df_odds['name'] == 'X']
                    row_2 = df_odds[df_odds['name'] == '2']
                    if not row_1.empty: odds_h = row_1.iloc[0]['odds']
                    if not row_x.empty: odds_d = row_x.iloc[0]['odds']
                    if not row_2.empty: odds_a = row_2.iloc[0]['odds']
                except: pass
                
            imp_h, imp_d, imp_a = get_implied_probabilities(odds_h, odds_d, odds_a)
            
            if imp_h:
                edges = [('1', prob_h - imp_h, odds_h, prob_h, actual_outcome=='1'), 
                         ('X', prob_d - imp_d, odds_d, prob_d, actual_outcome=='X'), 
                         ('2', prob_a - imp_a, odds_a, prob_a, actual_outcome=='2')]
                
                best_bet = max(edges, key=lambda x: x[1]) # Best Edge
                # best_bet: (Choice, Edge, Odds, Prob, Won?)
                
                for s_name, s_params in strategies.items():
                    stake = 0
                    if s_params['kelly']:
                        # Kelly = (bp - q) / b
                        # b = odds - 1
                        # p = prob
                        # q = 1 - p
                        b = best_bet[2] - 1
                        p = best_bet[3]
                        q = 1 - p
                        f = (b * p - q) / b
                        if f > 0:
                            stake = bet_size * (f * 0.5) # Half Kelly safe
                    else:
                        if best_bet[1] > s_params['threshold']:
                            stake = bet_size
                    
                    if stake > 0:
                        strat_bets[s_name] += 1
                        if best_bet[4]: # Won
                            strat_balances[s_name] += stake * (best_bet[2] - 1)
                            strat_wins[s_name] += 1
                        else:
                            strat_balances[s_name] -= stake
                            
        # Store Model Results
        accuracy = preds_correct / preds_total
        
        # Validation: Brier Score (Calibration)
        # Brier = Mean Squared Error of Probabilities
        brier_score = 0.0
        if predictions_prob: # predictions_prob should be populated in the loop
             # Calculate multi-class Brier manually or via sklearn if we stored y_true/y_prob
             from sklearn.metrics import brier_score_loss
             # Since brier_score_loss is for binary, we treat Home Win as the proxy for calibration quality
             # y_true_binary = [1 if o == '1' else 0 for o in actuals]
             # y_prob_home = [p[0] for p in predictions_prob]
             # brier = brier_score_loss(y_true_binary, y_prob_home)
             # Let's do a custom multi-class calculation for cleaner metric
             bs_sum = 0
             for (ph, pd_draw, pa), outcome in zip(predictions_prob, actuals):
                 o_vec = [1 if outcome=='1' else 0, 1 if outcome=='X' else 0, 1 if outcome=='2' else 0]
                 bs_sum += (ph - o_vec[0])**2 + (pd_draw - o_vec[1])**2 + (pa - o_vec[2])**2
             brier_score = bs_sum / len(predictions_prob)
             
        print(f"  Accuracy: {accuracy:.2%} | Brier Score: {brier_score:.4f}")
        
        for s, bal in strat_balances.items():
            roi = (bal - 1000) / (strat_bets[s] * bet_size) if strat_bets[s] > 0 else 0
            summary.append({
                'Model': model_name,
                'Strategy': s,
                'Accuracy': accuracy,
                'Brier': brier_score,
                'Bets': strat_bets[s],
                'WinRate': strat_wins[s] / strat_bets[s] if strat_bets[s] else 0,
                'Balance': bal,
                'ROI': roi
            })

    # 4. Print Summary Table
    print("\n--- COMPARATIVE RESULTS ---")
    df_summary = pd.DataFrame(summary)
    # Reorder columns
    cols = ['Model', 'Strategy', 'Accuracy', 'Brier', 'Bets', 'WinRate', 'ROI', 'Balance']
    df_summary = df_summary[cols].sort_values('Balance', ascending=False)
    
    print(df_summary.to_string(index=False, float_format="%.2f"))
    df_summary.to_csv("data/sofascore/comparison_results.csv", index=False)

if __name__ == "__main__":
    train_and_backtest()

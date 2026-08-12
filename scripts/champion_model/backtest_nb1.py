import os
import sys
import numpy as np
import pandas as pd
from dynamic_elo import DynamicEloEngine

def run_walkforward_backtest(csv_path, k_factor=20.0, home_advantage=60.0, season_regression=0.85, draw_param=0.26):
    """
    Executes a strict chronological walk-forward backtest over all NB I matches.
    Guarantees zero data leakage: match i is predicted using prior ratings ONLY.
    
    Parameters:
    - csv_path (str): Path to canonical matches CSV.
    - k_factor (float): K-factor weight for rating update.
    - home_advantage (float): Elo points bonus for home team.
    - season_regression (float): Off-season mean regression factor (0.0 to 1.0).
    - draw_param (float): Peak draw probability parameter.
    
    Returns:
    - metrics (dict): Log Loss, Brier Score, Accuracy.
    - season_df (DataFrame): Per-season metrics breakdown.
    - predictions_df (DataFrame): Match-level prediction log.
    """
    df = pd.read_csv(csv_path)
    
    # Sort chronologically
    df = df.sort_values(by=['season_id', 'matchday', 'date', 'match_id']).reset_index(drop=True)
    
    engine = DynamicEloEngine(
        k_factor=k_factor,
        home_advantage=home_advantage,
        season_regression=season_regression,
        draw_param=draw_param
    )
    
    records = []
    current_season = None
    
    for idx, row in df.iterrows():
        season_id = row['season_id']
        matchday = row['matchday']
        date_str = row['date']
        home_team = row['home_team']
        away_team = row['away_team']
        result = row['result'] # 'H', 'D', or 'A'
        home_score = row['home_score']
        away_score = row['away_score']
        
        # Off-season boundary check
        if current_season is not None and season_id != current_season:
            engine.apply_offseason_regression()
            
        current_season = season_id
        
        # 1. PRIOR STATE (Before match)
        r_home_prior = engine.get_rating(home_team)
        r_away_prior = engine.get_rating(away_team)
        
        # 2. PREDICT PROBABILITIES
        p_home, p_draw, p_away = engine.predict_proba(home_team, away_team)
        
        # 3. EVALUATE PREDICTION
        if result == 'H':
            p_actual = p_home
            y_vector = (1.0, 0.0, 0.0)
        elif result == 'D':
            p_actual = p_draw
            y_vector = (0.0, 1.0, 0.0)
        elif result == 'A':
            p_actual = p_away
            y_vector = (0.0, 0.0, 1.0)
        else:
            raise ValueError(f"Invalid result '{result}' at row {idx}")
            
        # Log Loss (Negative Log Likelihood)
        log_loss_i = -np.log(max(1e-15, p_actual))
        
        # Brier Score
        brier_i = (p_home - y_vector[0])**2 + (p_draw - y_vector[1])**2 + (p_away - y_vector[2])**2
        
        # Predicted Outcome (argmax)
        probas = {'H': p_home, 'D': p_draw, 'A': p_away}
        pred_outcome = max(probas, key=probas.get)
        is_correct = 1 if pred_outcome == result else 0
        
        # 4. POST-MATCH RATING UPDATE
        r_home_post, r_away_post, delta = engine.update_ratings(home_team, away_team, result)
        
        records.append({
            'season_id': season_id,
            'matchday': matchday,
            'date': date_str,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'actual_result': result,
            'r_home_prior': round(r_home_prior, 2),
            'r_away_prior': round(r_away_prior, 2),
            'p_home': round(p_home, 4),
            'p_draw': round(p_draw, 4),
            'p_away': round(p_away, 4),
            'pred_outcome': pred_outcome,
            'is_correct': is_correct,
            'log_loss': log_loss_i,
            'brier_score': brier_i,
            'r_home_post': round(r_home_post, 2),
            'r_away_post': round(r_away_post, 2),
            'rating_delta': round(delta, 2)
        })
        
    preds_df = pd.DataFrame(records)
    
    # Overall summary metrics
    overall_log_loss = float(preds_df['log_loss'].mean())
    overall_brier = float(preds_df['brier_score'].mean())
    overall_acc = float(preds_df['is_correct'].mean())
    
    metrics = {
        'total_matches': len(preds_df),
        'log_loss': overall_log_loss,
        'brier_score': overall_brier,
        'accuracy': overall_acc,
        'k_factor': k_factor,
        'home_advantage': home_advantage,
        'season_regression': season_regression,
        'draw_param': draw_param
    }
    
    # Per-season metrics breakdown
    season_grp = preds_df.groupby('season_id').agg(
        matches=('is_correct', 'count'),
        log_loss=('log_loss', 'mean'),
        brier_score=('brier_score', 'mean'),
        accuracy=('is_correct', 'mean')
    ).reset_index()
    
    return metrics, season_grp, preds_df

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")
    metrics, season_df, preds_df = run_walkforward_backtest(csv_path)
    
    print("=== BASELINE WALK-FORWARD BACKTEST RESULTS ===")
    print(f"Total Matches: {metrics['total_matches']}")
    print(f"Log Loss:      {metrics['log_loss']:.4f}")
    print(f"Brier Score:   {metrics['brier_score']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']*100:.2f}%")
    print("\n--- Per Season Breakdown ---")
    print(season_df.to_string(index=False))

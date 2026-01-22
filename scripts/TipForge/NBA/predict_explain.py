"""
NBA Prediction & Explanation & Telegram Sender
Based on run_predictions.py logic, but sends explanatory messages to Telegram.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import json
import joblib
import time
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# PATH SETUP & IMPORTS
# ============================================================================

# Ensure we can import local modules and parent modules
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PARENT_DIR.parent.parent # Assuming path structure: TSDP/scripts/TipForge/NBA -> TSDP

# 1. Add 'modules' for tx_module (used in odds fetching if needed)
sys.path.append(str(ROOT_DIR / "modules"))

# 2. Add 'TipForge' (parent) for telegram.py
sys.path.append(str(PARENT_DIR))

# 3. Ensure current dir is in path for nba_api_module
sys.path.append(str(SCRIPT_DIR))

load_dotenv()

# Import helpers
import nba_api_module as nbam
from nba_api.stats.endpoints import scoreboardv2
import telegram  # from telegram.py in parent dir

# ============================================================================
# COPIED / ADAPTED HELPER FUNCTIONS
# ============================================================================

team_id_dict = nbam.TEAM_IDS

def get_upcoming_games(days_ahead=3):
    """Lekéri a következő N napra tervezett meccseket - RETRY logikával."""
    upcoming = []
    teams_seen = set()
    today = datetime.now()
    
    for day_offset in range(days_ahead):
        check_date = today + pd.Timedelta(days=day_offset)
        date_str = check_date.strftime('%Y-%m-%d')
        
        max_retries = 1
        for attempt in range(max_retries):
            try:
                print(f"\nEllenőrzés: {date_str}")
                scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=30)
                games = scoreboard.get_data_frames()[0]
                
                if len(games) == 0:
                    break
                
                for _, game in games.iterrows():
                    game_id = str(game['GAME_ID'])
                    home_team_id = game['HOME_TEAM_ID']
                    away_team_id = game['VISITOR_TEAM_ID']
                    
                    if home_team_id not in teams_seen and away_team_id not in teams_seen:
                        upcoming.append({
                            'game_id': game_id,
                            'game_date': date_str,
                            'home_team_id': home_team_id,
                            'away_team_id': away_team_id,
                            'home_team': team_id_dict[home_team_id],
                            'away_team': team_id_dict[away_team_id]
                        })
                        teams_seen.add(home_team_id)
                        teams_seen.add(away_team_id)
                        print(f"  ✓ {team_id_dict[away_team_id]} @ {team_id_dict[home_team_id]}")
                
                time.sleep(3)
                break
            except Exception as e:
                print(f"  Hiba {date_str} lekérésekor: {e}")
                time.sleep(5)
                continue
    
    return pd.DataFrame(upcoming)

def create_pregame_features(game_id, season, game_date, team_ids):
    """Egy adott meccsre létrehozza az összes pregame feature-t."""
    features = {}
    try:
        pregame = nbam.extract_pregame(game_id, season, game_date, team_ids)
        features.update(pregame)
        time.sleep(0.5)
        
        injury = nbam.extract_injury(None)
        if injury:
            injury.pop('home_missing_starters', None)
            injury.pop('away_missing_starters', None)
            features.update(injury)
        
        advanced = nbam.extract_advanced_stats(game_id, game_date, season, home_id=team_ids[0], away_id=team_ids[1])
        features.update(advanced)
        
        form = nbam.extract_form(game_id, season, game_date, home_id=team_ids[0], away_id=team_ids[1])
        features.update(form)
        
        return features
    except Exception as e:
        print(f"Hiba a pregame features létrehozásakor ({game_id}): {e}")
        return None

def calculate_differential_features(features_dict):
    """Differenciális feature-ök hozzáadása."""
    df = pd.DataFrame([features_dict])
    
    # Basic diffs
    cols_to_diff = [
        ('ORtg', 'home_ORtg', 'away_ORtg'),
        ('DRtg', 'home_DRtg', 'away_DRtg'),
        ('NET_rtg', 'home_NET_rtg', 'away_NET_rtg'),
        ('PACE', 'home_PACE', 'away_PACE'),
        ('TS', 'home_TS%', 'away_TS%'),
        ('EFG', 'home_EFG%', 'away_EFG%'),
        ('AST_ratio', 'home_AST_ratio', 'away_AST_ratio'),
        ('OREB', 'home_OREB%', 'away_OREB%'),
        ('turnover', 'home_turnover_ratio', 'away_turnover_ratio'),
        ('starter_PER', 'home_starter_avg_PER', 'away_starter_avg_PER'),
        ('bench_PER', 'home_bench_avg_PER', 'away_bench_avg_PER'),
        ('star_usage', 'home_star_usage', 'away_star_usage'),
        ('avg_TS', 'home_avg_TS', 'away_avg_TS'),
        ('top3_points', 'home_top3_points_avg', 'away_top3_points_avg'),
        ('rest_days', 'home_rest_days', 'away_rest_days'),
        ('recent_form10', 'home_recent_form10', 'away_recent_form10'),
        ('recent_form5', 'home_recent_form5', 'away_recent_form5'),
        ('recent_form3', 'home_recent_form3', 'away_recent_form3'),
        ('injury_count', 'home_injury_count', 'away_injury_count'),
    ]
    
    for name, h_col, a_col in cols_to_diff:
        if h_col in df.columns and a_col in df.columns:
            df[f'{name}_diff'] = df[h_col] - df[a_col]
        else:
            df[f'{name}_diff'] = 0

    if 'away_is_back_to_back' in df.columns and 'home_is_back_to_back' in df.columns:
        df['b2b_advantage'] = df['away_is_back_to_back'].astype(int) - df['home_is_back_to_back'].astype(int)
    else:
        df['b2b_advantage'] = 0
            
    return df.iloc[0].to_dict()

def make_predictions(features_df):
    """Predictions minden modellel."""
    # Ensure current dir is correct for loading models
    original_cwd = os.getcwd()
    os.chdir(SCRIPT_DIR)
    
    try:
        meta_cols = ['game_id', 'game_date', 'home_team', 'away_team']
        meta_df = features_df[meta_cols].copy()
        
        with open('models/feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
        
        # Ensure all columns exist
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
                
        X = features_df[feature_cols]
        
        # Load Scaler
        scaler = joblib.load('models/scaler.joblib')
        X_scaled = scaler.transform(X)
        
        predictions = meta_df.copy()
        
        # Models
        # 1. Random Forest
        pca_rf = joblib.load('models/rf_pca_transformer.joblib')
        model_rf = joblib.load('models/rf_pca_model.joblib')
        X_pca_rf = pca_rf.transform(X_scaled)
        predictions['rf_pca_prob_home'] = model_rf.predict_proba(X_pca_rf)[:, 1]
        
        # 2. XGBoost Shallow
        pca_xgb_s = joblib.load('models/xgb_shallow_pca_transformer.joblib')
        model_xgb_s = joblib.load('models/xgb_shallow_model.joblib')
        X_pca_xgb_s = pca_xgb_s.transform(X_scaled)
        predictions['xgb_shallow_prob_home'] = model_xgb_s.predict_proba(X_pca_xgb_s)[:, 1]
        
        # 3. XGBoost Tuned
        pca_xgb_t = joblib.load('models/xgb_tuned_pca_transformer.joblib')
        model_xgb_t = joblib.load('models/xgb_tuned_model.joblib')
        X_pca_xgb_t = pca_xgb_t.transform(X_scaled)
        predictions['xgb_tuned_prob_home'] = model_xgb_t.predict_proba(X_pca_xgb_t)[:, 1]
        
        # Ensemble
        predictions['ensemble_prob_home'] = predictions[
            ['rf_pca_prob_home', 'xgb_shallow_prob_home', 'xgb_tuned_prob_home']
        ].mean(axis=1)
        
        return predictions
    finally:
        os.chdir(original_cwd)

# ============================================================================
# EXPLANATION GENERATION
# ============================================================================

def generate_reasoning_text(row, features_dict):
    """
    Generates a short explanation string based on the differential features.
    
    Args:
        row: row from predictions DataFrame (has teams, probs)
        features_dict: dictionary of features for this game
    """
    prob_home = row['ensemble_prob_home']
    home_team = row['home_team']
    away_team = row['away_team']
    
    if prob_home >= 0.5:
        favorite = home_team
        prob = prob_home
        is_home_fav = True
    else:
        favorite = away_team
        prob = 1 - prob_home
        is_home_fav = False
        
    reasons = []
    
    # 1. Net Rating
    net_rtg_diff = features_dict.get('NET_rtg_diff', 0)
    # If home fav and net_rtg_diff > 0 -> Supports Home
    # If away fav and net_rtg_diff < 0 -> Supports Away (Home - Away < 0 => Away better)
    
    if is_home_fav and net_rtg_diff > 1.0:
        reasons.append(f"Better NetRtg (+{net_rtg_diff:.1f})")
    elif not is_home_fav and net_rtg_diff < -1.0:
        reasons.append(f"Better NetRtg (Has +{abs(net_rtg_diff):.1f} adv)")
        
    # 2. Form (Last 10)
    form_diff = features_dict.get('recent_form10_diff', 0)
    if is_home_fav and form_diff > 0:
        reasons.append(f"Better Form (+{int(form_diff)} wins/10)")
    elif not is_home_fav and form_diff < 0:
        reasons.append(f"Better Form (+{int(abs(form_diff))} wins/10)")
        
    # 3. Injuries
    # injury_count_diff = home_inj - away_inj
    # If home fav and injury_count_diff < 0 (Home has FEWER injuries) -> Support
    inj_diff = features_dict.get('injury_count_diff', 0)
    if is_home_fav and inj_diff < 0:
        reasons.append(f"Fewer Injuries ({int(abs(inj_diff))} less)")
    elif not is_home_fav and inj_diff > 0:
        reasons.append(f"Fewer Injuries ({int(inj_diff)} less)")
        
    # 4. Rest / B2B
    b2b = features_dict.get('b2b_advantage', 0) # +1 if Home rested vs Away B2B
    if is_home_fav and b2b > 0:
        reasons.append("Rest Advantage (Opponent B2B)")
    elif not is_home_fav and b2b < 0:
        reasons.append("Rest Advantage (Opponent B2B)")
        
    # Start composing message
    reason_str = ", ".join(reasons)
    if not reason_str:
        reason_str = "Based on aggregate team stats"
        
    return f"🏆 {favorite} ({prob:.1%})\n   👉 {reason_str}"

# ============================================================================
# MAIN
# ============================================================================

def run_main():
    print("🏀 NBA PREDICT & EXPLAIN")
    
    # 1. Get Games
    upcoming = get_upcoming_games(days_ahead=2) # Only next 2 days to keep message short
    if upcoming.empty:
        print("Nincs közelgő meccs.")
        return

    # 2. Features
    all_features = []
    print(f"Features generálása {len(upcoming)} meccshez...")
    
    for idx, game in upcoming.iterrows():
        try:
            feats = create_pregame_features(
                game['game_id'], '2025-26', game['game_date'],
                [game['home_team_id'], game['away_team_id']]
            )
            if feats:
                feats = calculate_differential_features(feats)
                feats['game_id'] = game['game_id']
                feats['game_date'] = game['game_date']
                feats['home_team'] = game['home_team']
                feats['away_team'] = game['away_team']
                all_features.append(feats)
        except Exception as e:
            print(f"Skipping {game['game_id']}: {e}")
            
    if not all_features:
        print("No features generated.")
        return
        
    features_df = pd.DataFrame(all_features)
    
    # 3. Predictions
    print("Modellek futtatása...")
    preds = make_predictions(features_df)
    
    # 4. Try to get Odds (Optional)
    # We won't break if this fails, but it helps context
    try:
        from tx_module import get_league_odds
        # Simple fetch if possible, else skip
        # ... (skipping complex merging to keep script robust and fast, focusing on predictions as requested)
    except:
        pass

    # 5. Build Telegram Message
    msg_lines = [f"🏀 NBA Predictions ({datetime.now().strftime('%Y-%m-%d')})", ""]
    
    for _, row in preds.iterrows():
        game_id = row['game_id']
        home = row['home_team']
        away = row['away_team']
        
        # Get feature dict for reasoning
        feat_dict = features_df[features_df['game_id'] == game_id].iloc[0].to_dict()
        
        explanation = generate_reasoning_text(row, feat_dict)
        
        line = f"MATCH: {away} @ {home}\n{explanation}\n"
        msg_lines.append(line)
        
    final_msg = "\n".join(msg_lines)
    print("\n--- Generated Message ---")
    print(final_msg)
    
    # 6. Send
    print("\nSending to Telegram...")
    telegram.send_to_telegram(final_msg, to="owner")
    
if __name__ == "__main__":
    run_main()

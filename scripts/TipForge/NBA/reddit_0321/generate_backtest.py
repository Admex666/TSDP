import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
import scipy.stats
import os

def parse_matchup(matchup):
    if ' @ ' in str(matchup):
        parts = matchup.split(' @ ')
        return parts[0].strip(), parts[1].strip()
    elif ' vs. ' in str(matchup):
        parts = matchup.split(' vs. ')
        return parts[1].strip(), parts[0].strip() # 'ORL vs. CHA' -> ORL is Home, CHA is Away. Returns (Away, Home) -> (CHA, ORL)
    return "UNK", "UNK"

def main():
    data_path = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\processed_data.csv"
    games_path = r"E:\Data\TSDP\scripts\TipForge\NBA\data\games_2024_25.csv"
    output_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321"
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    df.sort_values('GAME_DATE', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    games = pd.read_csv(games_path)
    games['GAME_ID'] = games['GAME_ID'].astype(str).str.zfill(10)
    
    # 2. Features and Target
    exclude_cols = ['game_id', 'GAME_ID', 'GAME_DATE', 'home_win', 'home_margin', 'odds_home', 'odds_away']
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['home_margin']
    
    # 3. Model & Out-of-sample predictions
    model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=4, 
        subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', 
        random_state=42
    )
    
    print("Running 5-Fold Cross Validation to generate out-of-sample predictions...")
    kf = KFold(n_splits=5, shuffle=False)
    pred_margin = cross_val_predict(model, X, y, cv=kf)
    df['pred_margin'] = pred_margin
    
    # Fit globally just to get a stable sigma for residuals
    model.fit(X, y)
    train_preds = model.predict(X)
    sigma = np.std(y - train_preds)
    
    # 4. Probabilities & Betting Logic
    df['pred_prob_home'] = scipy.stats.norm.cdf(df['pred_margin'] / sigma)
    
    df['implied_prob_home'] = 1 / df['odds_home']
    df['implied_prob_away'] = 1 / df['odds_away']
    
    margin_edge = 0.02
    df['bet_home'] = df['pred_prob_home'] > (df['implied_prob_home'] + margin_edge)
    df['bet_away'] = (1 - df['pred_prob_home']) > (df['implied_prob_away'] + margin_edge)
    
    # Calculate Profit
    df['profit'] = 0.0
    df['bet_placed'] = 'No Bet'
    df['won_ML'] = 0
    
    for idx, row in df.iterrows():
        if row['bet_home']:
            df.at[idx, 'bet_placed'] = 'Home'
            if row['home_win'] == 1:
                df.at[idx, 'profit'] = row['odds_home'] - 1
                df.at[idx, 'won_ML'] = 1
            else:
                df.at[idx, 'profit'] = -1.0
        elif row['bet_away']:
            df.at[idx, 'bet_placed'] = 'Away'
            if row['home_win'] == 0:
                df.at[idx, 'profit'] = row['odds_away'] - 1
                df.at[idx, 'won_ML'] = 1
            else:
                df.at[idx, 'profit'] = -1.0
                
    # 5. Bring in Team Names
    matchup_dict = games.set_index('GAME_ID')['MATCHUP'].to_dict()
    
    df['Away_Team'] = ''
    df['Home_Team'] = ''
    
    df['GAME_ID'] = df['GAME_ID'].astype(str).str.zfill(10)
    for idx, row in df.iterrows():
        matchup = matchup_dict.get(row['GAME_ID'], '')
        away, home = parse_matchup(matchup)
        df.at[idx, 'Away_Team'] = away
        df.at[idx, 'Home_Team'] = home
        
    print(f"Generated backtest for {len(df)} games.")
    bets_placed_mask = df['bet_placed'] != 'No Bet'
    total_bets = bets_placed_mask.sum()
    won_bets = df[bets_placed_mask]['won_ML'].sum()
    print(f"Total Bets: {total_bets}")
    print(f"Hit Rate: {(won_bets/total_bets*100) if total_bets > 0 else 0:.1f}%")
    print(f"Total Profit: {df['profit'].sum():.2f} U")
    
    # Save
    out_path = os.path.join(output_dir, "full_backtest_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()

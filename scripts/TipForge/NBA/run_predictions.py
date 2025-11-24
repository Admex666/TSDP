"""
NBA Betting Predictions - Main Runner Script
Futtatja a teljes prediction pipeline-t és emailt küld az eredményekről
"""

import pandas as pd
import numpy as np
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys
import json
import joblib
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Import saját modulok
import nba_api_module as nbam
from nba_api.stats.endpoints import scoreboardv2

import os 
from pathlib import Path
import sys

wd = Path(os.getcwd())
root = wd.parents[2]
sys.path.append(str(root / "modules"))


# ============================================================================
# SEGÉDFÜGGVÉNYEK (ml_pred.ipynb-ből)
# ============================================================================

team_id_dict = nbam.TEAM_IDS
team_abr_dict = nbam.TEAM_ABBREVIATIONS

def get_upcoming_games(days_ahead=3):
    """Lekéri a következő N napra tervezett meccseket - RETRY logikával."""
    upcoming = []
    teams_seen = set()
    today = datetime.now()
    
    for day_offset in range(days_ahead):
        check_date = today + pd.Timedelta(days=day_offset)
        date_str = check_date.strftime('%Y-%m-%d')
        
        # RETRY LOGIKA - 3 próbálkozás
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"\nEllenőrzés: {date_str} (próbálkozás {attempt + 1}/{max_retries})")
                
                scoreboard = scoreboardv2.ScoreboardV2(
                    game_date=date_str,
                    timeout=90  # Növelt timeout
                )
                games = scoreboard.get_data_frames()[0]
                
                if len(games) == 0:
                    print(f"  Nincs meccs ezen a napon")
                    break  # Kilépünk a retry loop-ból
                
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
                
                time.sleep(3)  # Hosszabb várakozás
                break  # Sikeres, kilépünk a retry loop-ból
                
            except Exception as e:
                print(f"  Hiba {date_str} lekérésekor (próbálkozás {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10, 20, 30 mp
                    print(f"  Várakozás {wait_time} másodperc...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Sikertelen {max_retries} próbálkozás után")
                continue
    
    return pd.DataFrame(upcoming)


def create_pregame_features(game_id, season, game_date, team_ids):
    """Egy adott meccsre létrehozza az összes pregame feature-t."""
    features = {}
    
    try:
        pregame = nbam.extract_pregame(game_id, season, game_date, team_ids)
        features.update(pregame)
        time.sleep(1)
        
        injury = nbam.extract_injury(None)
        injury.pop('home_missing_starters', None)
        injury.pop('away_missing_starters', None)
        features.update(injury)
        time.sleep(0.01)
        
        advanced = nbam.extract_advanced_stats(game_id, game_date, season, home_id=team_ids[0], away_id=team_ids[1])
        features.update(advanced)
        time.sleep(1)
        
        form = nbam.extract_form(game_id, season, game_date, home_id=team_ids[0], away_id=team_ids[1])
        features.update(form)
        
        return features
    except Exception as e:
        print(f"Hiba a pregame features létrehozásakor: {e}")
        return None


def calculate_differential_features(features_dict):
    """Differenciális feature-ök hozzáadása."""
    df = pd.DataFrame([features_dict])
    
    df['ORtg_diff'] = df['home_ORtg'] - df['away_ORtg']
    df['DRtg_diff'] = df['home_DRtg'] - df['away_DRtg']
    df['NET_rtg_diff'] = df['home_NET_rtg'] - df['away_NET_rtg']
    df['PACE_diff'] = df['home_PACE'] - df['away_PACE']
    df['TS_diff'] = df['home_TS%'] - df['away_TS%']
    df['EFG_diff'] = df['home_EFG%'] - df['away_EFG%']
    df['AST_ratio_diff'] = df['home_AST_ratio'] - df['away_AST_ratio']
    df['OREB_diff'] = df['home_OREB%'] - df['away_OREB%']
    df['turnover_diff'] = df['home_turnover_ratio'] - df['away_turnover_ratio']
    df['starter_PER_diff'] = df['home_starter_avg_PER'] - df['away_starter_avg_PER']
    df['bench_PER_diff'] = df['home_bench_avg_PER'] - df['away_bench_avg_PER']
    df['star_usage_diff'] = df['home_star_usage'] - df['away_star_usage']
    df['avg_TS_diff'] = df['home_avg_TS'] - df['away_avg_TS']
    df['top3_points_diff'] = df['home_top3_points_avg'] - df['away_top3_points_avg']
    df['rest_days_diff'] = df['home_rest_days'] - df['away_rest_days']
    df['recent_form10_diff'] = df['home_recent_form10'] - df['away_recent_form10']
    df['recent_form5_diff'] = df['home_recent_form5'] - df['away_recent_form5']
    df['recent_form3_diff'] = df['home_recent_form3'] - df['away_recent_form3']
    df['injury_count_diff'] = df['home_injury_count'] - df['away_injury_count']
    df['b2b_advantage'] = df['away_is_back_to_back'].astype(int) - df['home_is_back_to_back'].astype(int)
    
    return df.iloc[0].to_dict()


def align_features_to_model(features_dict, feature_columns_path='models/feature_columns.json'):
    """Biztosítja, hogy a feature-ök pontosan egyezzenek a modell által elvárt feature listával."""
    with open(feature_columns_path, 'r') as f:
        expected_features = json.load(f)
    
    missing_features = [f for f in expected_features if f not in features_dict]
    if missing_features:
        for feat in missing_features:
            features_dict[feat] = 0
    
    aligned_features = {feat: features_dict[feat] for feat in expected_features}
    return pd.DataFrame([aligned_features])


# ============================================================================
# FŐ PIPELINE FUNKCIÓK
# ============================================================================

def run_full_prediction_pipeline():
    """Teljes prediction pipeline futtatása."""
    print("="*80)
    print("NBA BETTING PREDICTIONS - PIPELINE START")
    print("="*80)
    
    # 1. Upcoming games lekérése
    print("\n[1/4] Upcoming games lekérése...")
    upcoming_df = get_upcoming_games(days_ahead=3)
    
    if len(upcoming_df) == 0:
        print("Nincs közelgő meccs!")
        return None, None, None
    
    # 2. Features gyűjtése
    print(f"\n[2/4] Features gyűjtése {len(upcoming_df)} meccshez...")
    all_features = []
    
    for idx, game in upcoming_df.iterrows():
        try:
            features = create_pregame_features(
                game['game_id'], '2025-26', game['game_date'],
                [game['home_team_id'], game['away_team_id']]
            )
            
            if features is None:
                continue
            
            features = calculate_differential_features(features)
            features['game_id'] = game['game_id']
            features['game_date'] = game['game_date']
            features['home_team'] = game['home_team']
            features['away_team'] = game['away_team']
            
            all_features.append(features)
            time.sleep(2)
        except Exception as e:
            print(f"Hiba a {game['game_id']} feldolgozásakor: {e}")
            continue
    
    if len(all_features) == 0:
        print("Nem sikerült feature-t gyűjteni!")
        return None, None, None
    
    features_df = pd.DataFrame(all_features)
    
    # 3. Predictions
    print(f"\n[3/4] Predictions készítése...")
    predictions = make_predictions(features_df)
    
    # 4. Odds és value bets
    print(f"\n[4/4] Odds lekérése és value bets számítása...")
    preds_w_odds = fetch_odds_and_calculate_value(predictions)
    
    # 5. Paperbet update
    print(f"\n[5/5] Paperbets frissítése...")
    paperbet_results = update_paperbets(preds_w_odds)
    
    return preds_w_odds, paperbet_results, features_df


def make_predictions(features_df):
    """Predictions minden modellel."""
    meta_cols = ['game_id', 'game_date', 'home_team', 'away_team']
    meta_df = features_df[meta_cols].copy()
    
    with open('models/feature_columns.json', 'r') as f:
        feature_cols = json.load(f)
    
    X = features_df[feature_cols]
    scaler = joblib.load('models/scaler.joblib')
    X_scaled = scaler.transform(X)
    
    predictions = meta_df.copy()
    
    # Random Forest
    pca_rf = joblib.load('models/rf_pca_transformer.joblib')
    model_rf = joblib.load('models/rf_pca_model.joblib')
    X_pca_rf = pca_rf.transform(X_scaled)
    predictions['rf_pca_prob_home'] = model_rf.predict_proba(X_pca_rf)[:, 1]
    
    # XGBoost Shallow
    pca_xgb_shallow = joblib.load('models/xgb_shallow_pca_transformer.joblib')
    model_xgb_shallow = joblib.load('models/xgb_shallow_model.joblib')
    X_pca_xgb_shallow = pca_xgb_shallow.transform(X_scaled)
    predictions['xgb_shallow_prob_home'] = model_xgb_shallow.predict_proba(X_pca_xgb_shallow)[:, 1]
    
    # XGBoost Tuned
    pca_xgb_tuned = joblib.load('models/xgb_tuned_pca_transformer.joblib')
    model_xgb_tuned = joblib.load('models/xgb_tuned_model.joblib')
    X_pca_xgb_tuned = pca_xgb_tuned.transform(X_scaled)
    predictions['xgb_tuned_prob_home'] = model_xgb_tuned.predict_proba(X_pca_xgb_tuned)[:, 1]
    
    # Ensemble
    predictions['ensemble_prob_home'] = predictions[
        ['rf_pca_prob_home', 'xgb_shallow_prob_home', 'xgb_tuned_prob_home']
    ].mean(axis=1)
    
    # Away probs
    for col in ['rf_pca', 'xgb_shallow', 'xgb_tuned', 'ensemble']:
        predictions[f'{col}_prob_away'] = 1 - predictions[f'{col}_prob_home']
    
    return predictions


def fetch_odds_and_calculate_value(predictions):
    """Odds lekérése és value számítás."""
    from tx_module import get_league_odds
    
    teams_tx_map_rev = {
        "Atlanta": "Atlanta Hawks", "Miami": "Miami Heat", "Orlando": "Orlando Magic",
        "New York": "New York Knicks", "Milwaukee": "Milwaukee Bucks", "Phoenix": "Phoenix Suns",
        "Dallas": "Dallas Mavericks", "Minnesota": "Minnesota Timberwolves", 
        "Toronto": "Toronto Raptors", "Cleveland": "Cleveland Cavaliers",
        "Brooklyn": "Brooklyn Nets", "LA Lakers": "Los Angeles Lakers",
        "Houston": "Houston Rockets", "New Orleans": "New Orleans Pelicans",
        "Golden State": "Golden State Warriors", "Boston": "Boston Celtics",
        "Denver": "Denver Nuggets", "San Antonio": "San Antonio Spurs",
        "LA Clippers": "LA Clippers", "Chicago": "Chicago Bulls",
        "Indiana": "Indiana Pacers", "Portland": "Portland Trail Blazers",
        "Philadelphia": "Philadelphia 76ers", "Oklahoma City": "Oklahoma City Thunder",
        "Utah": "Utah Jazz", "Sacramento": "Sacramento Kings",
        "Detroit": "Detroit Pistons", "Memphis": "Memphis Grizzlies",
        "Washington": "Washington Wizards"
    }
    
    url = "https://www.tippmixpro.hu/hu/fogadas/i/bajnoksag-lokacio/kosarlabda/8/usa/229/nba/274663790763708416"
    odds = get_league_odds(url)
    odds['home_team_api'] = odds['home_team'].map(teams_tx_map_rev)
    odds['away_team_api'] = odds['away_team'].map(teams_tx_map_rev)
    
    preds_w_odds = pd.merge(
        predictions,
        odds[['home_team_api', 'away_team_api', 'home_odds', 'away_odds']],
        left_on=['home_team', 'away_team'],
        right_on=['home_team_api', 'away_team_api'],
        how="inner"
    )
    
    preds_w_odds.drop(columns=['home_team_api', 'away_team_api'], inplace=True)
    preds_w_odds['home_implied'] = 1 / preds_w_odds['home_odds']
    preds_w_odds['away_implied'] = 1 / preds_w_odds['away_odds']
    
    # Mentés
    path = "data/value_bets_2025_26.csv"
    prev_csv = pd.read_csv(path, dtype={"game_id": str})
    new_csv = pd.concat([prev_csv, preds_w_odds], ignore_index=True).drop_duplicates(subset="game_id", keep="first")
    new_csv.to_csv(path, index=False)
    
    return preds_w_odds


def update_paperbets(preds_w_odds):
    """Paperbets frissítése és értékelése."""
    path_pbets = "data/paperbets.csv"
    vbets = pd.read_csv("data/value_bets_2025_26.csv", dtype={"game_id": str})
    
    # Új betek létrehozása
    bets_list = []
    models = ['rf_pca', 'xgb_shallow', 'xgb_tuned', 'ensemble']
    
    for i, game in vbets.iterrows():
        for model_name in models:
            prob_home_mod = game[f"{model_name}_prob_home"]
            prob_away_mod = game[f"{model_name}_prob_away"]
            prob_home_impl = game["home_implied"]
            prob_away_impl = game["away_implied"]
            odds_home = game['home_odds']
            odds_away = game['away_odds']
            
            if prob_home_mod > prob_home_impl:
                bet, prob, odds = "home", prob_home_mod, odds_home
            elif prob_away_mod > prob_away_impl:
                bet, prob, odds = "away", prob_away_mod, odds_away
            else:
                bet, prob, odds = "-", 0, 0
            
            value = (odds * prob) - 1
            stake = 1 if value > 0 else 0
            
            bets_list.append({
                'game_id': game['game_id'],
                'model': model_name,
                'strategy': 'fixed',
                'bet': bet,
                'prob': prob,
                'odds': odds,
                'value': value,
                'stake': stake,
                'WL': None,
                'profit': None
            })
    
    pbets = pd.DataFrame(bets_list)
    pbets_prev = pd.read_csv(path_pbets, dtype={'game_id': str})
    
    # Eredmények értékelése
    game_log = nbam.get_season_game_log("2025-26")
    for i, bet in pbets_prev.iterrows():
        if pd.notna(bet['WL']):
            continue
        
        game_id = bet['game_id']
        home_mask = (game_log.MATCHUP.str.contains(" vs. "))
        game_row = game_log[(game_log.GAME_ID == game_id) & home_mask]
        
        if len(game_row) == 0:
            continue
        
        home_win = game_row.iloc[0]['PLUS_MINUS'] > 0
        
        if bet['bet'] == "home":
            wl = "W" if home_win else "L"
        elif bet['bet'] == "away":
            wl = "W" if not home_win else "L"
        else:
            wl = "-"
        
        if bet['stake'] > 0:
            profit = (bet['odds'] * bet['stake'] - bet['stake']) if wl == "W" else -bet['stake']
        else:
            profit = 0
        
        pbets_prev.at[i, 'WL'] = wl
        pbets_prev.at[i, 'profit'] = profit
    
    # Összesítés
    res_list = []
    for model in models:
        subset = pbets_prev[(pbets_prev['model'] == model) & (pbets_prev['strategy'] == 'fixed')]
        total_bets = len(subset[subset['WL'].notna() & (subset['WL'] != '-')])
        total_profit = subset['profit'].sum()
        winrate = len(subset[subset['WL'] == 'W']) / total_bets * 100 if total_bets > 0 else 0
        roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
        
        res_list.append({
            'model': model,
            'total_bets': total_bets,
            'total_profit': total_profit,
            'winrate_%': winrate,
            'ROI_%': roi
        })
    
    results = pd.DataFrame(res_list)
    
    # Mentés
    pbets_new = pd.concat([pbets_prev, pbets], ignore_index=True).drop_duplicates(
        subset=['game_id', 'model', 'strategy'], keep='first'
    )
    pbets_new.to_csv(path_pbets, index=False)
    
    return results


# ============================================================================
# EMAIL GENERÁLÁS
# ============================================================================

def create_email_body(preds_w_odds, paperbet_results):
    """
    HTML email body generálása az eredményekkel.
    """
    today = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Value bets összegyűjtése
    value_bets_html = ""
    if preds_w_odds is not None and len(preds_w_odds) > 0:
        value_bets_html = "<h2>🎯 Mai Value Betek</h2><table border='1' style='border-collapse: collapse; width: 100%;'>"
        value_bets_html += "<tr style='background-color: #f2f2f2;'><th>Meccs</th><th>Bet</th><th>Odds</th><th>Model Prob</th><th>Value</th></tr>"
        
        for i, game in preds_w_odds.iterrows():
            # RF modell alapján
            if game['home_implied'] <= game['rf_pca_prob_home']:
                bet_team = game['home_team']
                odds = game['home_odds']
                prob = game['rf_pca_prob_home']
                value = (odds * prob) - 1
                value_bets_html += f"<tr><td>{game['away_team']} @ <b>{game['home_team']}</b></td><td><b>{bet_team}</b></td><td>{odds:.2f}</td><td>{prob:.1%}</td><td style='color: green;'>{value:.2%}</td></tr>"
            elif game['away_implied'] <= game['rf_pca_prob_away']:
                bet_team = game['away_team']
                odds = game['away_odds']
                prob = game['rf_pca_prob_away']
                value = (odds * prob) - 1
                value_bets_html += f"<tr><td><b>{game['away_team']}</b> @ {game['home_team']}</td><td><b>{bet_team}</b></td><td>{odds:.2f}</td><td>{prob:.1%}</td><td style='color: green;'>{value:.2%}</td></tr>"
        
        value_bets_html += "</table>"
    else:
        value_bets_html = "<h2>🎯 Mai Value Betek</h2><p>Nincs value bet ma.</p>"
    
    # Paperbet eredmények
    paperbet_html = ""
    if paperbet_results is not None and len(paperbet_results) > 0:
        paperbet_html = "<h2>📊 Összesített Paperbet Eredmények</h2>"
        paperbet_html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        paperbet_html += "<tr style='background-color: #f2f2f2;'><th>Model</th><th>Betek száma</th><th>Profit</th><th>Winrate</th><th>ROI</th></tr>"
        
        for _, row in paperbet_results.iterrows():
            profit_color = 'green' if row['total_profit'] > 0 else 'red'
            paperbet_html += f"<tr><td><b>{row['model']}</b></td><td>{row['total_bets']}</td><td style='color: {profit_color};'><b>{row['total_profit']:.2f}</b></td><td>{row['winrate_%']:.1f}%</td><td>{row['ROI_%']:.1f}%</td></tr>"
        
        paperbet_html += "</table>"
    else:
        paperbet_html = "<h2>📊 Összesített Paperbet Eredmények</h2><p>Nincs adat.</p>"
    
    # Teljes body összeállítása
    body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ margin: 20px 0; }}
            th {{ padding: 10px; text-align: left; }}
            td {{ padding: 8px; }}
        </style>
    </head>
    <body>
        <h1>🏀 NBA Betting Report - {today}</h1>
        
        {value_bets_html}
        
        <br>
        
        {paperbet_html}
        
        <br>
        <hr>
        <p style='color: gray; font-size: 12px;'>Automatikus riport a GitHub Actions által generálva</p>
    </body>
    </html>
    """
    
    return body


def send_email(subject, body, to_email):
    """
    Email küldése Gmail SMTP-n keresztül.
    """
    from_email = os.getenv('GMAIL_USER')
    password = os.getenv('GMAIL_APP_PASSWORD')
    
    if not from_email or not password:
        print("❌ GMAIL_USER vagy GMAIL_APP_PASSWORD nincs beállítva!")
        return False
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    
    html_part = MIMEText(body, 'html')
    msg.attach(html_part)
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, password)
            server.send_message(msg)
        print(f"✅ Email elküldve: {to_email}")
        return True
    except Exception as e:
        print(f"❌ Email küldési hiba: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🏀 NBA BETTING PREDICTIONS - DAILY RUN")
    print("="*80 + "\n")
    
    try:
        # Pipeline futtatása
        preds_w_odds, paperbet_results, features_df = run_full_prediction_pipeline()
        
        if preds_w_odds is None:
            print("\n⚠️ Nincs feldolgozható adat, email nem kerül kiküldésre.")
            sys.exit(0)
        
        # Email body generálása
        email_body = create_email_body(preds_w_odds, paperbet_results)
        
        # Email küldése
        TO_EMAIL = "adam.jakus99@gmail.com"
        send_email(
            subject=f"🏀 NBA Betting Report - {datetime.now().strftime('%Y-%m-%d')}",
            body=email_body,
            to_email=TO_EMAIL
        )
        
        print("\n" + "="*80)
        print("✅ PIPELINE BEFEJEZVE")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ KRITIKUS HIBA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
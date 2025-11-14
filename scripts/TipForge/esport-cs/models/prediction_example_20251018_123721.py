"""
CS:GO Betting Model - Prediction Example
Használat: python prediction_example_20251018_123721.py
"""

import pickle
import pandas as pd
import numpy as np

# ============================================================================
# 1. Modell és feature lista betöltése
# ============================================================================

with open('models/logistic_regression_csgo_20251018_123721.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/full_pipeline_20251018_123721.pkl', 'rb') as f:
    pipeline_data = pickle.load(f)

features = pipeline_data['features']

print("Modell betöltve!")
print(f"Feature-ök száma: {len(features)}")

# ============================================================================
# 2. Új meccs adatainak előkészítése
# ============================================================================

# PÉLDA: Új meccs data
new_match = {
    'home_last_3_winrate': 0.667,
    'away_last_3_winrate': 0.333,
    'home_current_rank': 5,
    'away_current_rank': 10,
    'rank_diff': -5,
    'home_odds': 1.75,
    'away_odds': 2.10,
    'home_implied_odds': 0.571,
    'away_implied_odds': 0.476,
    'odds_diff': -0.35,
    'implied_odds_diff': 0.095,
    'H2H_games': 5,
    'H2H_winrate_team1': 0.6,
    # ... többi feature (MIND kell ami a training-ben volt!)
}

# Dataframe-mé alakítás
new_match_df = pd.DataFrame([new_match])

# Feature sorrendbe rendezés (KRITIKUS!)
new_match_df = new_match_df[features]

# ============================================================================
# 3. Predikció
# ============================================================================

# Home win valószínűség
pred_home_win = model.predict_proba(new_match_df)[0, 1]
pred_away_win = 1 - pred_home_win

print(f"\nPredikció:")
print(f"  Home win prob: {pred_home_win:.2%}")
print(f"  Away win prob: {pred_away_win:.2%}")

# ============================================================================
# 4. Value bet kalkuláció
# ============================================================================

edge_home = pred_home_win - new_match['home_implied_odds']
edge_away = pred_away_win - new_match['away_implied_odds']

print(f"\nEdge:")
print(f"  Home edge: {edge_home:.4f} ({edge_home*100:.2f}%)")
print(f"  Away edge: {edge_away:.4f} ({edge_away*100:.2f}%)")

# Betting decision (0.02 edge threshold)
EDGE_THRESHOLD = 0.02

if edge_home > EDGE_THRESHOLD:
    print(f"\n✅ BET ON HOME! (Edge: {edge_home*100:.2f}%)")
    print(f"   Odds: {new_match['home_odds']}")
    print(f"   Expected value: {(pred_home_win * new_match['home_odds'] - 1)*100:.2f}%")
elif edge_away > EDGE_THRESHOLD:
    print(f"\n✅ BET ON AWAY! (Edge: {edge_away*100:.2f}%)")
    print(f"   Odds: {new_match['away_odds']}")
    print(f"   Expected value: {(pred_away_win * new_match['away_odds'] - 1)*100:.2f}%")
else:
    print(f"\n❌ NO VALUE BET (Edge < {EDGE_THRESHOLD*100}%)")

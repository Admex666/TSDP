import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="🎮 LoL Live Predictor", layout="wide")

# Load models
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(__file__)
    gb_model = joblib.load(os.path.join(BASE_DIR, "models", "live_gb_model_20251031.joblib"))
    rf_model = joblib.load(os.path.join(BASE_DIR, "models", "live_rf_model_20251031.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "live_scaler_20251031.joblib"))
    return gb_model, rf_model, scaler

gb_model, rf_model, scaler = load_models()

# Feature columns (paste your actual feature list here)
feature_cols = [
    'kills_diff', 'towers_diff', 'drakes_diff', 'barons_BLUE', 'barons_RED',
    'gold_diff', 'gold_diff_pct', 'cs_diff', 'gold_per_min_blue', 'gold_per_min_red',
    'cs_per_min_blue', 'cs_per_min_red', 'kill_momentum_3min', 'gold_momentum_5min', 
    'drake_control_score', 'tower_sequence_score', 'baron_timing_advantage', 
    'early_advantage', 'mid_advantage', 'late_advantage', 'has_soul_point', 
    'gold_lead_critical', 'tower_lead_critical', 'phase_early', 'phase_mid',
    'phase_late', 'gold_diff_x_minute', 'kills_diff_x_minute', 'momentum_composite'
]

def calculate_features(minute, k_blue, k_red, t_blue, t_red, d_blue, d_red, 
                       b_blue, b_red, g_blue, g_red, cs_blue, cs_red):
    """Feature engineering from user inputs"""
    
    kills_diff = k_blue - k_red
    towers_diff = t_blue - t_red
    drakes_diff = d_blue - d_red
    gold_diff = g_blue - g_red
    
    gold_diff_pct = (gold_diff / g_red * 100) if g_red > 0 else 0
    gold_per_min_blue = g_blue / minute if minute > 0 else 0
    gold_per_min_red = g_red / minute if minute > 0 else 0
    
    cs_diff = cs_blue - cs_red
    cs_per_min_blue = cs_blue / (minute + 1)
    cs_per_min_red = cs_red / (minute + 1)
    
    kill_momentum_3min = int(kills_diff * 0.3)
    gold_momentum_5min = gold_diff * 0.2 / 5 if minute >= 5 else 0
    
    drake_control_score = drakes_diff * 1.5
    tower_sequence_score = towers_diff * 1.2
    baron_timing_advantage = -1 if (b_blue + b_red) == 0 else 2
    
    phase_early = 1 if minute < 15 else 0
    phase_mid = 1 if 15 <= minute < 25 else 0
    phase_late = 1 if minute >= 25 else 0
    
    early_advantage = (cs_diff + gold_diff/10 + kills_diff*300) / minute if minute > 0 and phase_early else 0
    mid_advantage = (drakes_diff * 2 + towers_diff * 1.5) if phase_mid else 0
    late_advantage = gold_diff_pct + (b_blue - b_red) * 3 if phase_late else 0
    
    has_soul_point = 1 if (d_blue >= 3 or d_red >= 3) else 0
    gold_lead_critical = 1 if abs(gold_diff) > 3000 else 0
    tower_lead_critical = 1 if abs(towers_diff) >= 3 else 0
    
    gold_diff_x_minute = gold_diff * minute
    kills_diff_x_minute = kills_diff * minute
    momentum_composite = kill_momentum_3min + gold_momentum_5min / 100
    
    features = {
        'kills_diff': kills_diff,
        'towers_diff': towers_diff,
        'drakes_diff': drakes_diff,
        'barons_BLUE': b_blue,
        'barons_RED': b_red,
        'gold_diff': gold_diff,
        'gold_diff_pct': gold_diff_pct,
        'cs_diff': cs_diff,
        'gold_per_min_blue': gold_per_min_blue,
        'gold_per_min_red': gold_per_min_red,
        'cs_per_min_blue': cs_per_min_blue,
        'cs_per_min_red': cs_per_min_red,
        'kill_momentum_3min': kill_momentum_3min,
        'gold_momentum_5min': gold_momentum_5min,
        'drake_control_score': drake_control_score,
        'tower_sequence_score': tower_sequence_score,
        'baron_timing_advantage': baron_timing_advantage,
        'early_advantage': early_advantage,
        'mid_advantage': mid_advantage,
        'late_advantage': late_advantage,
        'has_soul_point': has_soul_point,
        'gold_lead_critical': gold_lead_critical,
        'tower_lead_critical': tower_lead_critical,
        'phase_early': phase_early,
        'phase_mid': phase_mid,
        'phase_late': phase_late,
        'gold_diff_x_minute': gold_diff_x_minute,
        'kills_diff_x_minute': kills_diff_x_minute,
        'momentum_composite': momentum_composite
    }
    
    return features

# Title
st.title("🎮 LIVE MATCH WIN PREDICTOR")
st.markdown("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("⏰ Game State")
    minute = st.slider("Minute", 5, 40, 15)
    
    st.subheader("⚔️ Combat Stats")
    kills_blue = st.number_input("🔵 Kills BLUE", 0, 100, 5)
    kills_red = st.number_input("🔴 Kills RED", 0, 100, 3)
    
    st.subheader("🏰 Objectives")
    towers_blue = st.number_input("🔵 Towers BLUE", 0, 11, 2)
    towers_red = st.number_input("🔴 Towers RED", 0, 11, 1)
    drakes_blue = st.number_input("🔵 Drakes BLUE", 0, 4, 2)
    drakes_red = st.number_input("🔴 Drakes RED", 0, 4, 0)
    barons_blue = st.number_input("🔵 Barons BLUE", 0, 5, 0)
    barons_red = st.number_input("🔴 Barons RED", 0, 5, 0)

with col2:
    st.subheader("💰 Economy")
    gold_blue = st.number_input("🔵 Gold BLUE", 0, 100000, 25000, step=1000)
    gold_red = st.number_input("🔴 Gold RED", 0, 100000, 22000, step=1000)

    cs_blue = st.number_input("🔵 CS BLUE", 0, 1500, 500, step=5)
    cs_red = st.number_input("🔴 CS RED", 0, 1500, 520, step=5)
    
    st.subheader("🤖 Model Settings")
    model_choice = st.selectbox("Select Model", ["Gradient Boosting", "Random Forest"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🎯 PREDICT", type="primary", use_container_width=True)

# Prediction
if predict_btn:
    features = calculate_features(minute, kills_blue, kills_red, towers_blue, 
                                  towers_red, drakes_blue, drakes_red,
                                  barons_blue, barons_red, gold_blue, gold_red, cs_blue, cs_red)
    
    X_input = pd.DataFrame([features])[feature_cols]
    
    model = gb_model if model_choice == "Gradient Boosting" else rf_model
    prob_blue = model.predict_proba(X_input)[0, 1]
    prob_red = 1 - prob_blue
    
    odds_blue = 1 / prob_blue
    odds_red = 1 / prob_red
    
    st.markdown("---")
    st.subheader(f"🎯 MATCH PREDICTION @ {minute} MINUTES")
    
    # Current state
    st.markdown(f"""
    **📊 Current Game State:**  
    Kills: {kills_blue}-{kills_red} | Towers: {towers_blue}-{towers_red} | Drakes: {drakes_blue}-{drakes_red}  
    Gold: {gold_blue:,} vs {gold_red:,} ({gold_blue-gold_red:+,} diff) | Barons: {barons_blue}-{barons_red}
    """)
    
    # Win probability bars
    st.markdown("**🎯 WIN PROBABILITY:**")
    col_blue, col_red = st.columns(2)
    with col_blue:
        st.metric("🔵 BLUE", f"{prob_blue*100:.1f}%")
        st.progress(prob_blue)
    with col_red:
        st.metric("🔴 RED", f"{prob_red*100:.1f}%")
        st.progress(prob_red)
    
    # Odds
    st.markdown(f"**💰 IMPLIED ODDS:** 🔵 BLUE: {odds_blue:.2f} | 🔴 RED: {odds_red:.2f}")
    
    # Betting insight
    st.markdown("**📈 BETTING INSIGHT:**")
    if prob_blue > 0.65:
        st.success("✅ STRONG BLUE FAVORITE - Consider backing BLUE")
    elif prob_red > 0.65:
        st.success("✅ STRONG RED FAVORITE - Consider backing RED")
    elif 0.45 < prob_blue < 0.55:
        st.warning("⚠️ CLOSE MATCH - High uncertainty, avoid betting")
    else:
        st.info("ℹ️ Moderate favorite - proceed with caution")
    
    # Key advantages
    st.markdown("**🔑 Key Advantages:**")
    if features['has_soul_point']:
        team = 'BLUE' if drakes_blue >= 3 else 'RED'
        st.write(f"• {team} is at SOUL POINT! (critical)")
    
    if features['gold_lead_critical']:
        team = 'BLUE' if gold_blue > gold_red else 'RED'
        st.write(f"• {team} has critical gold lead (3k+)")
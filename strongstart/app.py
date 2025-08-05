import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and features
@st.cache_resource
def load_models():
    """Loads the machine learning models and feature list."""
    try:
        clf_top4 = joblib.load("strongstart/clf_top4.pkl")
        reg_avgpts = joblib.load("strongstart/reg_avgpts.pkl")
        clf_kieses = joblib.load("strongstart/clf_kieses.pkl")
        features = joblib.load("strongstart/features.pkl")
        return clf_top4, reg_avgpts, clf_kieses, features
    except FileNotFoundError as e:
        st.error(f"Error: One of the model files was not found. Please ensure all required files are in the same directory: {e}")
        st.stop()

clf_top4, reg_avgpts, clf_kieses, features = load_models()

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="End of the Season Outcome Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2b3a67;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3rem;
    }
    h3 {
        color: #3f51b5;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: white;
        color: #4CAF50;
        border: 2px solid #4CAF50;
    }
    .stMetric > div > div > div > div {
        color: #1a237e;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Header ---
st.title("⚽ Premier League – Season Outcome Predictor")
st.markdown("<h3 style='text-align: center;'>Predicting end of the season outcome based on the first 5 games</h3>", unsafe_allow_html=True)

# --- About Section ---
with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown("""
    This tool uses machine learning models (specifically Random Forest) trained on historical Premier League data to predict key season outcomes for any team.
    
    Simply enter your team's performance stats from the **first 5 matches**, and the models will provide predictions for:
    
    - 🏆 The probability of finishing in the **Top 4**
    - 📊 The expected **average points per game** for the season
    - ⚠️ The likelihood of **relegation**
    
    The predictions are for entertainment purposes only and should not be taken as financial or betting advice.
    """)

st.markdown("---")

# --- User Inputs ---
st.header("Enter Your Team's Performance Stats (First 5 Games)")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Match Results")
        total_points = st.number_input("🏅 **Total points** earned", min_value=0, max_value=15, value=7, help="Points from 5 matches (Win=3, Draw=1, Loss=0)")
        goals_scored = st.number_input("⚽ **Total goals** scored", min_value=0, max_value=30, value=8, help="All goals scored by your team")
        goals_conceded = st.number_input("🛡️ **Total goals** conceded", min_value=0, max_value=30, value=6, help="All goals conceded by your team")
        home_games = st.number_input("🏟️ Number of **home games** played", min_value=0, max_value=5, value=3, help="How many of the 5 games were played at home?")

    with col2:
        st.subheader("Statistical Differences (vs. Opponent)")
        shots_on_target_own = st.number_input("🎯 **Shots on target** (your team)", min_value=0, value=25, help="Total shots on target by your team in 5 games")
        shots_on_target_opp = st.number_input("🎯 **Shots on target** (opponent)", min_value=0, value=20, help="Total shots on target by the opponent in 5 games")
        total_shots_own = st.number_input("🔫 **Total shots** (your team)", min_value=0, value=55, help="Total shots by your team in 5 games")
        total_shots_opp = st.number_input("🔫 **Total shots** (opponent)", min_value=0, value=45, help="Total shots by the opponent in 5 games")
        corners_own = st.number_input("🚩 **Corners** (your team)", min_value=0, value=30, help="Total corners by your team in 5 games")
        corners_opp = st.number_input("🚩 **Corners** (opponent)", min_value=0, value=25, help="Total corners by the opponent in 5 games")
        fouls_own = st.number_input("❌ **Fouls** (your team)", min_value=0, value=60, help="Total fouls committed by your team in 5 games")
        fouls_opp = st.number_input("❌ **Fouls** (opponent)", min_value=0, value=55, help="Total fouls committed by the opponent in 5 games")

# --- Derived Feature Calculation ---
home_ratio = home_games / 5
avg_points = total_points / 5
goal_diff = goals_scored - goals_conceded
shot_diff = (total_shots_own - total_shots_opp) / 5
shots_on_target_diff = (shots_on_target_own - shots_on_target_opp) / 5
corners_diff = (corners_own - corners_opp) / 5
foul_diff = (fouls_own - fouls_opp) / 5

# Create a dictionary for model input
input_data = {
    "First10_AvgPoints": avg_points,
    "First10_GD": goal_diff,
    "HomeRatio_First10": home_ratio,
    "First10_ShotDiff": shot_diff,
    "First10_ShotsOnTgt": shots_on_target_diff,
    "First10_FoulDiff": foul_diff,
    "First10_CornerDiff": corners_diff,
}

# --- Prediction and Output ---
st.markdown("---")
if st.button("🔍 Predict Season Outcome", use_container_width=True):
    with st.spinner("Calculating predictions..."):
        X_input = np.array([[input_data[feat] for feat in features]])

        top4_prob = clf_top4.predict_proba(X_input)[0][1]
        avg_pts_prediction = reg_avgpts.predict(X_input)[0]
        releg_prob = clf_kieses.predict_proba(X_input)[0][1]

    st.subheader("🔮 Predictions")
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric(label="🏆 Chance of Top 4 Finish", value=f"{top4_prob:.1%}")
    with colB:
        st.metric(label="📊 Expected Season PPG", value=f"{avg_pts_prediction:.2f}")
    with colC:
        st.metric(label="⚠️ Relegation Probability", value=f"{releg_prob:.1%}")

    # Interpretive messages
    st.markdown("---")
    st.subheader("Analysis & Insights")
    if top4_prob > 0.5:
        st.success("🎉 **Excellent start!** Your team is showing strong performance and is a serious contender for a Champions League spot.")
    elif releg_prob > 0.5:
        st.error("🚨 **Warning:** The current trajectory suggests a high risk of relegation. Significant improvement will be needed to stay in the league.")
    else:
        st.info("💡 **Mixed performance.** This team is currently on a mid-table path. There is still plenty of the season left to either push for European places or fall into a relegation battle.")

    # Visualizations
    st.markdown("### 📊 Probability Breakdown")
    prob_data = pd.DataFrame({
        'Outcome': ['Top 4', 'Relegation', 'Mid-table'],
        'Probability': [top4_prob, releg_prob, 1 - top4_prob - releg_prob]
    })

    st.bar_chart(prob_data.set_index('Outcome'))

    st.markdown("---")
    st.info("Note: The 'Mid-table' category represents the remaining probability after accounting for Top 4 and Relegation.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("## ⚽ End of the Season Predictor")
st.sidebar.markdown("#### The model was trained on 2015-2025 Premier League data")
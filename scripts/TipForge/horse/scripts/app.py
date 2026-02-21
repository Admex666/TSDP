import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="TipForge | Horse Racing Predictor",
    page_icon="🐎",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
    }
    .race-card {
        border-left: 5px solid #ff4b4b;
        padding: 10px;
        background-color: #1a1c24;
        margin-bottom: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_json_cached(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_horse_stats(horse_id, all_horses):
    if not all_horses or "data" not in all_horses or horse_id not in all_horses["data"]:
        return None
    data = all_horses["data"][horse_id]
    # Results is a list of race performances
    results = data.get("results", [])
    total_runs = len(results)
    total_wins = sum(1 for r in results if r.get("rank") == "I.")
    win_rate = (total_wins / total_runs) if total_runs > 0 else 0.05
    return {"win_rate": win_rate, "total_runs": total_runs}

import pickle

def load_ml_assets():
    try:
        if os.path.exists("models/horse_model.pkl"):
            with open("models/horse_model.pkl", "rb") as f:
                model = pickle.load(f)
            if os.path.exists("models/shap_explainer.pkl"):
                with open("models/shap_explainer.pkl", "rb") as f:
                    explainer = pickle.load(f)
            else:
                explainer = None
            return model, explainer
    except Exception as e:
        print(f"Error loading model: {e}")
    return None, None

def calculate_fair_odds(participants, all_horses, all_drivers):
    model, _ = load_ml_assets()
    if model:
        st.sidebar.success("🤖 XGBoost Model Active")
        return calculate_ml_odds(participants, all_horses, model, all_drivers)
    
    st.sidebar.warning("⚠️ Heuristic Model Active (No trained model found)")
    # Fallback to heuristic
    scores = []
    for p in participants:
        stats = get_horse_stats(str(p["horse_id"]), all_horses)
        scores.append(stats["win_rate"] if stats else 0.05)
    
    total = sum(scores)
    probs = [s/total for s in scores]
    odds = [min(50.0, 1.0/p) for p in probs] 
    return probs, odds

def calculate_ml_odds(participants, all_horses, model, all_drivers):
    rows = []
    for p in participants:
        h_id = str(p["horse_id"])
        d_id = str(p["driver_id"])
        
        h_stats = get_horse_stats(h_id, all_horses)
        d_stats = get_horse_stats(d_id, all_drivers)
        
        try:
            dist_val = float(selected_race.get("distance", "1900").replace("A", "").replace("G", ""))
        except:
            dist_val = 1900.0

        row = [
            dist_val, 
            h_stats["win_rate"] if h_stats else 0.05,
            78.0, 
            h_stats["total_runs"] if h_stats else 0,
            d_stats["win_rate"] if d_stats else 0.05,
            d_stats["total_runs"] if d_stats else 0
        ]
        rows.append(row)
    
    feature_names = ["distance", "horse_win_rate", "horse_avg_km", "horse_runs", "driver_win_rate", "driver_runs"]
    probs = model.predict_proba(pd.DataFrame(rows, columns=feature_names))[:, 1]
    
    total = sum(probs)
    probs = [p/total for p in probs]
    odds = [min(50.0, 1.0/p) for p in probs]
    return probs, odds

# Title
st.title("🐎 TipForge Horse Racing | Daily Predictions")
st.markdown("### Budapest-Kincsem Park - " + datetime.now().strftime("%Y-%m-%d"))

# Sidebar for race selection
racecard = load_json_cached("data/today_racecard.json")
all_horses = load_json_cached("data/today_horses.json")
all_drivers = load_json_cached("data/today_drivers.json")

if not racecard:
    st.error("Racecard data not found. Please run scripts/parse_racecard.py")
else:
    race_names = [f"Race {i+1}: {r['race_name']} ({r['start_time']})" for i, r in enumerate(racecard)]
    selected_race_idx = st.sidebar.selectbox("Select Race", range(len(race_names)), format_func=lambda x: race_names[x])
    
    selected_race = racecard[selected_race_idx]
    
    st.header(f" {selected_race['race_name']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Start Time", selected_race["start_time"])
    col2.metric("Distance", selected_race["distance"])
    col3.metric("Participants", len(selected_race["participants"]))

    # Predictions
    st.subheader("🎯 Fair Odds & Probabilities")
    
    probs, odds = calculate_fair_odds(selected_race["participants"], all_horses, all_drivers)
    
    results = []
    for i, p in enumerate(selected_race["participants"]):
        results.append({
            "Horse": p["horse_name"],
            "Driver": p["driver_name"],
            "Probability (%)": round(probs[i] * 100, 1),
            "Fair Odds": round(odds[i], 2)
        })
    
    rdf = pd.DataFrame(results).sort_values("Probability (%)", ascending=False)
    
    # Visualizations
    fig = px.bar(rdf, x="Probability (%)", y="Horse", orientation='h', 
                 title="Win Probability by Horse",
                 text="Probability (%)",
                 color="Probability (%)",
                 color_continuous_scale="RdYlGn")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Table
    st.table(rdf)

    # Explainability Section
    st.subheader("💡 Why these odds? (AI Explanation)")
    
    model, explainer = load_ml_assets()
    if model and explainer:
        # Generate SHAP for the selected horse (highest probability)
        top_horse_idx = rdf.index[0]
        top_horse_name = rdf.loc[top_horse_idx, "Horse"]
        
        # Prepare feature vector for SHAP
        # Features: [distance, horse_win_rate, horse_avg_km, horse_runs, driver_win_rate, driver_runs]
        h_id = str(selected_race["participants"][top_horse_idx]["horse_id"])
        d_id = str(selected_race["participants"][top_horse_idx]["driver_id"])
        h_stats = get_horse_stats(h_id, all_horses)
        d_stats = get_horse_stats(d_id, all_drivers)
        
        model_features = ["distance", "horse_win_rate", "horse_avg_km", "horse_runs", "driver_win_rate", "driver_runs"]
        display_map = {
            "distance": "Distance", 
            "horse_win_rate": "Horse Win Rate", 
            "horse_avg_km": "Avg Km Time", 
            "horse_runs": "Horse Experience", 
            "driver_win_rate": "Driver Win Rate", 
            "driver_runs": "Driver Experience"
        }
        
        try:
            dist_val = float(selected_race.get("distance", "1900").replace("A", "").replace("G", ""))
        except:
            dist_val = 1900.0

        feature_values = [
            dist_val, 
            h_stats["win_rate"] if h_stats else 0.05,
            78.0, 
            h_stats["total_runs"] if h_stats else 0,
            d_stats["win_rate"] if d_stats else 0.05,
            d_stats["total_runs"] if d_stats else 0
        ]
        
        # XGBoost expects the exact feature names it was trained on
        shap_df = pd.DataFrame([feature_values], columns=model_features)
        shap_values = explainer.shap_values(shap_df)
        
        s_df = pd.DataFrame({
            "Feature": [display_map[f] for f in model_features],
            "Impact Score": shap_values[0]
        }).sort_values("Impact Score", ascending=False)
        
        st.write(f"Top factor for **{top_horse_name}**:")
        
        fig_shap = px.bar(s_df, x="Impact Score", y="Feature", orientation='h',
                          title=f"Factor Impact for {top_horse_name}",
                          color="Impact Score",
                          color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_shap, use_container_width=True)
        
        st.markdown("""
        - **Green bars:** These factors increased the horse's win probability.
        - **Red bars:** These factors decreased the horse's win probability.
        """)
    else:
        st.info("Heuristic Model: Career Win Rate Balance active. Train XGBoost for detailed AI explanations.")

# Footer
st.markdown("---")
st.caption("Data provided by MLA Kincsem Park API. Predictions are for educational purposes only.")

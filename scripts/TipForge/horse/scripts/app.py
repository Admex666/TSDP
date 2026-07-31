import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import sys
import os

# Add the project root to sys.path to allow absolute imports from 'scripts'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def get_path(rel_path):
    if os.path.exists(rel_path):
        return rel_path
    alt_path = os.path.join(parent_dir, rel_path)
    if os.path.exists(alt_path):
        return alt_path
    return rel_path

from scripts.prepare_features import calculate_point_in_time_stats

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
    resolved = get_path(path)
    if os.path.exists(resolved):
        with open(resolved, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_horse_stats(horse_id, all_horses):
    if not all_horses or "data" not in all_horses or horse_id not in all_horses["data"]:
        return None
    data = all_horses["data"][horse_id]
    # Results is a list of race performances
    results = data.get("results", [])
    total_runs = len(results)
    total_wins = sum(1 for r in results if r.get("placement") in ["1.", "I."] or r.get("rank") in ["1.", "I."])
    win_rate = (total_wins / total_runs) if total_runs > 0 else 0.05
    return {"win_rate": win_rate, "total_runs": total_runs}

import pickle

def load_ml_assets():
    """Prefers V4 calibrated model, falls back to V3, then V2."""
    for model_rel, shap_rel, label in [
        ("models/horse_model_v4.pkl", "models/shap_explainer_v4.pkl", "V4"),
        ("models/horse_model_v3.pkl", "models/shap_explainer_v3.pkl", "V3"),
        ("models/horse_model.pkl",    "models/shap_explainer.pkl",    "V2"),
    ]:
        model_path = get_path(model_rel)
        shap_path = get_path(shap_rel)
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                explainer = None
                if os.path.exists(shap_path):
                    with open(shap_path, "rb") as f:
                        explainer = pickle.load(f)
                return model, explainer, label
            except Exception as e:
                print(f"Error loading {label}: {e}")
    return None, None, None

def calculate_ml_odds(participants, all_horses, model, all_drivers, race_field_map, date_str, race_dist,
                      track_quality=0, temperature=15.0, pair_experience=None, trainer_stats=None):
    """Builds V4 feature rows and returns calibrated win probabilities & fair odds."""
    if pair_experience is None: pair_experience = {}
    if trainer_stats is None: trainer_stats = {}
    
    rows = []
    for p in participants:
        h_id = str(p["horse_id"])
        d_id = str(p["driver_id"])
        t_id = str(p.get("trainer_id"))
        
        h_history = all_horses.get("data", {}).get(h_id, {}).get("results", [])
        h_stats = calculate_point_in_time_stats(h_history, date_str, race_field_map)
        d_stats = calculate_point_in_time_stats(all_drivers.get("data", {}).get(d_id, {}).get("results", []), date_str, race_field_map)
        
        # New V4 features extraction
        h_age = p.get("age", 5)
        h_sex_str = str(p.get("sex", "male")).lower()
        h_sex_val = 0
        if "female" in h_sex_str: h_sex_val = 1
        elif "gelding" in h_sex_str or "herelt" in h_sex_str: h_sex_val = 2
        
        past_h = [r for r in h_history if r.get("date") and r["date"] < date_str][-10:]
        gallops = sum(1 for r in past_h if "gal" in str(r.get("placement", r.get("rank", ""))).lower())
        h_gallop_rate = gallops / len(past_h) if past_h else 0
        
        win_dists = [float(r.get("distance", 1900)) for r in h_history 
                     if r.get("date") and r["date"] < date_str and str(r.get("rank")).startswith(("1.", "I."))]
        avg_win_dist = sum(win_dists)/len(win_dists) if win_dists else 1900.0
        dist_diff = abs(race_dist - avg_win_dist)

        t_s = trainer_stats.get(t_id, {"win_rate": 0.05, "top3_rate": 0.15})

        row = [
            race_dist, track_quality, temperature,
            h_stats["win_rate"], h_stats["top_3_rate"], h_stats["avg_percentile"], h_stats["avg_speed"],
            h_stats.get("best_speed_life"), h_stats.get("speed_ratio", 1.0),
            h_stats["total_prize"], h_stats["days_since_last"],
            h_stats["win_rate_l5"], h_stats["top_3_rate_l5"], h_stats["avg_percentile_l5"], h_stats["avg_speed_l5"],
            h_stats.get("points_l5", 0), h_stats.get("top3_l3", 0),
            d_stats["win_rate"], d_stats["top_3_rate"],
            pair_experience.get((h_id, d_id), 0),
            # V4 extra columns
            h_age, h_sex_val, h_gallop_rate, dist_diff, t_s["win_rate"], t_s["top3_rate"]
        ]
        rows.append(row)
    
    feature_names = [
        "distance", "track_quality", "temperature",
        "h_win_rate", "h_top_3_rate", "h_avg_percentile", "h_avg_speed",
        "h_best_speed", "h_speed_ratio",
        "h_total_prize", "h_days_since",
        "h_win_rate_l5", "h_top_3_rate_l5", "h_avg_percentile_l5", "h_avg_speed_l5",
        "h_points_l5", "h_top3_l3",
        "d_win_rate", "d_top_3_rate",
        "hd_pair_runs",
        "h_age", "h_sex", "h_gallop_rate", "dist_diff", "t_win_rate", "t_top3_rate"
    ]
    feature_df = pd.DataFrame(rows, columns=feature_names).astype(float)
    
    for col in ["h_avg_speed", "h_avg_speed_l5", "h_best_speed"]:
        feature_df[col] = feature_df[col].fillna(feature_df[col].mean() or 12.8)
    feature_df["h_speed_ratio"] = feature_df["h_speed_ratio"].fillna(1.0)
    
    probs = model.predict_proba(feature_df)[:, 1]
    total = sum(probs)
    probs = [p/total for p in probs]
    odds = [min(50.0, 1.0/p) for p in probs]
    return probs, odds

# Sidebar Navigation
page = st.sidebar.selectbox("Navigation", ["Daily Predictions", "Model Analytics"])

def show_predictions_page():
    st.title("🐎 TipForge Horse Racing | Daily Predictions")
    st.markdown("### Budapest-Kincsem Park - " + datetime.now().strftime("%Y-%m-%d"))

    # Sidebar for race selection
    racecard = load_json_cached("data/today_racecard.json")
    all_horses = load_json_cached("data/today_horses.json")
    all_drivers = load_json_cached("data/today_drivers.json")
    trainer_stats = load_json_cached("data/trainer_stats.json")

    # Build map for today's races
    race_field_map = {}
    if racecard:
        for race in racecard:
            race_field_map[str(race.get("race_id"))] = len(race.get("participants", []))

    if not racecard:
        st.error("Racecard data not found. Please run scripts/parse_racecard.py")
        return

    race_names = [f"Race {i+1}: {r['race_name']} ({r['start_time']})" for i, r in enumerate(racecard)]
    selected_race_idx = st.sidebar.selectbox("Select Race", range(len(race_names)), format_func=lambda x: race_names[x])
    
    selected_race = racecard[selected_race_idx]
    date_str = datetime.now().strftime("%Y-%m-%d") # Use today for PiT in app
    
    st.header(f" {selected_race['race_name']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Start Time", selected_race["start_time"])
    col2.metric("Distance", selected_race["distance"])
    col3.metric("Participants", len(selected_race["participants"]))

    # Data Coverage Check
    stats_count = 0
    for p in selected_race["participants"]:
        if all_horses and str(p["horse_id"]) in all_horses.get("data", {}):
            stats_count += 1
    
    st.sidebar.write(f"📊 Data Coverage: {stats_count}/{len(selected_race['participants'])} horses found")

    # Predictions
    st.subheader("🎯 Fair Odds & Probabilities")
    
    model, explainer, model_label = load_ml_assets()
    
    try:
        race_dist = float(selected_race.get("distance", "1900").replace("A", "").replace("G", ""))
    except:
        race_dist = 1900.0

    if model:
        st.sidebar.success(f"🤖 XGBoost {model_label} Active (Calibrated)")
        probs, odds = calculate_ml_odds(selected_race["participants"], all_horses, model, all_drivers, 
                                        race_field_map, date_str, race_dist, trainer_stats=trainer_stats)
    else:
        st.sidebar.warning("⚠️ No Model Found")
        probs = [1.0/len(selected_race["participants"])] * len(selected_race["participants"])
        odds = [float(len(selected_race["participants"]))] * len(selected_race["participants"])
    
    results = []
    for i, p in enumerate(selected_race["participants"]):
        results.append({
            "_prob": probs[i],
            "_fair_odds": odds[i],
            "Horse": p["horse_name"],
            "Driver": p["driver_name"],
            "Probability (%)": round(probs[i] * 100, 2),
            "Fair Odds": round(odds[i], 2),
            "_market_odds": p.get("market_odds"),  # may be None for today
        })

    rdf = pd.DataFrame(results).sort_values("Probability (%)", ascending=False).reset_index(drop=True)

    # Visualizations
    fig = px.bar(rdf, x="Probability (%)", y="Horse", orientation='h',
                 title="Win Probability by Horse",
                 text="Probability (%)",
                 color="Probability (%)",
                 color_continuous_scale="RdYlGn")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # ── Kelly Bet Sizing Calculator ────────────────────────────────────────────
    st.subheader("📐 Kelly Bet Sizing Calculator")

    # Sidebar: bankroll + Kelly fraction (persistent)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Kelly Calculator")
    bankroll = st.sidebar.number_input(
        "Bankroll (Ft)", min_value=1_000, max_value=100_000_000,
        value=100_000, step=5_000, format="%d"
    )
    kelly_mode = st.sidebar.radio(
        "Kelly Fraction",
        options=["Full Kelly (1.0×)", "Half Kelly (0.5×)", "Quarter Kelly (0.25×)"],
        index=1  # default: Half Kelly
    )
    kelly_frac = {"Full Kelly (1.0×)": 1.0, "Half Kelly (0.5×)": 0.5, "Quarter Kelly (0.25×)": 0.25}[kelly_mode]

    def calc_kelly(prob, market_odds, broll, frac):
        if market_odds <= 1.0:
            return 0, 0.0, 0.0
        b = market_odds - 1.0
        raw_f = (b * prob - (1.0 - prob)) / b
        f = max(raw_f, 0.0) * frac
        f = min(f, 0.20)
        stake = max(round(broll * f / 100) * 100, 0)
        edge_pct = (market_odds / (1.0 / prob) - 1) * 100 if prob > 0 else 0
        return stake, raw_f * 100, edge_pct

    # Header row
    h1, h2, h3 = st.columns([5, 2.5, 2.5])
    h1.markdown("**🐎 Induló & AI Odds**")
    h2.markdown("**Odds (BetLovi)**")
    h3.markdown("**Értékelés & Tét**")

    st.divider()

    # Pre-read all market odds to calculate market ranks
    mkt_odds_list = []
    for i, row in rdf.iterrows():
        horse = row["Horse"]
        key = f"odds_{i}_{horse[:8]}"
        mkt_val = st.session_state.get(key, float(row["_fair_odds"]))
        mkt_odds_list.append((i, mkt_val))
    
    # Sort and rank: lower odds = smaller rank (favorite)
    sorted_mkt = sorted(mkt_odds_list, key=lambda x: (x[1], x[0]))
    ranks = {item[0]: rank + 1 for rank, item in enumerate(sorted_mkt)}

    for i, row in rdf.iterrows():
        prob      = row["_prob"]
        fair      = row["_fair_odds"]
        horse     = row["Horse"]
        rank      = ranks[i]

        c1, c2, c3 = st.columns([5, 2.5, 2.5])
        
        # Column 1: Horse info, AI Probability, Fair Odds, and Rank
        rank_badge = ""
        if rank == 1:
            rank_badge = "🥇 Favorit"
        elif rank == 2:
            rank_badge = "🥈 2. Favorit"
        elif rank == 3:
            rank_badge = "🥉 3. Favorit"
        else:
            rank_badge = f"{rank}. hely a piacon"
            
        info_html = f"**{horse}** ({row['Driver']})<br>`AI: {prob*100:.1f}%` | `Fair: {fair:.2f}`<br><small style='color:#cccccc;'>{rank_badge}</small>"
        c1.markdown(info_html, unsafe_allow_html=True)
        
        if fair < 30.0:
            min_val_odds = (30.0 * fair) / (30.0 - fair)
            c1.markdown(f"<small style='color: #4CAF50;'>Value odds küszöb: **{min_val_odds:.2f}**</small>", unsafe_allow_html=True)
        else:
            c1.markdown(f"<small style='color: #757575;'>Nincs value</small>", unsafe_allow_html=True)

        # Column 2: Number input for bookmaker odds
        mkt = c2.number_input(
            "Odds", min_value=1.01, max_value=100.0, value=max(1.01, float(fair)),
            step=0.05, format="%.2f",
            key=f"odds_{i}_{horse[:8]}",
            label_visibility="collapsed"
        )

        # Column 3: Kelly Sizing and Value assessment
        stake, raw_kelly_pct, edge_pct = calc_kelly(prob, mkt, bankroll, kelly_frac)
        
        if raw_kelly_pct <= 0:
            c3.markdown("**Tét: —**")
            c3.markdown("<small style='color:#757575;'>nincs érték</small>", unsafe_allow_html=True)
        else:
            required_edge_pct = 10.0 * (mkt / 3.0)
            if rank <= 3:
                if edge_pct >= required_edge_pct:
                    c3.markdown(f"**Tét: {stake:,} Ft**")
                    c3.markdown(f"<span style='color:#4CAF50; font-weight:bold;'>🔥 VALUE</span><br><small>`+{edge_pct:.1f}%` (elvárt: `>{required_edge_pct:.1f}%` / Rank: **{rank}.**)</small>", unsafe_allow_html=True)
                elif edge_pct >= (required_edge_pct * 0.5):
                    c3.markdown(f"**Tét: {stake:,} Ft**")
                    c3.markdown(f"<span style='color:#FF9800; font-weight:bold;'>🟡 gyenge edge</span><br><small>`+{edge_pct:.1f}%` (elvárt: `>{required_edge_pct:.1f}%` / Rank: **{rank}.**)</small>", unsafe_allow_html=True)
                else:
                    c3.markdown("**Tét: —**")
                    c3.markdown(f"<span style='color:#f44336; font-weight:bold;'>❌ nincs elég edge</span><br><small>(elvárt: `>{required_edge_pct:.1f}%` / Rank: **{rank}.**)</small>", unsafe_allow_html=True)
            else:
                c3.markdown("**Tét: —**")
                c3.markdown(f"<span style='color:#757575; font-weight:bold;'>❌ Nem TOP 3 favorit</span><br><small>(Rank: **{rank}.** / odds: {mkt:.2f})</small>", unsafe_allow_html=True)

        st.divider()

    st.caption(f"Bankroll: **{bankroll:,} Ft** | Módszer: **{kelly_mode}** | Max tét/fogadás: 20%")

    # Explainability Section
    st.subheader("💡 Why these odds? (AI Explanation)")
    
    if model and explainer:
        selected_explain_horse = st.selectbox(
            "Válaszd ki a lovat a magyarázathoz:",
            options=rdf["Horse"].tolist(),
            key="explain_horse_select"
        )
        
        # Get participant index of selected horse
        p_idx = next(idx for idx, p in enumerate(selected_race["participants"]) if p["horse_name"] == selected_explain_horse)
        top_p = selected_race["participants"][p_idx]
        h_id = str(top_p["horse_id"])
        d_id = str(top_p["driver_id"])
        t_id = str(top_p.get("trainer_id"))
        
        h_stats = calculate_point_in_time_stats(all_horses.get("data", {}).get(h_id, {}).get("results", []), date_str, race_field_map)
        d_stats = calculate_point_in_time_stats(all_drivers.get("data", {}).get(d_id, {}).get("results", []), date_str, race_field_map)
        
        feature_map = {
            "distance":           "Távolság (Distance)",
            "track_quality":      "Pálya minősége (Track quality)",
            "temperature":        "Hőmérséklet (Temperature)",
            "h_win_rate":         "Életút győzelmi arány (Career win rate)",
            "h_top_3_rate":       "Podium arány (Career top-3 rate)",
            "h_avg_percentile":   "Átlagos relatív helyezés (Avg rank %)",
            "h_avg_speed":        "Átlagsebesség (Career avg speed)",
            "h_best_speed":       "Csúcssebesség (Career best speed)",
            "h_speed_ratio":      "Forma index (L5 speed / best speed)",
            "h_total_prize":      "Összdíjazás (Total prize)",
            "h_days_since":       "Utolsó verseny óta (Days since last)",
            "h_win_rate_l5":      "L5 győzelmi arány (L5 win rate)",
            "h_top_3_rate_l5":    "L5 podium arány (L5 top-3 rate)",
            "h_avg_percentile_l5":"L5 relatív helyezés (L5 rank %)",
            "h_avg_speed_l5":     "L5 sebesség (L5 avg speed)",
            "h_points_l5":        "Forma pontok (L5 weighted points)",
            "h_top3_l3":          "L3 podium arány (L3 top-3 rate)",
            "d_win_rate":         "Hajtó győzelmi arány (Driver win rate)",
            "d_top_3_rate":       "Hajtó podium arány (Driver top-3 rate)",
            "hd_pair_runs":       "Pár tapasztalat (Horse-driver pair runs)",
            "h_age":              "Ló kora (Horse Age)",
            "h_sex":              "Ló neme (Horse Sex)",
            "h_gallop_rate":      "Hiba arány (Gallop/Break rate)",
            "dist_diff":          "Távolság preferencia (Dist delta)",
            "t_win_rate":         "Tréner győzelmi arány (Trainer win rate)",
            "t_top3_rate":        "Tréner podium arány (Trainer top-3 rate)",
        }
        
        model_features = list(feature_map.keys())
        
        # Helper to format feature values for tooltips
        def format_feature_value(name, val):
            if name == "distance": return f"{val:.0f} m"
            if name == "track_quality": return "Jó" if val == 0 else "Gát / Nehéz"
            if name == "temperature": return f"{val:.1f} °C"
            if name in ["h_win_rate", "h_top_3_rate", "h_win_rate_l5", "h_top_3_rate_l5", "h_top3_l3", "d_win_rate", "d_top_3_rate", "t_win_rate", "t_top3_rate", "h_gallop_rate"]:
                return f"{val * 100:.1f}%"
            if name in ["h_avg_percentile", "h_avg_percentile_l5"]:
                return f"{val * 100:.1f} percentile"
            if name in ["h_avg_speed", "h_avg_speed_l5", "h_best_speed"]:
                return f"{val:.2f} s/km"
            if name == "h_speed_ratio": return f"{val:.3f}"
            if name == "h_total_prize": return f"{val:,.0f} Ft"
            if name == "h_days_since": return f"{val:.0f} nap"
            if name == "h_points_l5": return f"{val:.1f} pont"
            if name == "hd_pair_runs": return f"{val:.0f} futás"
            if name == "h_age": return f"{val:.0f} éves"
            if name == "h_sex": return "Kanca" if val == 1 else ("Herélt" if val == 2 else "Mén")
            if name == "dist_diff": return f"{val:.0f} m"
            return str(val)

        try:
            # Re-calculate values for SHAP
            h_sex_str = str(top_p.get("sex", "male")).lower()
            h_sex_val = 1 if "female" in h_sex_str else (2 if ("gelding" in h_sex_str or "herelt" in h_sex_str) else 0)
            
            h_history = all_horses.get("data", {}).get(h_id, {}).get("results", [])
            past_h = [r for r in h_history if r.get("date") and r["date"] < date_str][-10:]
            h_gallop_rate = sum(1 for r in past_h if "gal" in str(r.get("placement", r.get("rank", ""))).lower()) / len(past_h) if past_h else 0
            
            win_dists = [float(r.get("distance", 1900)) for r in h_history 
                         if r.get("date") and r["date"] < date_str and str(r.get("rank")).startswith(("1.", "I."))]
            dist_diff = abs(race_dist - (sum(win_dists)/len(win_dists) if win_dists else 1900.0))
            
            t_s = (trainer_stats or {}).get(t_id, {"win_rate": 0.05, "top3_rate": 0.15})

            feature_values = [
                race_dist, 0, 15.0,
                h_stats["win_rate"], h_stats["top_3_rate"], h_stats["avg_percentile"], h_stats["avg_speed"] or 12.8,
                h_stats.get("best_speed_life") or 12.8, h_stats.get("speed_ratio", 1.0),
                h_stats["total_prize"], h_stats["days_since_last"],
                h_stats["win_rate_l5"], h_stats["top_3_rate_l5"], h_stats["avg_percentile_l5"], h_stats["avg_speed_l5"] or 12.8,
                h_stats.get("points_l5", 0), h_stats.get("top3_l3", 0),
                d_stats["win_rate"], d_stats["top_3_rate"],
                0, # hd_pair_runs
                top_p.get("age", 5), h_sex_val, h_gallop_rate, dist_diff, t_s["win_rate"], t_s["top3_rate"]
            ]
            
            shap_df = pd.DataFrame([feature_values], columns=model_features)
            
            # Align shap_df with the explainer's expected features
            if hasattr(explainer, "data_feature_names") and explainer.data_feature_names:
                expected_shap_features = list(explainer.data_feature_names)
            elif hasattr(explainer, "model") and hasattr(explainer.model, "feature_names"):
                expected_shap_features = list(explainer.model.feature_names)
            else:
                expected_shap_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else model_features
                
            for col in expected_shap_features:
                if col not in shap_df.columns:
                    shap_df[col] = 0.0
            shap_df = shap_df[expected_shap_features]
            
            shap_values = explainer.shap_values(shap_df)
            
            # Match calculated values and formats to expectations
            val_dict = dict(zip(model_features, feature_values))
            feature_desc_list = []
            val_formatted_list = []
            
            for f_key in expected_shap_features:
                val_raw = val_dict.get(f_key, 0.0)
                val_fmt = format_feature_value(f_key, val_raw)
                
                feature_desc_list.append(feature_map.get(f_key, f_key))
                val_formatted_list.append(val_fmt)
            
            s_df = pd.DataFrame({
                "Feature": feature_desc_list,
                "Impact Score": shap_values[0],
                "Tényleges Érték": val_formatted_list
            }).sort_values("Impact Score", ascending=False)
            
            st.write(f"A tényezők hatása **{selected_explain_horse}** nyerési esélyére:")
            fig_shap = px.bar(
                s_df, 
                x="Impact Score", 
                y="Feature", 
                orientation='h',
                title=f"Factor Impact for {selected_explain_horse}",
                color="Impact Score",
                color_continuous_scale="RdYlGn",
                hover_data={"Feature": True, "Impact Score": ":.4f", "Tényleges Érték": True}
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as shap_err:
            st.info(f"SHAP explanation unavailable for this model version. ({shap_err})")
        
        st.markdown("""
        - **Zöld oszlopok:** Ezek a tényezők növelték a ló nyerési esélyét.
        - **Piros oszlopok:** Ezek a tényezők csökkentették a ló nyerési esélyét.
        """)
    else:
        st.info("No explainer loaded. Run train_model.py to generate SHAP explainer.")

def show_analytics_page():
    st.title("📈 Model Performance & Analytics")
    st.markdown("### Historical Backtest Analysis (2025 Test Set)")

    v4_path = get_path("data/training_set_v4.csv")
    v3_path = get_path("data/training_set_v3.csv")
    csv_path = v4_path if os.path.exists(v4_path) else v3_path
    if not os.path.exists(csv_path):
        st.error("Training dataset not found. Run prepare_features.py first.")
        return

    df = pd.read_csv(csv_path)
    test_df = df[df["date"] >= "2025-01-01"].copy()
    
    if len(test_df) == 0:
        st.warning("No data found for 2025. Showing 2024 data as sample.")
        test_df = df.copy()

    # Load model
    model, _, model_label = load_ml_assets()
    if not model:
        st.error("No trained model found.")
        return

    st.info(f"Model: **{model_label}** | Dataset: `{csv_path}`")

    # V4 features (fallback to available subset)
    all_features = [
        "distance", "track_quality", "temperature",
        "h_win_rate", "h_top_3_rate", "h_avg_percentile", "h_avg_speed",
        "h_best_speed", "h_speed_ratio",
        "h_total_prize", "h_days_since",
        "h_win_rate_l5", "h_top_3_rate_l5", "h_avg_percentile_l5", "h_avg_speed_l5",
        "h_points_l5", "h_top3_l3",
        "d_win_rate", "d_top_3_rate",
        "hd_pair_runs",
        "h_age", "h_sex", "h_gallop_rate", "dist_diff", "t_win_rate", "t_top3_rate"
    ]
    features = [f for f in all_features if f in test_df.columns]
    X_test = test_df[features].fillna(test_df[features].mean())
    y_test = test_df["win"]
    
    probs = model.predict_proba(X_test)[:, 1]
    test_df["ml_prob"] = probs

    # Metrics
    from sklearn.metrics import accuracy_score, brier_score_loss
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", f"{acc:.2%}")
    col2.metric("Brier Score", f"{brier:.4f}")
    col3.metric("Test Races", len(test_df["race_id"].unique()))

    # ── Baseline Strategy Comparison ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 Baseline Strategy Comparison")
    st.write(f"How does Model {model_label} compare against simple betting strategies?")

    baseline_path = get_path("data/baseline_results.csv")
    if os.path.exists(baseline_path):
        bl_df = pd.read_csv(baseline_path)
        fig_bl = px.bar(
            bl_df, x="roi_pct", y="strategy", orientation="h",
            title="ROI by Strategy (2025 Test Set)",
            text="roi_pct",
            color="roi_pct",
            color_continuous_scale=["#ef553b", "#ffa15a", "#00cc96"],
            range_color=[-70, 40],
        )
        st.plotly_chart(fig_bl, use_container_width=True)
    else:
        st.info("Baseline results not found.")

    # Calibration Plot
    st.markdown("---")
    st.subheader(f"🎯 Probability Calibration ({model_label})")
    cal_bins = pd.cut(test_df['ml_prob'], bins=10)
    cal_data = test_df.groupby(cal_bins, observed=False)['win'].agg(['mean', 'count']).reset_index()
    cal_data.columns = ['bin', 'actual_win_rate', 'count']
    cal_data['mid_prob'] = [iv.mid for iv in cal_data['bin']]
    fig_cal = px.scatter(cal_data, x='mid_prob', y='actual_win_rate', size='count', title="Reliability Diagram")
    fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig_cal, use_container_width=True)

    # Feature Importance
    st.subheader(f"📊 Global Feature Importance ({model_label})")
    try:
        inner = model.calibrated_classifiers_[0].estimator
        importance_vals = inner.feature_importances_
        importances = pd.DataFrame({'Feature': features, 'Importance': importance_vals[:len(features)]}).sort_values('Importance', ascending=True)
        fig_imp = px.bar(importances, x="Importance", y="Feature", orientation='h', title="XGBoost Feature Gain", color="Importance", color_continuous_scale="Viridis")
        st.plotly_chart(fig_imp, use_container_width=True)
    except:
        st.info("Feature importance unavailable.")

if page == "Daily Predictions":
    show_predictions_page()
elif page == "Model Analytics":
    show_analytics_page()

# Footer
st.markdown("---")
st.caption(f"TipForge Horse Racing. Model V4 (XGBoost Calibrated). Data from Kincsem Park.")

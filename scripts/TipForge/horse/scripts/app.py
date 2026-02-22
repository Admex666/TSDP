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
    total_wins = sum(1 for r in results if r.get("placement") in ["1.", "I."] or r.get("rank") in ["1.", "I."])
    win_rate = (total_wins / total_runs) if total_runs > 0 else 0.05
    return {"win_rate": win_rate, "total_runs": total_runs}

import pickle

def load_ml_assets():
    """Prefers V3 calibrated model, falls back to V2."""
    for model_path, shap_path, label in [
        ("models/horse_model_v3.pkl", "models/shap_explainer_v3.pkl", "V3"),
        ("models/horse_model.pkl",    "models/shap_explainer.pkl",    "V2"),
    ]:
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
                      track_quality=0, temperature=15.0, pair_experience=None):
    """Builds V3 feature rows and returns calibrated win probabilities & fair odds."""
    if pair_experience is None:
        pair_experience = {}
    rows = []
    for p in participants:
        h_id = str(p["horse_id"])
        d_id = str(p["driver_id"])
        
        h_stats = calculate_point_in_time_stats(all_horses.get("data", {}).get(h_id, {}).get("results", []), date_str, race_field_map)
        d_stats = calculate_point_in_time_stats(all_drivers.get("data", {}).get(d_id, {}).get("results", []), date_str, race_field_map)
        
        row = [
            race_dist,
            track_quality,
            temperature,
            h_stats["win_rate"], h_stats["top_3_rate"], h_stats["avg_percentile"], h_stats["avg_speed"],
            h_stats.get("best_speed_life"), h_stats.get("speed_ratio", 1.0),
            h_stats["total_prize"], h_stats["days_since_last"],
            h_stats["win_rate_l5"], h_stats["top_3_rate_l5"], h_stats["avg_percentile_l5"], h_stats["avg_speed_l5"],
            h_stats.get("points_l5", 0), h_stats.get("top3_l3", 0),
            d_stats["win_rate"], d_stats["top_3_rate"],
            pair_experience.get((h_id, d_id), 0)
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
        "hd_pair_runs"
    ]
    feature_df = pd.DataFrame(rows, columns=feature_names)
    
    # Fillna for speed fields
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
        probs, odds = calculate_ml_odds(selected_race["participants"], all_horses, model, all_drivers, race_field_map, date_str, race_dist)
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
    h1, h2, h3, h4, h5, h6 = st.columns([3, 1.5, 1.5, 1.5, 1.5, 2])
    h1.markdown("**🐎 Ló**")
    h2.markdown("**AI Prob**")
    h3.markdown("**Fair Odds**")
    h4.markdown("**Odds (add meg)**")
    h5.markdown("**Kelly Tét**")
    h6.markdown("**Értékelés**")

    st.divider()

    for i, row in rdf.iterrows():
        prob      = row["_prob"]
        fair      = row["_fair_odds"]
        horse     = row["Horse"]

        c1, c2, c3, c4, c5, c6 = st.columns([3, 1.5, 1.5, 1.5, 1.5, 2])
        c1.markdown(f"**{horse}**<br><small>{row['Driver']}</small>", unsafe_allow_html=True)
        c2.markdown(f"`{prob*100:.1f}%`")
        c3.markdown(f"`{fair:.2f}`")

        mkt = c4.number_input(
            "", min_value=1.01, max_value=100.0, value=float(fair),
            step=0.05, format="%.2f",
            key=f"odds_{i}_{horse[:8]}",
            label_visibility="collapsed"
        )

        stake, raw_kelly_pct, edge_pct = calc_kelly(prob, mkt, bankroll, kelly_frac)

        if raw_kelly_pct <= 0:
            c5.markdown("❌ —")
            c6.markdown("*nincs érték*")
        else:
            c5.markdown(f"**{stake:,} Ft**")
            if edge_pct >= 15 and mkt <= 8.0:
                c6.markdown(f"🔥 **VALUE** `+{edge_pct:.1f}%`")
            elif edge_pct >= 5:
                c6.markdown(f"🟡 gyenge edge `+{edge_pct:.1f}%`")
            else:
                c6.markdown(f"⚪ `{edge_pct:+.1f}%`")

        st.divider()

    st.caption(f"Bankroll: **{bankroll:,} Ft** | Módszer: **{kelly_mode}** | Max tét/fogadás: 20%")



    # Explainability Section
    st.subheader("💡 Why these odds? (AI Explanation)")
    
    if model and explainer:
        top_horse_idx = rdf.index[0]
        top_horse_name = rdf.loc[top_horse_idx, "Horse"]
        
        top_p = selected_race["participants"][top_horse_idx]
        h_id = str(top_p["horse_id"])
        d_id = str(top_p["driver_id"])
        
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
        }
        
        model_features = list(feature_map.keys())
        
        try:
            race_dist = float(selected_race.get("distance", "1900").replace("A", "").replace("G", ""))
        except:
            race_dist = 1900.0

        feature_values = [
            race_dist, 0, 15.0,
            h_stats["win_rate"], h_stats["top_3_rate"], h_stats["avg_percentile"], h_stats["avg_speed"] or 12.8,
            h_stats.get("best_speed_life") or 12.8, h_stats.get("speed_ratio", 1.0),
            h_stats["total_prize"], h_stats["days_since_last"],
            h_stats["win_rate_l5"], h_stats["top_3_rate_l5"], h_stats["avg_percentile_l5"], h_stats["avg_speed_l5"] or 12.8,
            h_stats.get("points_l5", 0), h_stats.get("top3_l3", 0),
            d_stats["win_rate"], d_stats["top_3_rate"],
            0  # hd_pair_runs (unknown for today)
        ]
        
        shap_df = pd.DataFrame([feature_values], columns=model_features)
        try:
            shap_values = explainer.shap_values(shap_df)
            
            s_df = pd.DataFrame({
                "Feature": [feature_map[f] for f in model_features],
                "Impact Score": shap_values[0]
            }).sort_values("Impact Score", ascending=False)
            
            st.write(f"Top factor for **{top_horse_name}**:")
            fig_shap = px.bar(s_df, x="Impact Score", y="Feature", orientation='h',
                              title=f"Factor Impact for {top_horse_name}",
                              color="Impact Score",
                              color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as shap_err:
            st.info(f"SHAP explanation unavailable for this model version. ({shap_err})")
        
        st.markdown("""
        - **Green bars:** These factors increased the horse's win probability.
        - **Red bars:** These factors decreased the horse's win probability.
        """)
    else:
        st.info("No explainer loaded. Run train_model.py to generate SHAP explainer.")

def show_analytics_page():
    st.title("📈 Model Performance & Analytics")
    st.markdown("### Historical Backtest Analysis (2025 Test Set)")

    csv_path = "data/training_set_v3.csv" if os.path.exists("data/training_set_v3.csv") else "data/training_set_v2.csv"
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

    # V3 features (fallback to available subset)
    all_features = [
        "distance", "track_quality", "temperature",
        "h_win_rate", "h_top_3_rate", "h_avg_percentile", "h_avg_speed",
        "h_best_speed", "h_speed_ratio",
        "h_total_prize", "h_days_since",
        "h_win_rate_l5", "h_top_3_rate_l5", "h_avg_percentile_l5", "h_avg_speed_l5",
        "h_points_l5", "h_top3_l3",
        "d_win_rate", "d_top_3_rate",
        "hd_pair_runs",
    ]
    features = [f for f in all_features if f in test_df.columns]
    X_test = test_df[features].fillna(test_df[features].mean())
    y_test = test_df["win"]
    
    probs = model.predict_proba(X_test)[:, 1]
    test_df["ml_prob"] = probs

    # Metrics
    from sklearn.metrics import accuracy_score, brier_score_loss, precision_score
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
    st.write("How does Model V3 compare against simple betting strategies on the 2025 test set?")

    baseline_path = "data/baseline_results.csv"
    if os.path.exists(baseline_path):
        bl_df = pd.read_csv(baseline_path)
        
        # ROI bar chart
        colors = ["#ef553b" if r < 0 else "#00cc96" for r in bl_df["roi_pct"]]
        fig_bl = px.bar(
            bl_df, x="roi_pct", y="strategy", orientation="h",
            title="ROI by Strategy (2025 Test Set)",
            labels={"roi_pct": "ROI (%)", "strategy": "Strategy"},
            text="roi_pct",
            color="roi_pct",
            color_continuous_scale=["#ef553b", "#ffa15a", "#00cc96"],
            range_color=[-70, 40],
        )
        fig_bl.update_traces(texttemplate="%{text:+.1f}%", textposition="outside")
        fig_bl.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_bl.update_layout(coloraxis_showscale=False, showlegend=False,
                             yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bl, use_container_width=True)

        # Comparison table
        display_cols = {"strategy": "Strategy", "bets": "Bets", "pnl_ft": "P/L (Ft)",
                        "roi_pct": "ROI (%)", "hit_rate_pct": "Hit Rate (%)", "avg_odds": "Avg Odds"}
        bl_show = bl_df[[c for c in display_cols if c in bl_df.columns]].rename(columns=display_cols)
        st.dataframe(
            bl_show.style.format({"ROI (%)": "{:+.2f}", "P/L (Ft)": "{:+,d}"})
                         .background_gradient(subset=["ROI (%)"], cmap="RdYlGn"),
            use_container_width=True
        )
        
        st.caption("💡 Run `scripts/baseline_comparison.py` to refresh these results.")
    else:
        st.info("Baseline results not found. Run `scripts/baseline_comparison.py` to generate them.")

    # Real Money ROI Section
    st.markdown("---")
    st.subheader("💰 Real-World ROI Analysis (Lovi Odds)")
    st.write("Model V3 (Optuna + Isotonic Calibration) — backtested against actual starting odds from bet.lovi.hu.")
    
    sim_path = "data/simulation_results.csv"
    if os.path.exists(sim_path):
        import numpy as np
        sim_df = pd.read_csv(sim_path)
        
        # Strategy Controls
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            edge_req = st.slider("Required Edge (%)", 0, 40, 15, help="Min edge: Market Odds vs Fair Odds")
        with col_s2:
            max_odds = st.slider("Max Market Odds", 2.0, 30.0, 8.0, step=0.5, help="Avoid longshots — caps market odds")
        with col_s3:
            min_prob = st.slider("Min Win Prob (%)", 0, 50, 5, help="Only bet if AI win probability is above this")

        # Re-run simulation logic
        margin = edge_req / 100
        min_p = min_prob / 100
        
        sim_df['is_value_dynamic'] = (
            (sim_df['market_odds'] > sim_df['fair_odds'] * (1 + margin))
            & (sim_df['market_odds'] <= max_odds)
            & (sim_df['prob_norm'] > min_p)
        )
        
        stake = 1000
        sim_df['pnl_dynamic'] = np.where(sim_df['is_value_dynamic'], 
                                  np.where(sim_df['win'] == 1, (sim_df['market_odds'] - 1) * stake, -stake), 
                                  0)
        
        # Metrics
        total_bets = sim_df['is_value_dynamic'].sum()
        total_pnl = sim_df['pnl_dynamic'].sum()
        total_staked = total_bets * stake
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
        hits = sim_df[sim_df['is_value_dynamic']]['win'].sum() if total_bets > 0 else 0
        hit_rate = hits / total_bets * 100 if total_bets > 0 else 0
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Bets", int(total_bets))
        m2.metric("Total Staked", f"{total_staked:,.0f} Ft")
        m3.metric("Net P/L", f"{total_pnl:+,.0f} Ft")
        m4.metric("Real ROI", f"{roi:+.2f}%")
        m5.metric("Hit Rate", f"{hit_rate:.1f}%")
        
        # Cumulative P/L Chart
        actual_bets = sim_df[sim_df['is_value_dynamic']].copy()
        if not actual_bets.empty:
            actual_bets['cum_pnl'] = actual_bets['pnl_dynamic'].cumsum()
            actual_bets['bet_index'] = range(1, len(actual_bets) + 1)
            
            fig_pnl = px.line(actual_bets, x='bet_index', y='cum_pnl',
                              title=f"Cumulative P/L (Edge ≥{edge_req}%, MaxOdds ≤{max_odds})",
                              labels={"bet_index": "Bet Number", "cum_pnl": "Profit/Loss (Ft)"},
                              markers=True)
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            # Top Value Horses
            st.write("**Top Value Bets (Highest Edge):**")
            actual_bets['edge_pct'] = ((actual_bets['market_odds'] / actual_bets['fair_odds']) - 1) * 100
            top_value = actual_bets.sort_values('edge_pct', ascending=False).head(10)
            cols_show = [c for c in ['date', 'horse_name', 'market_odds', 'fair_odds', 'prob_norm', 'win', 'pnl_dynamic'] if c in top_value.columns]
            st.dataframe(top_value[cols_show].rename(columns={'pnl_dynamic': 'P/L (Ft)', 'prob_norm': 'AI Prob'}))
        else:
            st.warning("No bets match the selected criteria. Try lowering the edge or increasing max odds.")
    else:
        st.info("Simulation results not found. Run scripts/simulate_value_betting.py to see real ROI analysis.")

    # ── Bet-Sizing Analysis ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📐 Bet-Sizing Analysis (Kelly Criterion)")
    st.write("Comparing fixed stake vs dynamic bet-sizing. Starting bankroll: **100,000 Ft** | Max Odds: 8.0")

    bs_path    = "data/bet_sizing_results.csv"
    curve_path = "data/bet_sizing_curves.csv"

    if os.path.exists(bs_path) and os.path.exists(curve_path):
        import numpy as np
        bs_df    = pd.read_csv(bs_path)
        curve_df = pd.read_csv(curve_path)

        # Edge filter (matches the strategies available)
        edge_options = sorted(bs_df["edge_min_pct"].unique())
        sel_edge = st.select_slider(
            "Edge threshold for bankroll chart",
            options=[int(e) for e in edge_options],
            value=15
        )

        # Filter curves to selected edge
        filtered_curves = curve_df[curve_df["strategy"].str.contains(f"Edge {sel_edge}%")]

        if not filtered_curves.empty:
            fig_curves = px.line(
                filtered_curves, x="bet_num", y="bankroll", color="strategy",
                title=f"Bankroll Curve — Edge ≥{sel_edge}% | MaxOdds ≤8",
                labels={"bet_num": "Bet Number", "bankroll": "Bankroll (Ft)", "strategy": "Sizing Method"},
            )
            fig_curves.add_hline(y=100_000, line_dash="dash", line_color="white",
                                 opacity=0.4, annotation_text="Starting bankroll")
            st.plotly_chart(fig_curves, use_container_width=True)
        else:
            st.info("No curves for this edge threshold.")

        # Summary table filtered to selected edge
        bs_sel = bs_df[bs_df["edge_min_pct"] == sel_edge].copy()
        bs_disp = bs_sel[["name", "bets", "hit_rate_pct", "total_pnl",
                           "roi_on_staked_pct", "bank_growth_pct", "final_bank"]].rename(columns={
            "name": "Strategy", "bets": "Bets", "hit_rate_pct": "Hit %",
            "total_pnl": "P/L (Ft)", "roi_on_staked_pct": "ROI on Staked %",
            "bank_growth_pct": "Bankroll Growth %", "final_bank": "Final Bank (Ft)"
        })
        st.dataframe(
            bs_disp.style
                .format({"Bankroll Growth %": "{:+.2f}", "P/L (Ft)": "{:+,d}",
                         "Final Bank (Ft)": "{:,d}", "ROI on Staked %": "{:+.2f}"})
                .background_gradient(subset=["Bankroll Growth %"], cmap="RdYlGn"),
            use_container_width=True
        )

        # Best config callout
        best_row = bs_df.loc[bs_df["bank_growth_pct"].idxmax()]
        st.success(
            f"🏆 Best Config: **{best_row['name']}** @ Edge ≥{int(best_row['edge_min_pct'])}%  →  "
            f"Bankroll growth: **{best_row['bank_growth_pct']:+.2f}%** | "
            f"Final bank: **{int(best_row['final_bank']):,} Ft**"
        )
        st.caption("💡 Run `scripts/bet_sizing_comparison.py` to refresh. Kelly assumes calibrated win probabilities.")
    else:
        st.info("Bet-sizing results not found. Run `scripts/bet_sizing_comparison.py` to generate them.")

    # Calibration Plot
    st.markdown("---")
    st.subheader("🎯 Probability Calibration (V3 Isotonic)")
    cal_bins = pd.cut(test_df['ml_prob'], bins=10)
    cal_data = test_df.groupby(cal_bins, observed=False)['win'].agg(['mean', 'count']).reset_index()
    cal_data.columns = ['bin', 'actual_win_rate', 'count']
    cal_data['mid_prob'] = [iv.mid for iv in cal_data['bin']]
    
    fig_cal = px.scatter(cal_data, x='mid_prob', y='actual_win_rate', size='count',
                         title="Reliability Diagram (Calibration Plot)",
                         labels={"mid_prob": "Predicted Probability", "actual_win_rate": "Actual Win Rate"})
    fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash"))
    fig_cal.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig_cal, use_container_width=True)
    st.caption("Dots close to the red diagonal = well-calibrated. Larger dots = more data in that bin.")

    # Feature Importance
    st.subheader("📊 Global Feature Importance")
    try:
        # CalibratedClassifierCV — extract inner model
        inner = model.calibrated_classifiers_[0].estimator
        importance_vals = inner.feature_importances_
    except:
        try:
            importance_vals = model.feature_importances_
        except:
            importance_vals = None

    if importance_vals is not None:
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': importance_vals[:len(features)]
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(importances, x="Importance", y="Feature", orientation='h',
                         title="XGBoost Feature Gain Importance (V3 Model)",
                         color="Importance",
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

    # Probability Distribution
    st.subheader("📉 Probability Distribution")
    fig_dist = px.histogram(test_df, x="ml_prob", color="win", barmode="overlay",
                            title="ML Probability vs Actual Win (V3 Calibrated)",
                            labels={"ml_prob": "Predicted Probability", "win": "Actually Won"},
                            color_discrete_map={0: "#ef553b", 1: "#00cc96"})
    st.plotly_chart(fig_dist, use_container_width=True)

if page == "Daily Predictions":
    show_predictions_page()
elif page == "Model Analytics":
    show_analytics_page()

# Footer
st.markdown("---")
st.caption("Data provided by Kincsem Park. Model V3 (Calibrated XGBoost). Predictions are for educational purposes only.")

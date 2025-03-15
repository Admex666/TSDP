import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import glob
import os
from datetime import datetime, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Streamlit oldal konfigurálása
st.set_page_config(
    page_title="⚽ Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Cache funkcióval betöltjük a modelleket
@st.cache_data
def load_models():
    """
    Modellek betöltése és cache-elése
    """
    try:
        # Betöltjük a fő artifact file-t
        model_files = glob.glob('models/football_model_*_artifacts.pkl')
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            with open(latest_model, 'rb') as f:
                model_artifacts = pickle.load(f)
            
            models = model_artifacts['models']
            scaler = model_artifacts['scaler']
            feature_columns = model_artifacts['feature_columns']
            
            return models, scaler, feature_columns, True
        else:
            return None, None, None, False
            
    except Exception as e:
        return None, None, None, False

def prepare_new_data(match_data, feature_columns, scaler):
    """
    Előkészíti az új mérkőzés adatokat predikcióra
    """
    # Alap feature engineering
    df = pd.DataFrame([match_data])
    
    # Data cleaning
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Implied probabilities
    if all(col in match_data for col in ['AvgH', 'AvgD', 'AvgA']):
        df['Home_Implied_Prob'] = 1 / np.clip(df['AvgH'], 1.01, 1000)
        df['Draw_Implied_Prob'] = 1 / np.clip(df['AvgD'], 1.01, 1000)
        df['Away_Implied_Prob'] = 1 / np.clip(df['AvgA'], 1.01, 1000)
    
    # Hiányzó értékek kezelése
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Default érték
    
    # Csak a szükséges oszlopok
    X_new = df[feature_columns]
    
    # Skálázás
    X_scaled = scaler.transform(X_new)
    
    return X_scaled, df

def predict_match(models, X_new, odds_data=None, threshold=0.05):
    """
    Predikció készítése az összes modellel
    """
    predictions = {}
    
    for model_name, model in models.items():
        try:
            # Valószínűségek predikálása
            probs = model.predict_proba(X_new)[0]  # Csak az első mérkőzés
            pred_class = model.predict(X_new)[0]
            
            # Value bet számítás
            value_bets = {}
            if odds_data is not None:
                home_implied = 1 / odds_data['AvgH']
                draw_implied = 1 / odds_data['AvgD'] 
                away_implied = 1 / odds_data['AvgA']
                
                value_bets = {
                    'Home_Value': probs[0] > home_implied + threshold,
                    'Draw_Value': probs[1] > draw_implied + threshold,
                    'Away_Value': probs[2] > away_implied + threshold
                }
            
            predictions[model_name] = {
                'probabilities': probs,
                'predicted_class': pred_class,
                'value_bets': value_bets,
                'recommendation': get_recommendation(probs, value_bets, odds_data)
            }
            
        except Exception as e:
            st.error(f"Hiba {model_name} predikciójánál: {e}")
            predictions[model_name] = None
    
    return predictions

def get_recommendation(probs, value_bets, odds_data):
    """
    Ajánlás generálása a predikció alapján
    """
    if not value_bets:
        return "Nincs elég adat az ajánláshoz"
    
    recommendations = []
    outcomes = ['Home', 'Draw', 'Away']
    
    for i, outcome in enumerate(outcomes):
        if value_bets[f'{outcome}_Value']:
            recommendations.append(f"{outcome} win: {probs[i]:.1%} valószínűség")
    
    return " | ".join(recommendations) if recommendations else "Nincs érték fogadás"

def create_probability_chart(predictions):
    """
    Valószínűségek összehasonlító chart
    """
    fig = go.Figure()
    
    outcomes = ['Home', 'Draw', 'Away']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, outcome in enumerate(outcomes):
        model_names = []
        probabilities = []
        
        for model_name, pred in predictions.items():
            if pred is not None:
                model_names.append(model_name)
                probabilities.append(pred['probabilities'][i] * 100)
        
        fig.add_trace(go.Bar(
            name=outcome,
            x=model_names,
            y=probabilities,
            marker_color=colors[i],
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Models',
        yaxis_title='Probability (%)',
        barmode='group',
        height=400
    )
    
    return fig

def create_odds_comparison_chart(predictions, market_odds):
    """
    Fair odds vs Market odds összehasonlítás
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Home Win', 'Draw', 'Away Win'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    outcomes = ['Home', 'Draw', 'Away']
    market_odds_values = [market_odds['AvgH'], market_odds['AvgD'], market_odds['AvgA']]
    
    for i, outcome in enumerate(outcomes):
        model_names = []
        fair_odds = []
        
        for model_name, pred in predictions.items():
            if pred is not None:
                model_names.append(model_name)
                fair_odds.append(1 / pred['probabilities'][i])
        
        # Fair odds
        fig.add_trace(
            go.Bar(name='Fair Odds', x=model_names, y=fair_odds, 
                   marker_color='lightblue', showlegend=(i==0)),
            row=1, col=i+1
        )
        
        # Market odds (horizontal line)
        fig.add_hline(y=market_odds_values[i], line_dash="dash", 
                      line_color="red", row=1, col=i+1,
                      annotation_text=f"Market: {market_odds_values[i]:.2f}")
    
    fig.update_layout(height=400, title_text="Fair Odds vs Market Odds")
    return fig

# Főoldal
st.title("⚽ Football Match Prediction System")
st.markdown("---")

# Modellek betöltése
models, scaler, feature_columns, models_loaded = load_models()

if not models_loaded:
    st.error("❌ Nem sikerült betölteni a modelleket! Ellenőrizd, hogy létezik-e a 'models/' mappa a megfelelő fájlokkal.")
    st.stop()

st.success("✅ Modellek sikeresen betöltve!")
st.info(f"📊 Elérhető modellek: {', '.join(models.keys())}")

# Sidebar a beviteli mezőkkel
with st.sidebar:
    st.header("📝 Match Details")
    
    # Alapadatok
    st.subheader("Basic Info")
    match_date = st.date_input("Match Date", value=date.today())
    home_team = st.text_input("Home Team", value="Manchester United")
    away_team = st.text_input("Away Team", value="Liverpool")
    
    # Odds
    st.subheader("📈 Market Odds")
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_h = st.number_input("Home Win", min_value=1.01, value=2.50, step=0.01)
    with col2:
        avg_d = st.number_input("Draw", min_value=1.01, value=3.20, step=0.01)
    with col3:
        avg_a = st.number_input("Away Win", min_value=1.01, value=2.80, step=0.01)
    
    # Forma adatok
    st.subheader("📊 Recent Form (Last 5 matches)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Home Team**")
        home_goals_for_last_5 = st.slider("Goals scored per game", 0.0, 5.0, 1.4, 0.1, key="home_goals_for")
        home_goals_against_last_5 = st.slider("Goals conceded per game", 0.0, 5.0, 1.2, 0.1, key="home_goals_against")
        home_form_last_5 = st.slider("Total points", 0, 15, 8, key="home_form")
        home_points_last_5 = home_form_last_5 / 5

    with col2:
        st.markdown("**Away Team**")
        away_goals_for_last_5 = st.slider("Goals scored per game", 0.0, 5.0, 2.1, 0.1, key="away_goals_for")
        away_goals_against_last_5 = st.slider("Goals conceded per game", 0.0, 5.0, 0.8, 0.1, key="away_goals_against")
        away_form_last_5 = st.slider("Total points", 0, 15, 12, key="away_form")
        away_points_last_5 = away_form_last_5 / 5

    # További paraméterek
    st.subheader("⚙️ Additional Parameters")
    home_advantage = st.selectbox("Home Advantage", [0, 1], index=1)
    days_since_last_home = st.number_input("Days since last home match", min_value=1, value=7)
    days_since_last_away = st.number_input("Days since last away match", min_value=1, value=6)
    
    # Value bet threshold
    st.subheader("💰 Value Betting")
    threshold = st.slider("Value bet threshold", 0.01, 0.20, 0.05, 0.01)

# Predikció gomb
if st.button("🔮 Predict Match", type="primary"):
    # Match adatok összeállítása
    match_data = {
        'Date': match_date.strftime('%Y-%m-%d'),
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'AvgH': avg_h,
        'AvgD': avg_d,
        'AvgA': avg_a,
        'Home_Points_Last_5': home_points_last_5,
        'Away_Points_Last_5': away_points_last_5,
        'Home_Goals_For_Last_5': home_goals_for_last_5,
        'Away_Goals_For_Last_5': away_goals_for_last_5,
        'Home_Goals_Against_Last_5': home_goals_against_last_5,
        'Away_Goals_Against_Last_5': away_goals_against_last_5,
        'Home_Form_Last_5': home_form_last_5,
        'Away_Form_Last_5': away_form_last_5,
        'Home_Advantage': home_advantage,
        'Days_Since_Last_Home_Match': days_since_last_home,
        'Days_Since_Last_Away_Match': days_since_last_away
    }
    
    # Adatok előkészítése
    X_new, prepared_df = prepare_new_data(match_data, feature_columns, scaler)
    
    # Odds adatok
    odds_data = {
        'AvgH': avg_h,
        'AvgD': avg_d, 
        'AvgA': avg_a
    }
    
    # Predikció készítése
    predictions = predict_match(models, X_new, odds_data, threshold)
    
    # Eredmények megjelenítése
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    # Match info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏠 Home Team", home_team)
    with col2:
        st.metric("📅 Date", match_date.strftime('%Y-%m-%d'))
    with col3:
        st.metric("✈️ Away Team", away_team)
    
    # Market odds display
    st.subheader("📊 Market Odds")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Home Win", f"{avg_h:.2f}")
    with col2:
        st.metric("Draw", f"{avg_d:.2f}")
    with col3:
        st.metric("Away Win", f"{avg_a:.2f}")
    
    # Model predictions
    st.subheader("🤖 Model Predictions")
    
    for model_name, pred in predictions.items():
        if pred is None:
            continue
        
        with st.expander(f"📈 {model_name} Results", expanded=True):
            # Probabilities
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "🏠 Home Win", 
                    f"{pred['probabilities'][0]:.1%}",
                    delta=f"Fair odds: {1/pred['probabilities'][0]:.2f}"
                )
            with col2:
                st.metric(
                    "🤝 Draw", 
                    f"{pred['probabilities'][1]:.1%}",
                    delta=f"Fair odds: {1/pred['probabilities'][1]:.2f}"
                )
            with col3:
                st.metric(
                    "✈️ Away Win", 
                    f"{pred['probabilities'][2]:.1%}",
                    delta=f"Fair odds: {1/pred['probabilities'][2]:.2f}"
                )
            
            # Value bets
            if any(pred['value_bets'].values()):
                st.success(f"💰 **Value Bets Found:** {pred['recommendation']}")
                value_outcomes = [k.replace('_Value', '') for k, v in pred['value_bets'].items() if v]
                st.info(f"🎯 Recommended bets: {', '.join(value_outcomes)}")
            else:
                st.warning("⚠️ No value bets identified")
    
    # Charts
    st.subheader("📊 Visual Analysis")
    
    # Probability comparison chart
    prob_chart = create_probability_chart(predictions)
    st.plotly_chart(prob_chart, use_container_width=True)
    
    # Odds comparison chart
    odds_chart = create_odds_comparison_chart(predictions, odds_data)
    st.plotly_chart(odds_chart, use_container_width=True)
    
    # Summary table
    st.subheader("📋 Summary Table")
    summary_data = []
    
    for model_name, pred in predictions.items():
        if pred is not None:
            summary_data.append({
                'Model': model_name,
                'Home %': f"{pred['probabilities'][0]:.1%}",
                'Draw %': f"{pred['probabilities'][1]:.1%}",
                'Away %': f"{pred['probabilities'][2]:.1%}",
                'Predicted': ['Home', 'Draw', 'Away'][pred['predicted_class']],
                'Value Bets': ', '.join([k.replace('_Value', '') for k, v in pred['value_bets'].items() if v]) or 'None'
            })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("⚽ **Football Match Predictor** | Powered by Machine Learning")
st.markdown("💡 *Tip: The app automatically calculates points per game from total points (Total Points ÷ 5)*")
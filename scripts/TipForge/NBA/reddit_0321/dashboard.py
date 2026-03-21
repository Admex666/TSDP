import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Styling & Config ---
st.set_page_config(layout="wide", page_title="NBA Betting Dashboard", page_icon="🏀")

st.markdown("""
<style>
.stDataFrame { font-size: 13px; }
h1, h2, h3 { color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

st.title("🏀 NBA Betting Dashboard - Moneyline & Value Edge")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\full_backtest_results.csv")
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

df = load_data()

# --- Sidebar Controls ---
st.sidebar.title("Configuration")
sizing_strategy = st.sidebar.selectbox(
    "Bet Sizing Strategy",
    ["Flat (1 Unit)", "1/4 Kelly", "Full Kelly", "Confidence Tiers"]
)

# --- Data Preparation ---
df['p'] = np.where(df['bet_placed'] == 'Home', df['pred_prob_home'], 1 - df['pred_prob_home'])
df['bet_odds'] = np.where(df['bet_placed'] == 'Home', df['odds_home'], df['odds_away'])
df['implied'] = 1 / df['bet_odds']
df['edge_val'] = df['p'] - df['implied']
df['b'] = df['bet_odds'] - 1
df['q'] = 1 - df['p']

# Kelly calculations
df['kelly_units'] = ((df['p'] * df['b'] - df['q']) / df['b']) * 100
df['kelly_units'] = df['kelly_units'].apply(lambda x: max(x, 0)) # Prevent negative bets just in case edge is tight
conditions = [df['edge_val'] > 0.07, (df['edge_val'] >= 0.04) & (df['edge_val'] <= 0.07), df['edge_val'] < 0.04]
df['tier_risk'] = np.select(conditions, [3.0, 2.0, 1.0], default=1.0)

if sizing_strategy == "Flat (1 Unit)":
    df['risk_units'] = 1.0
elif sizing_strategy == "1/4 Kelly":
    df['risk_units'] = df['kelly_units'] / 4
elif sizing_strategy == "Full Kelly":
    df['risk_units'] = df['kelly_units']
elif sizing_strategy == "Confidence Tiers":
    df['risk_units'] = df['tier_risk']

# Force No Bet to 0 risk
df['risk_units'] = np.where(df['bet_placed'] == 'No Bet', 0.0, df['risk_units'])

# Recalculate profit using the new sizing rules
df['profit'] = np.where(
    df['bet_placed'] != 'No Bet',
    np.where(df['won_ML'] == 1, df['risk_units'] * df['b'], -df['risk_units']),
    0.0
)

# --- KPI Metrics ---
bets_only = df[df['bet_placed'] != 'No Bet']
total_bets = len(bets_only)
win_rate = bets_only['won_ML'].sum() / total_bets if total_bets > 0 else 0
total_risked = bets_only['risk_units'].sum()
total_profit = bets_only['profit'].sum()
overall_roi = total_profit / total_risked if total_risked > 0 else 0

st.markdown("### 🏆 Overall Performance")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Bets", f"{total_bets}")
kpi2.metric("Win Rate", f"{win_rate:.1%}")
kpi3.metric("Units Risked", f"{total_risked:.2f} U")
kpi4.metric("Total Profit", f"{total_profit:.2f} U", delta=f"{total_profit:.2f} U")
kpi5.metric("Overall ROI", f"{overall_roi:.2%}")

st.markdown("---")

# -----------------
# Top Section: Selections
# -----------------
st.subheader("Recent Selections & Results")

# Format display dataframe
disp_df = df[['GAME_DATE', 'Away_Team', 'Home_Team', 'odds_away', 'odds_home', 'pred_prob_home', 'bet_placed', 'risk_units', 'profit']].copy()
disp_df['Date'] = disp_df['GAME_DATE'].dt.strftime('%m/%d/%y')

# Calculate the model's perceived edge
disp_df['Edge'] = np.where(disp_df['bet_placed'] == 'Home', disp_df['pred_prob_home'] - (1/disp_df['odds_home']),
                  np.where(disp_df['bet_placed'] == 'Away', (1-disp_df['pred_prob_home']) - (1/disp_df['odds_away']), 0))

# Take most recent 50 games for the view panel
disp_df = disp_df.sort_values('GAME_DATE', ascending=False).drop(columns=['GAME_DATE']).head(50)
disp_df = disp_df[['Date', 'Away_Team', 'Home_Team', 'odds_away', 'odds_home', 'pred_prob_home', 'Edge', 'bet_placed', 'risk_units', 'profit']]

def style_profit(val):
    if val > 0:
        return 'color: #2ecc71; font-weight: bold'
    elif val < 0:
        return 'color: #e74c3c; font-weight: bold'
    return 'color: gray'

st.dataframe(
    disp_df.style.map(style_profit, subset=['profit'])\
           .format({'odds_away': '{:.2f}', 'odds_home': '{:.2f}', 'pred_prob_home': '{:.1%}', 'Edge': '{:.1%}', 'risk_units': '{:.2f} U', 'profit': '{:.2f} U'}),
    use_container_width=True,
    height=300
)

# -----------------
# Middle Section: Summaries
# -----------------
st.markdown("---")
col1, col2, col3 = st.columns(3)

bets_df = df[df['bet_placed'] != 'No Bet'].copy()
bets_df['Month'] = bets_df['GAME_DATE'].dt.strftime('%Y - %B')

with col1:
    st.subheader("Summary Results by Month")
    if not bets_df.empty:
        monthly = bets_df.groupby('Month').agg(
            Bets=('profit', 'count'),
            Units_Risked=('risk_units', 'sum'),
            Wins=('won_ML', 'sum'),
            Total_Profit=('profit', 'sum')
        ).reset_index()
        monthly['Win Rate'] = (monthly['Wins'] / monthly['Bets'])
        monthly['ROI'] = (monthly['Total_Profit'] / monthly['Units_Risked'])
        
        # Style ROI red/green
        st.dataframe(
            monthly.style.map(style_profit, subset=['ROI', 'Total_Profit'])\
                   .format({'Win Rate': '{:.1%}', 'Units_Risked': '{:.2f}', 'Total_Profit': '{:.2f} U', 'ROI': '{:.1%}'}),
            use_container_width=True
        )

with col2:
    st.subheader("Moneyline Wager by Odds")
    if not bets_df.empty:
        # Determine odds of the placed bet
        bets_df['bet_odds'] = np.where(bets_df['bet_placed'] == 'Home', bets_df['odds_home'], bets_df['odds_away'])
        # Group odds into tiers similar to Excel image
        bins = [1.0, 1.50, 1.80, 2.20, 3.0, 100.0]
        labels = ['1.0 - 1.5', '1.5 - 1.8', '1.8 - 2.2', '2.2 - 3.0', '3.0+']
        bets_df['Odds_Group'] = pd.cut(bets_df['bet_odds'], bins=bins, labels=labels)
        
        odds_grp = bets_df.groupby('Odds_Group').agg(
            Bets=('profit', 'count'),
            Units_Risked=('risk_units', 'sum'),
            Wins=('won_ML', 'sum'),
            Total_Profit=('profit', 'sum')
        ).reset_index()
        odds_grp['Win Rate'] = (odds_grp['Wins'] / odds_grp['Bets']).fillna(0)
        odds_grp['ROI'] = (odds_grp['Total_Profit'] / odds_grp['Units_Risked']).fillna(0)
        
        st.dataframe(
            odds_grp.style.map(style_profit, subset=['ROI', 'Total_Profit'])\
                    .format({'Win Rate': '{:.1%}', 'Units_Risked': '{:.2f}', 'Total_Profit': '{:.2f} U', 'ROI': '{:.1%}'}),
            use_container_width=True
        )

with col3:
    st.subheader("Daily Results (Last 10 Days)")
    if not bets_df.empty:
        daily = bets_df.groupby(bets_df['GAME_DATE'].dt.strftime('%m/%d/%y')).agg(
            Bets=('profit', 'count'),
            Wins=('won_ML', 'sum'),
            Profit=('profit', 'sum')
        ).reset_index().rename(columns={'GAME_DATE': 'Date'})
        daily['Win Rate'] = daily['Wins'] / daily['Bets']
        daily = daily.sort_values('Date', ascending=False).head(10)
        
        st.dataframe(
            daily.style.map(style_profit, subset=['Profit'])\
                 .format({'Win Rate': '{:.1%}', 'Profit': '{:.2f} U'}),
            use_container_width=True
        )

# -----------------
# Bottom Section: Bar Charts
# -----------------
st.markdown("---")
st.subheader("Moneyline Win Rate by Team")

# Calculate overall team win rates
teams = pd.concat([df['Home_Team'], df['Away_Team']]).unique()
team_stats = []
for t in teams:
    if t == "UNK": continue
    home_g = df[df['Home_Team'] == t]
    away_g = df[df['Away_Team'] == t]
    wins = (home_g['home_win'] == 1).sum() + (away_g['home_win'] == 0).sum()
    total = len(home_g) + len(away_g)
    if total > 0:
        team_stats.append({'Team': t, 'Win Rate': wins/total, 'Wins': wins, 'Losses': total - wins})

team_df = pd.DataFrame(team_stats).sort_values('Win Rate', ascending=False)

# Recreate the stacked bar chart visually matching the Excel sheet
fig = px.bar(team_df, x='Team', y=['Wins', 'Losses'], title='Team Overall Wins vs Losses',
             color_discrete_map={'Wins': '#2ecc71', 'Losses': '#e74c3c'}, barmode='stack')
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
st.plotly_chart(fig, use_container_width=True)

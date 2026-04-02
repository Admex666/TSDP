You are a Python coding agent. Your task is to systematically gather, clean, analyze, and backtest historical Betfair SP data for horse and greyhound racing. Follow these steps:

1️⃣ DATA COLLECTION:
- Access the Betfair SP historical archive at https://promo.betfair.com/betfairsp/prices
- Download all relevant CSV files (UK, Ireland, Australia, France, USA, South Africa, greyhound and horse racing; ignore other empty files)
- Maintain file metadata (date, filename) for reference
- Store CSVs locally for further processing

2️⃣ DATA CLEANING & STRUCTURING:
- Read each CSV into pandas DataFrames
- Standardize column names (lowercase, underscores)
- Ensure correct data types: 
    - floats for odds and volume
    - datetime for event_dt
    - integers for win_lose
- Combine all files into a single master DataFrame
- Create unique event identifiers by combining event_id and event_dt if necessary
- Remove duplicates and malformed rows

3️⃣ FEATURE ENGINEERING:
- Calculate implied probabilities: `implied_prob = 1 / bsp`
- Compute early vs pre-off odds drift: `odds_change = bsp - morningwap`
- Compute volume features:
    - early_volume = morningtradedvol
    - preplay_volume = pptradedvol
    - inplay_volume = iptradedvol
    - late_money_ratio = preplay_volume / (early_volume + 1e-6)
- Calculate Closing Line Value (CLV): `clv = (1 / bsp) - (1 / morningwap)`
- Flag longshots: odds > 15

4️⃣ STRATEGY SIMULATION / BACKTEST:
- Define simple strategies:
    1. Fade drifters: bet against horses where odds increase significantly with low early volume
    2. Follow steam: bet on horses where odds decrease significantly with high early volume
    3. Longshot bias: back horses with odds > 15, lay horses with odds < 3
- Assume flat staking (1 unit per bet)
- Track performance metrics:
    - ROI (total P/L / total stake)
    - Average CLV per strategy
    - Win rate
- Include volume-weighted ROI if possible

5️⃣ VISUALIZATION & REPORTING:
- Plot distributions of BSP vs morningwap, odds change, CLV
- Plot volume distribution (early, pre-off, in-play)
- Summary table of strategy performance: ROI, CLV, win rate, number of bets
- Highlight top 3 strategies by ROI

6️⃣ OUTPUT:
- Save the cleaned master dataset with computed features
- Save backtest results as CSV
- Save all plots as PNG
- Generate a concise markdown report summarizing the findings

Additional requirements:
- Code must be modular, with clear functions for each step
- Include logging for each major process (download, clean, feature, backtest)
- Handle missing data gracefully
- All plots should have titles, axis labels, and legends

End of instructions.
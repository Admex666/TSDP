# ⚽ Match Report Generator

A comprehensive pre-match analysis tool using SofaScore API data. Generate detailed tactical reports for upcoming football matches with advanced metrics, visualizations, and predictions.

## Features

### 📊 7 Comprehensive Analysis Tabs

1. **Overview** - Quick match snapshot with radar comparison and key statistics
2. **Team Analysis** - Deep dive into playing styles, attacking/defending metrics, and shot quality
3. **Set Pieces** - Corners, free kicks, and penalties analysis
4. **Key Players** - Top scorers, assisters, and best-rated players
5. **Form & Trends** - Recent results and performance trends
6. **Match Prediction** - Statistical predictions and key battles
7. **Tactical Analysis** - Shot maps and tactical visualizations

### 🎯 Advanced Metrics

- **xG per shot** - Shot quality indicator
- **Shot location breakdown** - Inside/outside box, zones, methods
- **Playing style indicators** - Possession, direct play, wing orientation
- **Defensive metrics** - Clean sheets, tackles, duels won
- **Set piece efficiency** - Conversion rates and tendencies
- **Form analysis** - Last 5-10 matches with goals and results

### 📈 Rich Visualizations

- Interactive Plotly radar charts
- Form trend lines
- Football pitch shot maps (mplsoccer)
- Player performance bars
- Set piece comparison charts

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify SofaScore module:**
Make sure the `SofaScore_module.py` is in the `modules` directory with the enhanced anti-403 protection.

## Usage

1. **Run the Streamlit app:**
```bash
streamlit run app.py
```

2. **Select a match:**
   - Choose a league from the sidebar
   - Select a round number
   - Pick a match from the dropdown
   - Click "Generate Report"

3. **Explore the analysis:**
   - Navigate through the 7 tabs
   - View interactive charts and statistics
   - Get tactical insights and predictions

## Supported Leagues

- **Premier League** (England)
- **La Liga** (Spain)
- **Bundesliga** (Germany)
- **Serie A** (Italy)
- **Ligue 1** (France)
- **Champions League** (Europe)

## Project Structure

```
match_report/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── config/
│   ├── __init__.py
│   └── leagues.py             # League configurations
├── modules/
│   ├── __init__.py
│   ├── data_fetcher.py        # API data retrieval
│   ├── metrics_calculator.py  # Metrics calculation
│   └── visualizations.py      # Chart generation
└── README.md
```

## Data Sources

All data is fetched from the **SofaScore public API** including:
- Team season statistics (120+ metrics)
- Recent match results and form
- Top players by category
- Shot maps from past matches
- Match-specific data

## Key Metrics Explained

### xG per Shot
- Measures shot quality
- \>0.12 = High quality
- 0.08-0.12 = Medium quality
- <0.08 = Low quality

### Shot Locations
- **Inside Box**: Shots from penalty area (>83 on 0-100 scale)
- **6-Yard Box**: Close-range shots (>94 on 0-100 scale)
- **Central**: Shots from central corridor (37-63 Y-axis)

### Playing Styles
- **High Possession**: >55% average possession
- **Direct Play**: >10% long balls
- **Wing Oriented**: >5% crosses
- **Speculative Shooting**: >30% shots from outside box

## Notes

- All statistics are **per 90 minutes** unless stated otherwise
- Shot maps aggregate data from the last 10 matches
- xG data is available for most major leagues
- Reports are best for upcoming matches (not live or completed)

## Troubleshooting

### 403 Errors
The SofaScore module includes advanced anti-403 protection:
- User-agent rotation
- Exponential backoff retry
- Rate limiting
- Session management

If you still encounter issues:
1. Wait a few minutes between requests
2. Try a different league/match
3. Check your internet connection

### No Data Available
Some matches may not have complete data:
- Very recent matches might not have shot maps yet
- Lower-tier leagues may have limited statistics
- Future matches won't have historical match data

## Future Enhancements

Potential additions:
- Pass network visualizations (requires match-specific data)
- Head-to-head history
- Injury/suspension tracking
- Export to PDF
- Custom league additions
- Historical match analysis

## Credits

- **Data Source**: SofaScore
- **Visualizations**: Plotly, Matplotlib, mplsoccer
- **Framework**: Streamlit

---

**Enjoy your match analysis! ⚽**

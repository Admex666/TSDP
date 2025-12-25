# Football Match Prediction & Betting Backtest

## Overview
This project replicates the methodology of [arXiv:2210.06327](https://arxiv.org/abs/2210.06327) ("Betting the system: Using lineups to predict football scores").
It builds a production-quality pipeline to predict football scores using lineup-based features from FBref, trains SVR models, models probabilities, and backtests a value betting strategy.

## Key Features
- **Data Source**: FBref (via custom scraper/loader)
- **Features**: Time-aware, lineup-aggregrated metrics (GK, DEF, MID, ATT) avoiding data leakage.
- **Models**: Support Vector Regression (SVR) for Home/Away goals.
- **Betting**: Poisson-based probability modeling and value betting execution.

## Project Structure
- `data/`: Raw and processed data storage.
- `src/`: Source code modules.
    - `scraping/`: Data ingestion from FBref.
    - `features/`: Feature engineering logic.
    - `models/`: ML models (SVR, Probability).
    - `betting/`: Betting strategy and backtesting.
- `notebooks/`: Analysis and verification notebooks.
- `configs/`: Configuration files.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main pipeline:
   ```bash
   python main.py
   ```

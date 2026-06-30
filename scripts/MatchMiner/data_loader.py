import pandas as pd
import numpy as np
from config import TEAM_METRICS, PLAYER_METRICS

def clean_percentage(val):
    """Helper to clean percentage strings and other numeric issues from Opta CSVs."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str == '-' or val_str == '' or val_str.lower() == 'nan':
        return np.nan
    if val_str.endswith('%'):
        val_str = val_str[:-1]
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def load_team_data(filepath):
    """Loads and cleans team match-level data from Opta Provision CSV."""
    df = pd.read_csv(filepath)
    
    # Filter rows with missing critical match keys
    df = df.dropna(subset=['gameId', 'Date'])
    
    # Clean date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Identify numeric columns that we need to clean/convert
    cols_to_clean = set(TEAM_METRICS.values()).union({'score', 'finalScoreOpponent'})
    
    for col in df.columns:
        if col in cols_to_clean:
            df[col] = df[col].apply(clean_percentage)
            
    return df

def load_player_data(filepath):
    """Loads and cleans player match-level data from Opta Provision CSV."""
    df = pd.read_csv(filepath)
    
    # Filter rows with missing critical keys
    df = df.dropna(subset=['gameId', 'Date', 'playerId'])
    
    # Clean date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Identify numeric columns to clean
    cols_to_clean = set(PLAYER_METRICS.values()).union({'Min', 'Age', 'score', 'finalScoreOpponent'})
    
    for col in df.columns:
        if col in cols_to_clean:
            df[col] = df[col].apply(clean_percentage)
            
    # Calculate derived field: TakeOnSuccess = TakeOn - TakeOnFail
    # Fill NaN values in TakeOn and TakeOnFail with 0 for subtraction, but preserve NaN if both are empty
    df['TakeOnSuccess'] = np.where(
        df['TakeOn'].isna() & df['TakeOnFail'].isna(),
        np.nan,
        df['TakeOn'].fillna(0) - df['TakeOnFail'].fillna(0)
    )
    
    return df

# data_loader.py
import pandas as pd
import pickle
import glob
import os
from config import MODEL_PATH, FUZZ_PATH

def load_models():
    """Modellek betöltése"""
    try:
        model_files = glob.glob(f'{MODEL_PATH}football_model_*_artifacts.pkl')
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            with open(latest_model, 'rb') as f:
                model_artifacts = pickle.load(f)
            return model_artifacts['models'], model_artifacts['scaler'], model_artifacts['feature_columns'], True
        return None, None, None, False
    except Exception as e:
        print(f"Model load error: {e}")
        return None, None, None, False

def load_league_data(season, league_code):
    """Ligaadatok betöltése"""
    try:
        url = f'https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv'
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"League data load error: {e}")
        return None

def load_fuzz_data():
    """Csapatnév mapping betöltése"""
    try:
        return pd.read_excel(FUZZ_PATH)
    except Exception as e:
        print(f"Fuzz data load error: {e}")
        return None
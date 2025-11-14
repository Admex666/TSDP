# data_loader.py
import pandas as pd
import pickle
import glob
import os
from config import MODEL_PATH, FUZZ_PATH

def load_models():
    """Modellek betöltése"""
    try:
        # Abszolút útvonal használata
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        model_files = glob.glob(os.path.join(model_path, 'football_model_*_artifacts.pkl'))
        
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            with open(latest_model, 'rb') as f:
                model_artifacts = pickle.load(f)
            return model_artifacts['models'], model_artifacts['scaler'], model_artifacts['feature_columns'], True
        else:
            print(f"Nem találhatóak model fájlok a következő útvonalon: {model_path}")
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
        # Abszolút útvonal használata
        fuzz_path = os.path.join(os.path.dirname(__file__), FUZZ_PATH)
        return pd.read_excel(fuzz_path)
    except Exception as e:
        print(f"Fuzz data load error: {e}")
        return None
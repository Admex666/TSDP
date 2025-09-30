import os 
import pickle

# MODELL BETÖLTÉSE
def load_latest_model():
    """Legfrissebb modell betöltése"""
    models_dir = "Tennis/models"
    
    if not os.path.exists(models_dir):
        print(f"❌ Models mappa nem található: {models_dir}")
        return None
    
    # Összes .pkl fájl keresése
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print(f"❌ Nincs mentett modell a {models_dir} mappában")
        return None
    
    # Legfrissebb fájl kiválasztása
    latest_file = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_file)
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✅ Modell betöltve: {latest_file}")
        return model_data
    except Exception as e:
        print(f"❌ Hiba a modell betöltésekor: {str(e)}")
        return None
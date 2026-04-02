import os
import pandas as pd
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def load_and_clean_data():
    all_files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    logging.info(f"Found {len(all_files)} CSV files to process.")
    
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Ensure minimum required columns exist
            required_cols = ['event_id', 'event_dt', 'selection_id', 'win_lose', 
                             'bsp', 'morningwap', 'morningtradedvol', 'pptradedvol', 'iptradedvol']
            
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logging.warning(f"File {file} missing columns: {missing}. Skipping.")
                continue
                
            df_list.append(df)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            
    if not df_list:
        logging.error("No valid data loaded!")
        return None
        
    master_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined data into master DataFrame with {len(master_df)} rows.")
    
    # Duplicates and malformed
    master_df['event_dt'] = pd.to_datetime(master_df['event_dt'], errors='coerce')
    master_df = master_df.dropna(subset=['event_dt', 'bsp', 'win_lose'])
    master_df['win_lose'] = master_df['win_lose'].astype(int)
    master_df = master_df.drop_duplicates(subset=['event_id', 'selection_id'])
    
    # Ensure types
    for col in ['bsp', 'morningwap', 'morningtradedvol', 'pptradedvol', 'iptradedvol']:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
        
    return master_df

def feature_engineering(df):
    logging.info("Starting feature engineering...")
    
    # 1. Implied prob
    df['implied_prob'] = 1 / df['bsp']
    
    # 2. Odds drift (BSP vs MORNINGWAP)
    # Be careful with 0 morningwap
    import numpy as np
    df['morningwap'] = np.where(df['morningwap'] == 0, df['bsp'], df['morningwap'])
    df['odds_change'] = df['bsp'] - df['morningwap']
    
    # 3. Volume
    df['early_volume'] = df['morningtradedvol']
    df['preplay_volume'] = df['pptradedvol']
    df['inplay_volume'] = df['iptradedvol']
    df['total_pre_volume'] = df['early_volume'] + df['preplay_volume']
    
    df['late_money_ratio'] = df['preplay_volume'] / (df['early_volume'] + 1e-6)
    
    # 4. CLV = (1/bsp) - (1/morningwap) 
    # Usually CLV > 0 is good (meaning BSP was lower, i.e. odds dropped)
    df['clv'] = (1 / df['bsp']) - (1 / df['morningwap'])
    
    # 5. Longshot flag
    df['is_longshot'] = (df['bsp'] > 15).astype(int)
    
    return df

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = load_and_clean_data()
    if df is not None:
        df = feature_engineering(df)
        out_path = os.path.join(PROCESSED_DIR, "master_dataset.csv")
        df.to_csv(out_path, index=False)
        logging.info(f"Saved master dataset to {out_path} with {len(df)} rows.")

if __name__ == "__main__":
    main()

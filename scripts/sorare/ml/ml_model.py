import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import logging
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SorareMLPipeline:
    def __init__(self, db_path=None, 
                 score_model_path=None, 
                 price_model_path=None,
                 meta_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = db_path or os.path.join(base_dir, "sorare_historical.db")
        self.score_model_path = score_model_path or os.path.join(base_dir, "ml", "sorare_ml_model.pkl")
        self.price_model_path = price_model_path or os.path.join(base_dir, "ml", "sorare_price_model.pkl")
        self.meta_path = meta_path or os.path.join(base_dir, "ml", "ml_metadata.json")

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    # =========================================================================
    # 1. PONT-PREDIKCIÓS MODELL (SCORE MODEL) CSŐVEZETÉKE
    # =========================================================================
    def prepare_score_dataset(self):
        conn = self.get_connection()
        df_players = pd.read_sql_query("SELECT id, display_name, position, age FROM players", conn)
        df_matches = pd.read_sql_query("SELECT * FROM match_performances", conn)
        conn.close()

        if df_players.empty or df_matches.empty:
            return None, None

        X_rows = []
        y_rows = []
        
        for idx, player in df_players.iterrows():
            p_id = player['id']
            p_pos = player['position']
            p_age = player['age']
            
            p_matches = df_matches[df_matches['player_id'] == p_id].sort_values(by='match_date', ascending=True)
            
            if len(p_matches) < 6:
                continue
                
            scores = p_matches['total_score'].values
            decisives = p_matches['decisive_score'].values
            is_home_vals = p_matches['is_home'].values
            
            for t in range(5, len(p_matches)):
                target_score = scores[t]
                prev_scores = scores[t-5:t]
                prev_decisives = decisives[t-5:t]
                
                feature_row = {
                    'prev_score_1': float(prev_scores[-1]),
                    'prev_score_2': float(prev_scores[-2]),
                    'prev_score_3': float(prev_scores[-3]),
                    'rolling_mean_3': float(np.mean(prev_scores[-3:])),
                    'rolling_mean_5': float(np.mean(prev_scores)),
                    'rolling_std_5': float(np.std(prev_scores)) if np.std(prev_scores) > 0 else 0.0,
                    'decisive_share_5': float(np.sum(prev_decisives) / (np.sum(prev_scores) + 0.1) * 100),
                    'is_home': int(is_home_vals[t]),
                    'age': int(p_age) if p_age else 25,
                    'pos_GK': 1 if p_pos == 'Goalkeeper' else 0,
                    'pos_DF': 1 if p_pos == 'Defender' else 0,
                    'pos_MD': 1 if p_pos == 'Midfielder' else 0,
                    'pos_FW': 1 if p_pos == 'Forward' else 0
                }
                
                X_rows.append(feature_row)
                y_rows.append(target_score)

        if not X_rows:
            return None, None

        return pd.DataFrame(X_rows), pd.Series(y_rows)

    # =========================================================================
    # 2. ÁR-PREDIKCIÓS MODELL (PRICE ROI MODEL) CSŐVEZETÉKE [NEW]
    # =========================================================================
    def prepare_price_dataset(self):
        """
        Adatelőkészítés a kártyák jövőbeli árváltozásának (ROI %) becsléséhez.
        Célváltozó: a következő tranzakció százalékos árváltozása.
        """
        conn = self.get_connection()
        df_players = pd.read_sql_query("SELECT id, display_name, position, age, average_score FROM players", conn)
        df_matches = pd.read_sql_query("SELECT * FROM match_performances", conn)
        df_auctions = pd.read_sql_query("SELECT * FROM auctions WHERE price_type = 'recent_sale'", conn)
        conn.close()

        if df_players.empty or df_auctions.empty:
            return None, None

        X_rows = []
        y_rows = []

        for idx, player in df_players.iterrows():
            p_id = player['id']
            p_pos = player['position']
            p_age = player['age']
            p_avg_score = player['average_score']

            # A játékos lezárult eladásai időrendben
            p_sales = df_auctions[df_auctions['player_id'] == p_id].sort_values(by='date', ascending=True)

            if len(p_sales) < 3:
                # Minimum 3 eladás kell a trendek és a target kiszámításához
                continue

            prices = p_sales['price_eur'].values
            dates = p_sales['date'].values

            for t in range(1, len(p_sales) - 1):
                # Jelenlegi eladási ár
                current_price = prices[t]
                # Következő eladási ár (célváltozó alapja)
                next_price = prices[t+1]
                
                # Célváltozó: Százalékos árváltozás a következő tranzakcióig (ROI %)
                roi_target = ((next_price - current_price) / current_price) * 100
                
                # Jellemzők (feature-ök) kinyerése a t időpont előtt
                prev_price = prices[t-1]
                price_trend = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0.0
                
                # Megkeressük az eladás időpontja előtti meccspontokat a hitelességért
                sale_date_str = dates[t]
                p_matches = df_matches[(df_matches['player_id'] == p_id) & (df_matches['match_date'] < sale_date_str)]
                
                if not p_matches.empty:
                    match_scores = p_matches.sort_values(by='match_date', ascending=False)['total_score'].values
                    l5_score = float(np.mean(match_scores[:5])) if len(match_scores) >= 5 else float(np.mean(match_scores))
                    l15_score = float(np.mean(match_scores[:15])) if len(match_scores) >= 15 else float(np.mean(match_scores))
                else:
                    l5_score = p_avg_score if p_avg_score else 45.0
                    l15_score = p_avg_score if p_avg_score else 45.0

                feature_row = {
                    'current_price': float(current_price),
                    'prev_price_1': float(prev_price),
                    'price_trend_pct': float(price_trend),
                    'score_l5': float(l5_score),
                    'score_l15': float(l15_score),
                    'score_momentum': float(l5_score - l15_score),
                    'age': int(p_age) if p_age else 25,
                    'pos_GK': 1 if p_pos == 'Goalkeeper' else 0,
                    'pos_DF': 1 if p_pos == 'Defender' else 0,
                    'pos_MD': 1 if p_pos == 'Midfielder' else 0,
                    'pos_FW': 1 if p_pos == 'Forward' else 0
                }
                
                X_rows.append(feature_row)
                y_rows.append(roi_target)

        if not X_rows:
            return None, None

        return pd.DataFrame(X_rows), pd.Series(y_rows)

    # =========================================================================
    # 3. TANÍTÁSI ÉS KIÉRTÉKELÉSI FŐFOLYAMAT
    # =========================================================================
    def train_and_evaluate(self):
        """
        Betanítja mindkét modellt (Score Predictor és Price ROI Predictor),
        kiértékeli őket és elmenti a Streamlit számára szükséges metaadatokat.
        """
        # --- A. PONT PREDIKTOR MODELL TANÍTÁSA ---
        df_X_score, df_y_score = self.prepare_score_dataset()
        score_metrics = {}
        score_feat_imp = []
        score_model = None

        score_calibration = {}
        if df_X_score is not None and not df_X_score.empty:
            df_X_score = df_X_score.fillna(0.0)
            X_train, X_test, y_train, y_test = train_test_split(df_X_score, df_y_score, test_size=0.2, random_state=42)
            score_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
            score_model.fit(X_train, y_train)
            
            y_pred = score_model.predict(X_test)
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            
            y_baseline = X_test['rolling_mean_5'].values
            mae_base = float(mean_absolute_error(y_test, y_baseline))
            
            score_metrics = {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'r2': round(r2, 3),
                'mae_baseline': round(mae_base, 2),
                'improvement_percent': round(((mae_base - mae) / mae_base) * 100, 1) if mae_base > 0 else 0.0
            }
            
            # Kalibrációs adatok mentése
            score_calibration = {
                'actual': [float(v) for v in y_test],
                'predicted': [float(v) for v in y_pred]
            }
            
            for name, imp in zip(df_X_score.columns, score_model.feature_importances_):
                score_feat_imp.append({'feature': name, 'importance': round(float(imp) * 100, 2)})
            score_feat_imp = sorted(score_feat_imp, key=lambda x: x['importance'], reverse=True)
            
            with open(self.score_model_path, 'wb') as f:
                pickle.dump(score_model, f)
            logger.info("Pontszám predikciós modell sikeresen betanítva és mentve.")
        else:
            logger.warning("Nincs elég adat a pontszám modell betanításához.")

        # --- B. ÁR ROI PREDIKTOR MODELL TANÍTÁSA [NEW] ---
        df_X_price, df_y_price = self.prepare_price_dataset()
        price_metrics = {}
        price_feat_imp = []
        price_model = None

        price_calibration = {}
        if df_X_price is not None and not df_X_price.empty:
            df_X_price = df_X_price.fillna(0.0)
            X_train, X_test, y_train, y_test = train_test_split(df_X_price, df_y_price, test_size=0.2, random_state=42)
            price_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
            price_model.fit(X_train, y_train)
            
            y_pred = price_model.predict(X_test)
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            
            # Baseline: Azt jósoljuk, hogy az ár nem változik (ROI = 0.0%)
            y_baseline = np.zeros(len(y_test))
            mae_base = float(mean_absolute_error(y_test, y_baseline))
            
            price_metrics = {
                'mae': round(mae, 2), # MAE százalékpontban mérve
                'rmse': round(rmse, 2),
                'r2': round(r2, 3),
                'mae_baseline': round(mae_base, 2),
                'improvement_percent': round(((mae_base - mae) / mae_base) * 100, 1) if mae_base > 0 else 0.0
            }
            
            # Kalibrációs adatok mentése
            price_calibration = {
                'actual': [float(v) for v in y_test],
                'predicted': [float(v) for v in y_pred]
            }
            
            for name, imp in zip(df_X_price.columns, price_model.feature_importances_):
                price_feat_imp.append({'feature': name, 'importance': round(float(imp) * 100, 2)})
            price_feat_imp = sorted(price_feat_imp, key=lambda x: x['importance'], reverse=True)
            
            with open(self.price_model_path, 'wb') as f:
                pickle.dump(price_model, f)
            logger.info("Árfolyam ROI predikciós modell sikeresen betanítva és mentve.")
        else:
            logger.warning("Nincs elég adat az árfolyam modell betanításához.")

        # --- C. JÁTÉKOS PREDIKCIÓK GENERÁLÁSA MIND KÉT MODELL ALAPJÁN ---
        predictions = {}
        if score_model and price_model:
            predictions = self.generate_combined_predictions(score_model, price_model)
            
        # Metaadatok kiírása JSON-be a Streamlitnek
        meta_data = {
            'score_model': {
                'metrics': score_metrics,
                'feature_importances': score_feat_imp,
                'calibration': score_calibration
            },
            'price_model': {
                'metrics': price_metrics,
                'feature_importances': price_feat_imp,
                'calibration': price_calibration
            },
            'predictions': predictions,
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Minden modell metaadat elmentve ide: {self.meta_path}")
        return True

    def generate_combined_predictions(self, score_model, price_model):
        """
        Minden játékoshoz kiszámítja mind a várható következő meccspontokat, 
        mind a várható kártya ROI-t (%) a legfrissebb piaci floor árak alapján.
        """
        conn = self.get_connection()
        df_players = pd.read_sql_query("SELECT id, display_name, position, age, average_score, is_injured, is_suspended FROM players", conn)
        df_matches = pd.read_sql_query("SELECT * FROM match_performances", conn)
        df_auctions = pd.read_sql_query("SELECT * FROM auctions", conn)
        conn.close()
        
        predictions = {}
        
        for idx, player in df_players.iterrows():
            p_id = player['id']
            p_name = player['display_name']
            p_pos = player['position']
            p_age = player['age']
            p_avg = player['average_score']
            
            # --- 1. Meccsadatok kinyerése a predikciókhoz ---
            p_matches = df_matches[df_matches['player_id'] == p_id].sort_values(by='match_date', ascending=True)
            if len(p_matches) < 5:
                continue
                
            last_5 = p_matches.tail(5)
            scores = last_5['total_score'].values
            decisives = last_5['decisive_score'].values
            
            # --- 2. Áradatok kinyerése a predikciókhoz ---
            p_auctions = df_auctions[df_auctions['player_id'] == p_id]
            listings = p_auctions[p_auctions['price_type'] == 'direct_listing']
            sales = p_auctions[p_auctions['price_type'] == 'recent_sale'].sort_values(by='date', ascending=True)
            
            floor_price = listings['price_eur'].min() if not listings.empty else None
            
            # Ha nincsenek eladások, nehéz ármodellt futtatni erre a játékosra
            if sales.empty or not floor_price:
                continue
                
            prev_sale_price = sales.iloc[-1]['price_eur']
            price_trend = ((floor_price - prev_sale_price) / prev_sale_price) * 100 if prev_sale_price > 0 else 0.0
            
            # --- 3. PONT PREDIKCIÓ FUTTATÁSA ---
            features_score_h = {
                'prev_score_1': float(scores[-1]),
                'prev_score_2': float(scores[-2]),
                'prev_score_3': float(scores[-3]),
                'rolling_mean_3': float(np.mean(scores[-3:])),
                'rolling_mean_5': float(np.mean(scores)),
                'rolling_std_5': float(np.std(scores)) if np.std(scores) > 0 else 0.0,
                'decisive_share_5': float(np.sum(decisives) / (np.sum(scores) + 0.1) * 100),
                'is_home': 1,
                'age': int(p_age) if p_age else 25,
                'pos_GK': 1 if p_pos == 'Goalkeeper' else 0,
                'pos_DF': 1 if p_pos == 'Defender' else 0,
                'pos_MD': 1 if p_pos == 'Midfielder' else 0,
                'pos_FW': 1 if p_pos == 'Forward' else 0
            }
            features_score_a = features_score_h.copy()
            features_score_a['is_home'] = 0
            
            p_injured = int(player.get('is_injured', 0))
            p_suspended = int(player.get('is_suspended', 0))
            
            # Ha sérült vagy eltiltott, a várható pontszáma automatikusan 0
            if p_injured == 1 or p_suspended == 1:
                pred_score_h = 0.0
                pred_score_a = 0.0
            else:
                pred_score_h = round(float(score_model.predict(pd.DataFrame([features_score_h]).fillna(0.0))[0]), 2)
                pred_score_a = round(float(score_model.predict(pd.DataFrame([features_score_a]).fillna(0.0))[0]), 2)
            
            # --- 4. ÁR ROI PREDIKCIÓ FUTTATÁSA ---
            features_price = {
                'current_price': float(floor_price),
                'prev_price_1': float(prev_sale_price),
                'price_trend_pct': float(price_trend),
                'score_l5': float(np.mean(scores)),
                'score_l15': float(p_avg) if p_avg else float(np.mean(scores)),
                'score_momentum': float(np.mean(scores) - (p_avg if p_avg else np.mean(scores))),
                'age': int(p_age) if p_age else 25,
                'pos_GK': 1 if p_pos == 'Goalkeeper' else 0,
                'pos_DF': 1 if p_pos == 'Defender' else 0,
                'pos_MD': 1 if p_pos == 'Midfielder' else 0,
                'pos_FW': 1 if p_pos == 'Forward' else 0
            }
            
            pred_roi = round(float(price_model.predict(pd.DataFrame([features_price]).fillna(0.0))[0]), 2)
            
            predictions[p_id] = {
                'player_name': p_name,
                'predicted_score_home': pred_score_h,
                'predicted_score_away': pred_score_a,
                'predicted_roi': pred_roi,
                'current_floor': round(float(floor_price), 2),
                'recent_average': round(float(np.mean(scores)), 2),
                'is_injured': p_injured,
                'is_suspended': p_suspended
            }
            
        return predictions

if __name__ == "__main__":
    pipeline = SorareMLPipeline()
    pipeline.train_and_evaluate()

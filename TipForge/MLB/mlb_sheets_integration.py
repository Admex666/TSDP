# mlb_sheets_integration.py
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import requests
import time
from datetime import datetime
import streamlit as st
from mlb_config import GOOGLE_SHEETS_ID, GOOGLE_CREDENTIALS_FILE

class MLBSheetsIntegration:
    def __init__(self):
        self.spreadsheet_id = GOOGLE_SHEETS_ID
        self.client = None
        self.spreadsheet = None
        self.init_client()
    
    def init_client(self):
        """Google Sheets kliens inicializálása"""
        try:
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_file(
                GOOGLE_CREDENTIALS_FILE,
                scopes=scope
            )
            
            self.client = gspread.authorize(credentials)
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            print("Google Sheets kapcsolat létrehozva")
            
        except Exception as e:
            print(f"Google Sheets kapcsolat hiba: {e}")
            self.client = None
    
    def ensure_worksheets(self):
        """Szükséges munkalapok létrehozása"""
        if not self.client:
            return False
        
        try:
            worksheets = [ws.title for ws in self.spreadsheet.worksheets()]
            
            # Bets worksheet
            if 'MLB_Bets' not in worksheets:
                bets_sheet = self.spreadsheet.add_worksheet(title='MLB_Bets', rows=1000, cols=15)
                # Header setup
                headers = [
                    'Date', 'Game_ID', 'Home_Team', 'Away_Team', 'Bet_Type', 
                    'Odds', 'Stake', 'Prediction_Home', 'Prediction_Away',
                    'Result', 'Payout', 'Profit_Loss', 'Status', 'Notes', 'ROI'
                ]
                bets_sheet.update('A1:O1', [headers])
            
            # Stats worksheet
            if 'MLB_Stats' not in worksheets:
                stats_sheet = self.spreadsheet.add_worksheet(title='MLB_Stats', rows=100, cols=10)
                headers = [
                    'Metric', 'Value', 'Last_Updated', 'Total_Bets', 'Won_Bets',
                    'Lost_Bets', 'Win_Rate', 'Total_Profit', 'ROI', 'Average_Odds'
                ]
                stats_sheet.update('A1:J1', [headers])
            
            return True
            
        except Exception as e:
            print(f"Worksheet creation error: {e}")
            return False
    
    def add_bet(self, game_data, bet_type, odds, stake, predictions):
        """Új fogadás hozzáadása a spreadsheethez"""
        if not self.client:
            return False
        
        try:
            bets_sheet = self.spreadsheet.worksheet('MLB_Bets')
            
            bet_row = [
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                game_data.get('game_id', ''),
                game_data.get('home_team', ''),
                game_data.get('away_team', ''),
                bet_type,  # 'Home' or 'Away'
                odds,
                stake,
                f"{predictions.get('home_prob', 0):.3f}",
                f"{predictions.get('away_prob', 0):.3f}",
                '',  # Result - to be filled later
                '',  # Payout
                '',  # Profit/Loss
                'Pending',  # Status
                '',  # Notes
                ''   # ROI
            ]
            
            bets_sheet.append_row(bet_row)
            return True
            
        except Exception as e:
            print(f"Add bet error: {e}")
            return False
    
    def update_bet_result(self, game_id, winner, final_score=None):
        """Fogadás eredményének frissítése"""
        if not self.client:
            return False
        
        try:
            bets_sheet = self.spreadsheet.worksheet('MLB_Bets')
            records = bets_sheet.get_all_records()
            
            for i, record in enumerate(records):
                if str(record['Game_ID']) == str(game_id) and record['Status'] == 'Pending':
                    row_num = i + 2  # +2 because records start from row 2
                    
                    # Determine if bet won
                    bet_type = record['Bet_Type']
                    won = (bet_type == 'Home' and winner == 'Home') or (bet_type == 'Away' and winner == 'Away')
                    
                    stake = float(record['Stake'])
                    odds = float(record['Odds'])
                    
                    if won:
                        payout = stake * odds  # JAVÍTÁS: nem kell *100, az odds már helyes formátumban van
                        profit_loss = payout - stake  # Nyeremény = teljes visszafizetés - eredeti tét
                        status = 'Won'
                    else:
                        payout = 0
                        profit_loss = -stake
                        status = 'Lost'
                    
                    roi = (profit_loss / stake) * 100
                    
                    # Update row
                    bets_sheet.update(f'J{row_num}:O{row_num}', [[
                        winner,
                        payout,
                        profit_loss,
                        status,
                        final_score or '',
                        f"{roi:.2f}%"
                    ]])
            
            # Update overall stats
            time.sleep(1)  # Várakozás a rate limit elkerülése érdekében
            self.update_stats()
            return True
            
        except Exception as e:
            print(f"Update result error: {e}")
            return False
        
    def update_stats(self):
        """Összesített statisztikák frissítése"""
        if not self.client:
            return False
        
        try:
            bets_sheet = self.spreadsheet.worksheet('MLB_Bets')
            stats_sheet = self.spreadsheet.worksheet('MLB_Stats')
            
            records = bets_sheet.get_all_records()
            completed_bets = [r for r in records if r['Status'] in ['Won', 'Lost']]
            
            if not completed_bets:
                return True
            
            # Calculate stats
            total_bets = len(completed_bets)
            won_bets = len([r for r in completed_bets if r['Status'] == 'Won'])
            lost_bets = total_bets - won_bets
            win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
            
            total_profit = sum(float(r['Profit_Loss']) for r in completed_bets if r['Profit_Loss'])
            total_staked = sum(float(r['Stake']) for r in completed_bets if r['Stake'])
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            
            avg_odds = sum(float(r['Odds']) for r in completed_bets if r['Odds']) / total_bets if total_bets > 0 else 0
            
            # Update stats sheet
            stats_data = [
                ['Total Bets', total_bets, datetime.now().strftime('%Y-%m-%d %H:%M')],
                ['Won Bets', won_bets, ''],
                ['Lost Bets', lost_bets, ''],
                ['Win Rate (%)', f"{win_rate:.2f}%", ''],
                ['Total Profit', f"${total_profit:.2f}", ''],
                ['Total Staked', f"${total_staked:.2f}", ''],
                ['ROI (%)', f"{roi:.2f}%", ''],
                ['Average Odds', f"{avg_odds:.2f}", '']
            ]
            
            stats_sheet.update('A2:C9', stats_data)
            return True
            
        except Exception as e:
            print(f"Update stats error: {e}")
            return False
    
    def get_stats(self):
        """Statisztikák lekérése"""
        if not self.client:
            return None
        
        try:
            stats_sheet = self.spreadsheet.worksheet('MLB_Stats')
            records = stats_sheet.get_all_records()
            return {record['Metric']: record['Value'] for record in records}
        except Exception as e:
            print(f"Get stats error: {e}")
            return None
    
    def get_pending_bets(self):
        """Függő fogadások lekérése"""
        if not self.client:
            return []
        
        try:
            bets_sheet = self.spreadsheet.worksheet('MLB_Bets')
            records = bets_sheet.get_all_records()
            return [r for r in records if r['Status'] == 'Pending']
        except Exception as e:
            print(f"Get pending bets error: {e}")
            return []
        
    def update_all_pending_results(self):
        """Összes függő fogadás eredményének automatikus frissítése"""
        if not self.client:
            return False, "Google Sheets kapcsolat hiányzik"
        
        try:
            # Egyszer olvassuk be az összes adatot
            bets_sheet = self.spreadsheet.worksheet('MLB_Bets')
            records = bets_sheet.get_all_records()
            
            pending_bets = [r for r in records if r['Status'] == 'Pending' and r['Game_ID']]
            
            if not pending_bets:
                return True, "Nincsenek frissítendő fogadások"
            
            updated_count = 0
            errors = []
            updates_batch = []  # Batch frissítéshez
            
            for i, bet in enumerate(pending_bets):
                try:
                    game_id = bet['Game_ID']
                    
                    # MLB API lekérdezés
                    url = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={game_id}"
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    if not data["dates"] or not data["dates"][0]["games"]:
                        continue
                    
                    game = data["dates"][0]["games"][0]
                    
                    # Ellenőrizzük, hogy befejezett-e a meccs
                    if game["status"]["statusCode"] not in ["F", "O"]:  # Final, Official
                        continue
                    
                    home_runs = game["teams"]["home"]["score"]
                    away_runs = game["teams"]["away"]["score"]
                    
                    # Győztes meghatározása
                    winner = "Home" if home_runs > away_runs else "Away"
                    final_score = f"{home_runs}-{away_runs}"
                    
                    # Batch frissítéshez készítjük elő az adatokat
                    row_num = None
                    for j, record in enumerate(records):
                        if str(record['Game_ID']) == str(game_id) and record['Status'] == 'Pending':
                            row_num = j + 2  # +2 mert az első sor header
                            break
                    
                    if row_num:
                        # Profit számítás javítása
                        bet_type = bet['Bet_Type']
                        won = (bet_type == 'Home' and winner == 'Home') or (bet_type == 'Away' and winner == 'Away')
                        
                        stake = float(bet['Stake'])
                        odds = float(bet['Odds'])
                        
                        if won:
                            payout = stake * odds  # Teljes visszafizetés (tét + nyeremény)
                            profit_loss = payout - stake  # Csak a nyeremény
                            status = 'Won'
                        else:
                            payout = 0
                            profit_loss = -stake  # Elveszített tét
                            status = 'Lost'
                        
                        roi = (profit_loss / stake) * 100
                        
                        updates_batch.append({
                            'range': f'J{row_num}:O{row_num}',
                            'values': [[winner, payout, profit_loss, status, final_score, f"{roi:.2f}%"]]
                        })
                        updated_count += 1
                    
                    # Rate limit elkerülése - várakozás minden 3. kérés után
                    if i % 3 == 0 and i > 0:
                        time.sleep(1)
                    
                except Exception as e:
                    errors.append(f"Game ID {game_id}: {str(e)}")
                    continue
            
            # Batch frissítés végrehajtása
            if updates_batch:
                for update in updates_batch:
                    bets_sheet.update(update['range'], update['values'])
                    time.sleep(0.5)  # Kis várakozás minden frissítés között
                
                # Stats frissítése egyszer a végén
                time.sleep(1)
                self.update_stats()
            
            message = f"{updated_count} fogadás frissítve"
            if errors:
                message += f". Hibák: {len(errors)}"
            
            return True, message
            
        except Exception as e:
            return False, f"Frissítési hiba: {str(e)}"
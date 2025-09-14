# mlb_sheets_integration.py
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
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
                        payout = stake * odds
                        profit_loss = payout - stake
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
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import config
import uuid
import os
import json

class Database:
    def __init__(self):
        scopes = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds_json = os.getenv("GOOGLE_CREDENTIALS")
        creds_dict = json.loads(creds_json)

        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes)
        self.client = gspread.authorize(creds)
        self.sheet = self.client.open_by_key(config.SHEET_ID)
        
        self.users = self.sheet.worksheet('Users')
        self.transactions = self.sheet.worksheet('Transactions')
        self.referrals = self.sheet.worksheet('Referrals')
    
    # === USER MANAGEMENT ===
    def get_user(self, discord_id):
        """Felhasználó lekérése Discord ID alapján"""
        try:
            cell = self.users.find(str(discord_id))
            if cell:
                row = self.users.row_values(cell.row)
                return {
                    'discord_id': row[0],
                    'username': row[1],
                    'tier': row[2],
                    'total_points': int(row[3]) if row[3] else 0,
                    'monthly_points': int(row[4]) if row[4] else 0,
                    'join_date': row[5],
                    'last_activity': row[6],
                    'referral_count': int(row[7]) if row[7] else 0,
                    'email': row[8] if len(row) > 8 else ''
                }
        except:
            return None
    
    def create_user(self, discord_id, username, tier='free'):
        """Új user létrehozása"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.users.append_row([
            str(discord_id),
            username,
            tier,
            0,  # total_points
            0,  # monthly_points
            now,  # join_date
            now,  # last_activity
            0,  # referral_count
            ''  # email
        ])
        return True
    
    def update_user_points(self, discord_id, points_change, reason):
        """Pontok frissítése"""
        user = self.get_user(discord_id)
        if not user:
            return False
        
        cell = self.users.find(str(discord_id))
        row = cell.row
        
        new_total = user['total_points'] + points_change
        new_monthly = user['monthly_points'] + points_change
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Frissítés
        self.users.update_cell(row, 4, new_total)  # total_points
        self.users.update_cell(row, 5, new_monthly)  # monthly_points
        self.users.update_cell(row, 7, now)  # last_activity
        
        # Transaction log
        self.add_transaction(discord_id, points_change, reason)
        
        return True
    
    def update_user_tier(self, discord_id, new_tier):
        """Tier frissítése"""
        cell = self.users.find(str(discord_id))
        if cell:
            self.users.update_cell(cell.row, 3, new_tier)
            return True
        return False
    
    def get_leaderboard(self, limit=10):
        """Top userek lekérése"""
        all_records = self.users.get_all_records()
        sorted_users = sorted(all_records, 
                            key=lambda x: int(x['total_points']) if x['total_points'] else 0, 
                            reverse=True)
        return sorted_users[:limit]
    
    # === TRANSACTION MANAGEMENT ===
    def add_transaction(self, discord_id, points_change, reason):
        """Új tranzakció rögzítése"""
        transaction_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.transactions.append_row([
            transaction_id,
            str(discord_id),
            points_change,
            reason,
            timestamp,
            'completed'
        ])
        return transaction_id
    
    def get_user_transactions(self, discord_id, limit=10):
        """User tranzakciói"""
        all_records = self.transactions.get_all_records()
        user_trans = [t for t in all_records if str(t['discord_id']) == str(discord_id)]
        return user_trans[-limit:]
    
    # === REFERRAL MANAGEMENT ===
    def create_referral(self, referrer_id, referred_id, code):
        """Új referral rögzítése"""
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.referrals.append_row([
            str(referrer_id),
            str(referred_id),
            code,
            date,
            'registered'
        ])
        
        # Referrer pontjainak növelése
        self.update_user_points(referrer_id, config.POINTS['referral_register'], 
                               f'Referral: {referred_id}')
        
        # Referral count növelése
        user = self.get_user(referrer_id)
        if user:
            cell = self.users.find(str(referrer_id))
            new_count = user['referral_count'] + 1
            self.users.update_cell(cell.row, 8, new_count)
    
    def get_referral_code(self, discord_id):
        """User referral code-ja"""
        return f"REF{discord_id}"
    
    # === UTILITY ===
    def check_daily_limit(self, discord_id):
        """Napi pont limit ellenőrzése"""
        today = datetime.now().strftime('%Y-%m-%d')
        all_trans = self.transactions.get_all_records()
        
        today_points = sum([
            int(t['points_change']) 
            for t in all_trans 
            if str(t['discord_id']) == str(discord_id) 
            and t['timestamp'].startswith(today)
            and int(t['points_change']) > 0
        ])
        
        return today_points < config.POINTS['daily_limit']
    
    def reset_monthly_points(self):
        """Havi pontok nullázása (cron job)"""
        all_records = self.users.get_all_records()
        for i, record in enumerate(all_records, start=2):
            self.users.update_cell(i, 5, 0)  # monthly_points = 0
import os
from dotenv import load_dotenv

load_dotenv()

# Discord
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GUILD_ID = int(os.getenv('GUILD_ID', 0))

# Google Sheets
SHEET_ID = os.getenv('SHEET_ID')
CREDENTIALS_FILE = 'credentials.json'

# Channels (Discord csatorna ID-k)
CHANNELS = {
    'free': int(os.getenv('CHANNEL_FREE', 0)),
    'basic': int(os.getenv('CHANNEL_BASIC', 0)),
    'standard': int(os.getenv('CHANNEL_STANDARD', 0)),
    'premium': int(os.getenv('CHANNEL_PREMIUM', 0)),
    'support': int(os.getenv('CHANNEL_SUPPORT', 0)),
    'announcements': int(os.getenv('CHANNEL_ANNOUNCEMENTS', 0))
}

# Roles (Discord rang ID-k)
ROLES = {
    'free': int(os.getenv('ROLE_FREE', 0)),
    'basic': int(os.getenv('ROLE_BASIC', 0)),
    'standard': int(os.getenv('ROLE_STANDARD', 0)),
    'premium': int(os.getenv('ROLE_PREMIUM', 0)),
    'elite': int(os.getenv('ROLE_ELITE', 0))
}

# Pontrendszer
POINTS = {
    'message': 1,
    'reaction': 1,
    'win_post': 50,
    'referral_register': 100,
    'referral_basic': 500,
    'referral_standard': 1000,
    'referral_premium': 2000,
    'daily_limit': 50,
    'monthly_limit': 1500
}

# Tier thresholds
TIER_POINTS = {
    'basic': 1000,
    'standard': 5000,
    'premium': 10000,
    'elite': 25000
}

# Timezone
TIMEZONE = 'Europe/Budapest'
import discord
from discord.ext import commands, tasks
import config
from database import Database
from datetime import datetime
import asyncio

# Web endpoint
from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "✅ Bot is running!"

def run():
    app.run(host='0.0.0.0', port=8080)

# indítsd el külön szálon
Thread(target=run).start()


# Bot setup
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)
db = Database()

@bot.event
async def on_ready():
    print(f'✅ Bot bejelentkezve: {bot.user.name}')
    print(f'📊 {len(bot.guilds)} szerverhez csatlakozva')
    
    # Load cogs
    try:
        await bot.load_extension('cogs.tips')
        await bot.load_extension('cogs.points')
        await bot.load_extension('cogs.support')
        print('✅ Cogs betöltve')
    except Exception as e:
        print(f'❌ Cog betöltési hiba: {e}')
    
    # Indítsd el a scheduled taskokat
    daily_tips.start()
    check_tier_upgrades.start()

@bot.event
async def on_member_join(member):
    """Új tag érkezik"""
    # Létrehozás az adatbázisban
    db.create_user(member.id, str(member), 'free')
    
    # Free role hozzáadása
    guild = member.guild
    free_role = guild.get_role(config.ROLES['free'])
    if free_role:
        await member.add_roles(free_role)
    
    # Üdvözlő üzenet
    channel = guild.get_channel(config.CHANNELS['announcements'])
    if channel:
        embed = discord.Embed(
            title="🎉 Üdvözlünk!",
            description=f"Üdv {member.mention}! Kezdő pontjaid: **0**\n\n"
                       f"📝 Használd a `!help` parancsot a funkciók megtekintéséhez!",
            color=discord.Color.green()
        )
        await channel.send(embed=embed)

# === BASIC COMMANDS ===
@bot.command(name='balance')
async def balance(ctx):
    """Pontegyenleg lekérdezése"""
    user = db.get_user(ctx.author.id)
    
    if not user:
        db.create_user(ctx.author.id, str(ctx.author), 'free')
        user = db.get_user(ctx.author.id)
    
    embed = discord.Embed(
        title=f"💰 {ctx.author.name} pontjai",
        color=discord.Color.gold()
    )
    embed.add_field(name="Összes pont", value=f"**{user['total_points']}** 🪙", inline=True)
    embed.add_field(name="Havi pont", value=f"**{user['monthly_points']}** 🪙", inline=True)
    embed.add_field(name="Tier", value=f"**{user['tier'].upper()}**", inline=True)
    embed.add_field(name="Meghívottak", value=f"**{user['referral_count']}** fő", inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name='leaderboard')
async def leaderboard(ctx):
    """Top 10 felhasználó"""
    leaders = db.get_leaderboard(10)
    
    embed = discord.Embed(
        title="🏆 TOP 10 Felhasználó",
        description="A legtöbb ponttal rendelkező tagok",
        color=discord.Color.gold()
    )
    
    medals = ["🥇", "🥈", "🥉"]
    for i, user in enumerate(leaders, 1):
        medal = medals[i-1] if i <= 3 else f"{i}."
        embed.add_field(
            name=f"{medal} {user['username']}",
            value=f"**{user['total_points']}** pont | Tier: {user['tier']}",
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name='referral')
async def referral(ctx):
    """Meghívó kód generálása"""
    code = db.get_referral_code(ctx.author.id)
    user = db.get_user(ctx.author.id)
    
    embed = discord.Embed(
        title="🎁 Meghívó Kódod",
        description=f"Oszd meg ezt a kódot barátaiddal!\n\n**Kód:** `{code}`",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="💰 Jutalmak",
        value=f"• Regisztráció: **{config.POINTS['referral_register']}** pont\n"
              f"• Basic előfizetés: **{config.POINTS['referral_basic']}** pont\n"
              f"• Standard előfizetés: **{config.POINTS['referral_standard']}** pont\n"
              f"• Premium előfizetés: **{config.POINTS['referral_premium']}** pont",
        inline=False
    )
    embed.add_field(
        name="📊 Statisztikád",
        value=f"Meghívott tagok: **{user['referral_count']}** fő",
        inline=False
    )
    
    await ctx.send(embed=embed)

@bot.command(name='redeem')
async def redeem(ctx, code: str):
    """Meghívó kód beváltása"""
    if not code.startswith('REF'):
        await ctx.send("❌ Érvénytelen kód formátum!")
        return
    
    referrer_id = int(code.replace('REF', ''))
    
    # Ellenőrzések
    if referrer_id == ctx.author.id:
        await ctx.send("❌ Nem használhatod a saját kódodat!")
        return
    
    referrer = db.get_user(referrer_id)
    if not referrer:
        await ctx.send("❌ Érvénytelen meghívó kód!")
        return
    
    # Referral létrehozása
    db.create_referral(referrer_id, ctx.author.id, code)
    
    await ctx.send(f"✅ Meghívó kód beváltva! {referrer['username']} **{config.POINTS['referral_register']}** pontot kapott!")

@bot.command(name='history')
async def history(ctx):
    """Tranzakciós előzmények"""
    transactions = db.get_user_transactions(ctx.author.id, 10)
    
    if not transactions:
        await ctx.send("Még nincsenek tranzakcióid.")
        return
    
    embed = discord.Embed(
        title=f"📜 {ctx.author.name} tranzakciói",
        description="Utolsó 10 tranzakció",
        color=discord.Color.blue()
    )
    
    for trans in reversed(transactions):
        sign = "+" if int(trans['points_change']) > 0 else ""
        embed.add_field(
            name=f"{trans['reason']}",
            value=f"{sign}{trans['points_change']} pont | {trans['timestamp']}",
            inline=False
        )
    
    await ctx.send(embed=embed)

# === ADMIN COMMANDS ===
@bot.command(name='addpoints')
@commands.has_permissions(administrator=True)
async def addpoints(ctx, member: discord.Member, points: int, *, reason: str):
    """Admin: pontok hozzáadása"""
    db.update_user_points(member.id, points, f"Admin: {reason}")
    await ctx.send(f"✅ **{points}** pont hozzáadva {member.mention} számára. Ok: {reason}")

@bot.command(name='settier')
@commands.has_permissions(administrator=True)
async def settier(ctx, member: discord.Member, tier: str):
    """Admin: tier beállítása"""
    tier = tier.lower()
    if tier not in config.ROLES:
        await ctx.send(f"❌ Érvénytelen tier! Választható: {', '.join(config.ROLES.keys())}")
        return
    
    # Tier frissítés adatbázisban
    db.update_user_tier(member.id, tier)
    
    # Role frissítés Discordon
    for role_name, role_id in config.ROLES.items():
        role = ctx.guild.get_role(role_id)
        if role:
            if role_name == tier:
                await member.add_roles(role)
            else:
                await member.remove_roles(role)
    
    await ctx.send(f"✅ {member.mention} tier-je beállítva: **{tier.upper()}**")

# === SCHEDULED TASKS ===
@tasks.loop(hours=24)
async def daily_tips():
    """Napi tippek posztolása (reggel 6:00)"""
    now = datetime.now()
    if now.hour != 6:
        return
    
    # Itt csatlakoztatható az algoritmus
    tips = [
        {
            'match': 'Real Madrid vs Barcelona',
            'tip': '1X',
            'odds': 1.85,
            'confidence': 75,
            'tier': 'free'
        },
        {
            'match': 'Liverpool vs Manchester City',
            'tip': 'Over 2.5',
            'odds': 2.10,
            'confidence': 82,
            'tier': 'basic'
        }
    ]
    
    for tip in tips:
        channel_id = config.CHANNELS.get(tip['tier'])
        if not channel_id:
            continue
        
        channel = bot.get_channel(channel_id)
        if not channel:
            continue
        
        embed = discord.Embed(
            title=f"⚽ {tip['match']}",
            description=f"**Tipp:** {tip['tip']}\n**Odds:** {tip['odds']}\n**Bizalom:** {tip['confidence']}%",
            color=discord.Color.green()
        )
        embed.set_footer(text=f"Tier: {tip['tier'].upper()} | {now.strftime('%Y-%m-%d %H:%M')}")
        
        await channel.send(embed=embed)

@tasks.loop(minutes=30)
async def check_tier_upgrades():
    """Automatikus tier upgrade pont alapján"""
    try:
        all_users = db.users.get_all_records()
        if not all_users:
            print("⚠️  Üres a sheet!")
            return
    except IndexError:
        print("⚠️  A sheet nem tartalmaz értelmezhető adatot!")
        return

    
    for user in all_users:
        total_points = int(user['total_points']) if user['total_points'] else 0
        current_tier = user['tier']
        new_tier = current_tier
        
        # Tier meghatározása pontok alapján
        if total_points >= config.TIER_POINTS['elite']:
            new_tier = 'elite'
        elif total_points >= config.TIER_POINTS['premium']:
            new_tier = 'premium'
        elif total_points >= config.TIER_POINTS['standard']:
            new_tier = 'standard'
        elif total_points >= config.TIER_POINTS['basic']:
            new_tier = 'basic'
        
        # Ha változott a tier
        if new_tier != current_tier and new_tier in config.ROLES:
            db.update_user_tier(user['discord_id'], new_tier)
            
            # Discord role frissítés
            for guild in bot.guilds:
                member = guild.get_member(int(user['discord_id']))
                if member:
                    # Régi role-ok eltávolítása
                    for role_name, role_id in config.ROLES.items():
                        role = guild.get_role(role_id)
                        if role and role in member.roles:
                            await member.remove_roles(role)
                    
                    # Új role hozzáadása
                    new_role = guild.get_role(config.ROLES[new_tier])
                    if new_role:
                        await member.add_roles(new_role)
                    
                    # Értesítés
                    channel = guild.get_channel(config.CHANNELS['announcements'])
                    if channel:
                        await channel.send(
                            f"🎊 Gratulálunk {member.mention}! Elérted a **{new_tier.upper()}** tier-t!"
                        )

# Bot indítása
if __name__ == '__main__':
    bot.run(config.DISCORD_TOKEN)
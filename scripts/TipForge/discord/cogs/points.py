import discord
from discord.ext import commands
import config
from database import Database

class Points(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = Database()
    
    @commands.Cog.listener()
    async def on_message(self, message):
        """Pontok az üzenetekért"""
        # Bot saját üzeneteit ne számoljuk
        if message.author.bot:
            return
        
        # Support csatorna ne adjon pontot
        if message.channel.id == config.CHANNELS.get('support', 0):
            return
        
        # User létrehozása ha még nincs
        user = self.db.get_user(message.author.id)
        if not user:
            self.db.create_user(message.author.id, str(message.author), 'free')
            user = self.db.get_user(message.author.id)
        
        # Napi limit ellenőrzés
        if not self.db.check_daily_limit(message.author.id):
            return
        
        # Pont hozzáadása
        self.db.update_user_points(
            message.author.id, 
            config.POINTS['message'], 
            'Üzenet írása'
        )
    
    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        """Pontok reakciókért"""
        if user.bot:
            return
        
        # User létrehozása ha még nincs
        user_data = self.db.get_user(user.id)
        if not user_data:
            self.db.create_user(user.id, str(user), 'free')
        
        # Napi limit ellenőrzés
        if not self.db.check_daily_limit(user.id):
            return
        
        # Pont hozzáadása
        self.db.update_user_points(
            user.id, 
            config.POINTS['reaction'], 
            'Reakció hozzáadása'
        )
    
    @commands.command(name='shop')
    async def shop(self, ctx):
        """Pontbolt megtekintése"""
        embed = discord.Embed(
            title="🏪 Pontbolt",
            description="Válts be pontjaidat jutalmakra!",
            color=discord.Color.purple()
        )
        
        embed.add_field(
            name="💎 Digitális Jutalmak",
            value=(
                "**500 pont** - Exkluzív tipp PDF (1 nap)\n"
                "**1000 pont** - ROI kalkulátor Excel\n"
                "**2000 pont** - Betting stratégia guide\n"
                "**5000 pont** - 1 hónap Premium tier"
            ),
            inline=False
        )
        
        embed.add_field(
            name="🎁 Fizikai Jutalmak",
            value=(
                "**10000 pont** - 50€ Betting kredit\n"
                "**15000 pont** - Csapat mez (választható)\n"
                "**25000 pont** - VIP meccs jegy (2 fő)"
            ),
            inline=False
        )
        
        embed.add_field(
            name="🏆 Különleges",
            value=(
                "**50000 pont** - 1-on-1 konzultáció profi tipsterrel\n"
                "**100000 pont** - Élő meccs látogatás profi tipsterrel"
            ),
            inline=False
        )
        
        embed.set_footer(text="Használd a !buy <jutalom_id> parancsot a vásárláshoz")
        
        await ctx.send(embed=embed)
    
    @commands.command(name='buy')
    async def buy(self, ctx, *, item: str):
        """Jutalom vásárlása"""
        user = self.db.get_user(ctx.author.id)
        
        if not user:
            await ctx.send("❌ Nem található felhasználói fiókod!")
            return
        
        # Itt implementálható a konkrét vásárlási logika
        # Most csak példa
        
        embed = discord.Embed(
            title="🛒 Vásárlás",
            description=f"Szeretnéd megvásárolni: **{item}**?",
            color=discord.Color.blue()
        )
        embed.add_field(
            name="💰 Egyenleged",
            value=f"{user['total_points']} pont",
            inline=True
        )
        embed.add_field(
            name="📞 Kapcsolat",
            value="Írj egy adminnnak a vásárlás befejezéséhez!",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='daily')
    async def daily(self, ctx):
        """Napi bonus pont (egyszer használható)"""
        user = self.db.get_user(ctx.author.id)
        
        if not user:
            self.db.create_user(ctx.author.id, str(ctx.author), 'free')
        
        # Ellenőrizni kellene, hogy ma már használta-e
        # Most egyszerűség kedvéért minden nap 50 pontot ad
        
        self.db.update_user_points(ctx.author.id, 50, 'Napi bonus')
        
        await ctx.send(f"✅ {ctx.author.mention} Kaptál **50** napi bonus pontot! 🎁")
    
    @commands.command(name='give')
    async def give_points(self, ctx, member: discord.Member, points: int):
        """Pontok adása másik usernek"""
        if points <= 0:
            await ctx.send("❌ Csak pozitív mennyiséget adhatsz!")
            return
        
        giver = self.db.get_user(ctx.author.id)
        
        if not giver:
            await ctx.send("❌ Nem található a fiókod!")
            return
        
        if giver['total_points'] < points:
            await ctx.send(f"❌ Nincs elég pontod! Egyenleged: {giver['total_points']}")
            return
        
        # Pontok levonása a küldőtől
        self.db.update_user_points(ctx.author.id, -points, f'Pont küldés: {member.name}')
        
        # Pontok hozzáadása a fogadóhoz
        receiver = self.db.get_user(member.id)
        if not receiver:
            self.db.create_user(member.id, str(member), 'free')
        
        self.db.update_user_points(member.id, points, f'Pont fogadás: {ctx.author.name}')
        
        await ctx.send(f"✅ {ctx.author.mention} küldött **{points}** pontot {member.mention} részére! 💸")

async def setup(bot):
    await bot.add_cog(Points(bot))
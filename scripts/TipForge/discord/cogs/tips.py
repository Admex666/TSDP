import discord
from discord.ext import commands
import config
from database import Database

class Tips(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = Database()
    
    @commands.command(name='posttip')
    @commands.has_permissions(administrator=True)
    async def post_tip(self, ctx, tier: str, match: str, tip: str, odds: float, confidence: int):
        """
        Admin parancs: Tipp posztolása
        Használat: !posttip free "Real Madrid vs Barcelona" "1X" 1.85 75
        """
        tier = tier.lower()
        if tier not in config.CHANNELS:
            await ctx.send(f"❌ Érvénytelen tier! Választható: {', '.join(config.CHANNELS.keys())}")
            return
        
        channel_id = config.CHANNELS[tier]
        channel = self.bot.get_channel(channel_id)
        
        if not channel:
            await ctx.send(f"❌ Nem található a csatorna: {tier}")
            return
        
        # Szín confidence alapján
        if confidence >= 80:
            color = discord.Color.green()
        elif confidence >= 60:
            color = discord.Color.gold()
        else:
            color = discord.Color.orange()
        
        embed = discord.Embed(
            title=f"⚽ {match}",
            description=f"**Tipp:** {tip}\n**Odds:** {odds}\n**Bizalom:** {confidence}%",
            color=color
        )
        embed.set_footer(text=f"Tier: {tier.upper()} | Posztolva: {ctx.author.name}")
        
        await channel.send(embed=embed)
        await ctx.send(f"✅ Tipp posztolva a #{channel.name} csatornába!")
    
    @commands.command(name='tipresult')
    @commands.has_permissions(administrator=True)
    async def tip_result(self, ctx, message_id: int, result: str):
        """
        Admin parancs: Tipp eredményének frissítése
        Használat: !tipresult 123456789 win/loss/pending
        """
        result = result.lower()
        if result not in ['win', 'loss', 'pending']:
            await ctx.send("❌ Érvénytelen eredmény! (win/loss/pending)")
            return
        
        try:
            message = await ctx.channel.fetch_message(message_id)
            
            if not message.embeds:
                await ctx.send("❌ Az üzenet nem tartalmaz embedet!")
                return
            
            embed = message.embeds[0]
            
            # Emoji és szín beállítása
            if result == 'win':
                emoji = "✅"
                color = discord.Color.green()
            elif result == 'loss':
                emoji = "❌"
                color = discord.Color.red()
            else:
                emoji = "⏳"
                color = discord.Color.orange()
            
            # Embed frissítése
            new_embed = discord.Embed(
                title=f"{emoji} {embed.title}",
                description=embed.description,
                color=color
            )
            new_embed.set_footer(text=f"{embed.footer.text} | Eredmény: {result.upper()}")
            
            await message.edit(embed=new_embed)
            await ctx.send(f"✅ Tipp eredménye frissítve: **{result.upper()}**")
            
        except Exception as e:
            await ctx.send(f"❌ Hiba: {e}")
    
    @commands.command(name='mytips')
    async def my_tips(self, ctx):
        """Saját tippek megtekintése"""
        embed = discord.Embed(
            title=f"📊 {ctx.author.name} tippjei",
            description="Itt láthatod a saját tippjeidet és azok eredményét.",
            color=discord.Color.blue()
        )
        embed.add_field(
            name="🎯 Funkció fejlesztés alatt",
            value="Hamarosan követheted a saját tippjeidet és ROI-d!",
            inline=False
        )
        await ctx.send(embed=embed)

async def setup(bot):
    await bot.add_cog(Tips(bot))
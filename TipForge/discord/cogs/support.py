import discord
from discord.ext import commands
import config
from database import Database

class Support(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = Database()
    
    @commands.command(name='ticket')
    async def create_ticket(self, ctx, *, issue: str = None):
        """
        Support ticket létrehozása
        Használat: !ticket Probléma leírása
        """
        # Support kategória vagy csatorna
        support_channel = ctx.guild.get_channel(config.CHANNELS.get('support', 0))
        
        if not support_channel:
            await ctx.send("❌ Support csatorna nem található!")
            return
        
        # Privát szál létrehozása
        thread_name = f"ticket-{ctx.author.name}-{ctx.message.id}"
        
        thread = await support_channel.create_thread(
            name=thread_name,
            auto_archive_duration=1440,  # 24 óra
            type=discord.ChannelType.private_thread
        )
        
        # User hozzáadása a thread-hez
        await thread.add_user(ctx.author)
        
        # Admin role ping (ha van)
        admin_role = ctx.guild.get_role(config.ROLES.get('admin', 0))
        admin_mention = admin_role.mention if admin_role else "@Admin"
        
        # Ticket üzenet
        embed = discord.Embed(
            title="🎫 Új Support Ticket",
            description=f"**Beküldő:** {ctx.author.mention}\n**Probléma:**\n{issue or 'Nincs megadva'}",
            color=discord.Color.blue()
        )
        embed.set_footer(text=f"Ticket ID: {ctx.message.id}")
        
        await thread.send(f"{admin_mention}", embed=embed)
        await thread.send(f"{ctx.author.mention}, egy admin hamarosan válaszol! 😊")
        
        # Visszajelzés a user számára
        await ctx.send(f"✅ Ticket létrehozva! Kérlek nézd meg: {thread.mention}")
    
    @commands.command(name='close')
    async def close_ticket(self, ctx):
        """
        Ticket lezárása (csak thread-ben működik)
        """
        if not isinstance(ctx.channel, discord.Thread):
            await ctx.send("❌ Ez a parancs csak ticket thread-ben használható!")
            return
        
        # Admin ellenőrzés vagy ticket létrehozó
        is_admin = ctx.author.guild_permissions.administrator
        is_owner = ctx.channel.name.startswith(f"ticket-{ctx.author.name}")
        
        if not (is_admin or is_owner):
            await ctx.send("❌ Nincs jogosultságod lezárni ezt a ticket-et!")
            return
        
        embed = discord.Embed(
            title="🔒 Ticket Lezárva",
            description=f"Lezárta: {ctx.author.mention}",
            color=discord.Color.red()
        )
        
        await ctx.send(embed=embed)
        await ctx.channel.edit(archived=True, locked=True)
    
    @commands.command(name='faq')
    async def faq(self, ctx):
        """Gyakori kérdések"""
        embed = discord.Embed(
            title="❓ Gyakori Kérdések (FAQ)",
            description="Itt találod a leggyakoribb kérdésekre a válaszokat:",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="💰 Hogyan szerezhetek pontokat?",
            value=(
                "• Üzenetek írása: 1 pont/üzenet\n"
                "• Reakciók: 1 pont/reakció\n"
                "• Nyerő tippek: 50 pont\n"
                "• Barátok meghívása: 100-2000 pont\n"
                "• Napi bonus: !daily parancs"
            ),
            inline=False
        )
        
        embed.add_field(
            name="🎯 Hogyan működnek a tier-ek?",
            value=(
                "**Free:** 0 pont (alap)\n"
                "**Basic:** 1000 pont\n"
                "**Standard:** 5000 pont\n"
                "**Premium:** 10000 pont\n"
                "**Elite:** 25000 pont\n\n"
                "A tier automatikusan frissül pont alapján!"
            ),
            inline=False
        )
        
        embed.add_field(
            name="🎁 Mit válthatok be pontokért?",
            value="Nézd meg a pontboltot: `!shop`",
            inline=False
        )
        
        embed.add_field(
            name="🤝 Hogyan működik a referral?",
            value=(
                "1. `!referral` parancs a kódért\n"
                "2. Barát használja: `!redeem KÓDOD`\n"
                "3. Mindketten kaptok pontokat!"
            ),
            inline=False
        )
        
        embed.add_field(
            name="📞 További segítség?",
            value="Használd a `!ticket` parancsot!",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='report')
    async def report_user(self, ctx, member: discord.Member, *, reason: str):
        """
        User jelentése
        Használat: !report @user Indoklás
        """
        # Support csatorna
        support_channel = ctx.guild.get_channel(config.CHANNELS.get('support', 0))
        
        if not support_channel:
            await ctx.send("❌ Support csatorna nem található!")
            return
        
        # Report embed
        embed = discord.Embed(
            title="⚠️ Felhasználó Jelentés",
            description=f"**Bejelentő:** {ctx.author.mention}\n**Jelentett:** {member.mention}",
            color=discord.Color.orange()
        )
        embed.add_field(name="Indok", value=reason, inline=False)
        embed.set_footer(text=f"Report ID: {ctx.message.id}")
        
        # Admin ping
        admin_role = ctx.guild.get_role(config.ROLES.get('admin', 0))
        admin_mention = admin_role.mention if admin_role else "@Admin"
        
        await support_channel.send(f"{admin_mention}", embed=embed)
        await ctx.send("✅ Jelentés elküldve az adminoknak!", delete_after=10)
        await ctx.message.delete()  # Eredeti üzenet törlése (privacy)
    
    @commands.command(name='suggestion')
    async def suggestion(self, ctx, *, suggestion: str):
        """
        Javaslat küldése
        Használat: !suggestion Az ötleted
        """
        # Feedback csatorna
        feedback_channel = ctx.guild.get_channel(config.CHANNELS.get('feedback', 0))
        
        if not feedback_channel:
            # Ha nincs feedback csatorna, support-ba megy
            feedback_channel = ctx.guild.get_channel(config.CHANNELS.get('support', 0))
        
        if not feedback_channel:
            await ctx.send("❌ Feedback csatorna nem található!")
            return
        
        embed = discord.Embed(
            title="💡 Új Javaslat",
            description=suggestion,
            color=discord.Color.gold()
        )
        embed.set_author(name=ctx.author.name, icon_url=ctx.author.display_avatar.url)
        embed.set_footer(text=f"Javaslat ID: {ctx.message.id}")
        
        msg = await feedback_channel.send(embed=embed)
        
        # Reakciók hozzáadása (szavazás)
        await msg.add_reaction("👍")
        await msg.add_reaction("👎")
        
        await ctx.send("✅ Javaslatod elküldve! Köszönjük! 💚")

async def setup(bot):
    await bot.add_cog(Support(bot))
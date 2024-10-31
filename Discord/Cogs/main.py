from discord.ext import commands
import discord
import json

with open('Discord/setting.json', 'r', encoding='utf8') as file:
    jfile = json.load(file)

class Main(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.command()
    async def join(self, ctx: commands.Context):
        voice = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if ctx.author.voice == None:
            embed = discord.Embed(
                title=">> 請先加入一個語音頻道！ <<",
                color = discord.Color.random()
            )
            await ctx.send(embed=embed)
        elif voice == None:
            channel = ctx.author.voice.channel
            await channel.connect()
            embed = discord.Embed(
                title=f">> 已加入語音頻道: {channel.name} <<",
                color = discord.Color.green()
            )
            await ctx.send(embed=embed)

    @commands.command()
    async def leave(self, ctx: commands.Context):
        voice = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        await voice.disconnect()
        channel = ctx.author.voice.channel
        embed = discord.Embed(
            title = F">> 已離開語音頻道: {channel.name} <<",
            color = discord.Color.red()
        )
        await ctx.send(embed=embed)

    @commands.command()
    async def ping(self, ctx: commands.Context):
        await ctx.send(f'{round(self.bot.latency, 3)} 秒')

    @commands.command()
    async def clear(self, ctx: commands.Context):
        await ctx.channel.purge(limit= 100)
        message = await ctx.channel.fetch_message(jfile['project_channel'])
        await message.delete()

async def setup(bot):
    await bot.add_cog(Main(bot))
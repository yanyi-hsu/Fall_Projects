from discord.ext import commands
import discord
import json

with open('Discord/setting.json', 'r', encoding='UTF8') as file:
    jfile = json.load(file)

playing_list = []

class React(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
    
    @commands.command()
    async def get_channel_id(self, ctx: commands.Context):
        # 遍歷伺服器中的所有頻道，並查找名稱匹配的頻道
        for channel in ctx.guild.channels:
            if channel .name == channel.name:
                await ctx.send(f'頻道_{channel.name}的ID: {channel.id}')
            else:
                await ctx.send(f'頻道_{channel.name} 沒找到')

    @commands.command()
    async def pause(self, ctx: commands.Context):
        voice = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if voice.is_playing():
            voice.pause()
        else:
            await ctx.send("Currently no audio is playing")
    
    @commands.command()
    async def resume(self, ctx: commands.Context):
        voice = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if voice.is_paused():
            voice.resume()
        else:
            await ctx.send("The audio is not pause")

    @commands.command()
    async def skip(self, ctx: commands.Context):
        voice = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        voice.stop()    

async def setup(bot):
    await bot.add_cog(React(bot))
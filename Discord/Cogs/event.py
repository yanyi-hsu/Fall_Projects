import discord
import json
from discord.ext import commands

with open('Discord/setting.json', 'r', encoding='UTF8') as file:
    jfile = json.load(file)

class Event(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        channel = self.bot.get_channel(jfile['project_channel'])
        await channel.send(f'{member} 已加入')
    
    @commands.Cog.listener()
    async def on_member_leave(self, member: discord.Member):
        channel = self.bot.get_channel(jfile['project_channel'])
        await channel.send(f'{member} 已離開')

    @commands.Cog.listener()
    async def on_message(self, msg: discord.Message):
        if msg.content == 'apple':
            await msg.channel.send('hi')

async def setup(bot):
    await bot.add_cog(Event(bot))
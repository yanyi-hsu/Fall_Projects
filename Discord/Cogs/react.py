import discord
import json
from discord.ext import commands

with open('Discord/setting.json', 'r', encoding='UTF8') as file:
    jfile = json.load(file)

class React(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def fall_pic(self, ctx):
        embed = discord.Embed()
        pic = embed.set_image(url=jfile['picture'])
        await ctx.send(embed = pic)

async def setup(bot):
    await bot.add_cog(React(bot))
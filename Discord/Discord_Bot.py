import discord
import json
from discord.ext import commands

with open('Discord/setting.json', 'r', encoding='UTF8') as file:
    jfile = json.load(file)

intent = discord.Intents.all()
bot = commands.Bot(command_prefix='$', intents=intent)

@bot.event
async def on_ready():
    print(f'Bot Identify --> {bot.user}')

    channel_id = jfile['project_channel']
    target_channel = bot.get_channel(channel_id)

    await target_channel.send('>> Service Online <<')

@bot.command()
async def ping(ctx):
    await ctx.send(f'{round(bot.latency*1000)} ms')


bot.run(jfile['TOKEN'])
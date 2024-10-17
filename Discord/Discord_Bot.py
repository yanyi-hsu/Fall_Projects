import discord
import asyncio
import json
import os
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
    embed = discord.Embed(
        title = '>> Service Online <<', 
        color = discord.Color.random()
    )
    await target_channel.send(embed=embed)

# 載入指令程式檔案
@bot.command()
async def load(ctx, extension):
    await bot.load_extension(f"Cogs.{extension}")
    await ctx.send(f"Loaded {extension} done.")

# 卸載指令檔案
@bot.command()
async def unload(ctx, extension):
    await bot.unload_extension(f"Cogs.{extension}")
    await ctx.send(f"UnLoaded {extension} done.")

# 重新載入程式檔案
@bot.command()
async def reload(ctx, extension):
    await bot.reload_extension(f"Cogs.{extension}")
    await ctx.send(f"ReLoaded {extension} done.")

async def load_extensions():
    for filename in os.listdir("Discord/Cogs"):
        if filename.endswith(".py"):
            await bot.load_extension(f"Cogs.{filename[:-3]}")

async def main():
    async with bot:
        await load_extensions()
        await bot.start(jfile['TOKEN'])

if __name__ == "__main__":
    asyncio.run(main())
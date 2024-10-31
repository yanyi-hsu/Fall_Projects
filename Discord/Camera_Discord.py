# nohup python3 main.py >/dev/null 2>&1 &
# source discord_bot/bin/activate
# sudo docker run -d --runtime=nvidia --gpus all --name alert_service discord_bot
import discord
from discord.ext import commands
from ultralytics import YOLO
import cv2
import torch
from datetime import datetime
import os
import asyncio
import io

print(torch.cuda.is_available())
device_count = torch.cuda.device_count()
device_current = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(device_current)
print(f"device count:{device_count}\ncurrent device:{device_current}\ndevice name:{device_name}")

# discord bot variable
intent = discord.Intents.default()
bot = commands.Bot(command_prefix = "$", intents=intent)

# YOLO variable
model = YOLO('yolov8m-pose.pt')

async def human_detect():
    monitor = cv2.VideoCapture('rtsp://Program:000000@192.168.0.102/stream1')
    last_detect:datetime = None
    delta = datetime.now() - datetime.now()
    while True:
        ret, frame = monitor.read()
        work_time = datetime.now()
        if(ret):
            start_cord_x = 190
            start_cord_y = 0
            end_cord_x = 255
            end_cord_y = 185
            crop_img = frame[start_cord_y:end_cord_y, start_cord_x:end_cord_x]#(y, x)
            result = model.predict(frame, verbose=False, conf=0.6)
            if(last_detect == None):
                pass
            else:
                delta = datetime.now() - last_detect

            if((result[0].keypoints.has_visible == True) and ((delta.seconds > 5) or (last_detect == None))):
                print(f'last: {last_detect}')
                print(f'delta:{delta}')
                channel_id = 1296137183606079518
                target_channel = bot.get_channel(channel_id)
                embed = discord.Embed(
                    title = 'Alert',
                    description = f"Someone show up\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    color = discord.Color.random()
                )
                try:
                    byte_img = cv2.imencode('.png', img=crop_img)[1]
                    io_img = io.BytesIO(byte_img.tobytes())
                    await target_channel.send(embed=embed, file=discord.File(fp=io_img, filename=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png"))
                    last_detect = datetime.now()
                except Exception as e:
                    print(e)
                # last_detect = datetime.now()
                # detect_time = datetime.now()
                # store_path = os.getcwd() + '/result'
                # print(store_path)
                # detect_result_dir = os.listdir(path=store_path)
                # if(len(detect_result_dir) == 0):
                #     print('store first')
                #     cv2.imwrite(f"{store_path}/{detect_time.strftime('%Y-%m-%d %H:%M:%S')}.png", crop_img)
                # else:
                #     #sort() return None, directly modify original list
                #     detect_result_dir.sort(reverse=True)
                #     pre_detect_time = datetime.strptime(detect_result_dir[0].split('.')[0], '%Y-%m-%d %H:%M:%S')
                #     delta = detect_time - pre_detect_time
                #     if(delta.seconds > 5):
                #         cv2.imwrite(f"{store_path}/{detect_time.strftime('%Y-%m-%d %H:%M:%S')}.png", crop_img)
        else:
            pass
        await asyncio.sleep(0.04)
'''
async def alert_message():
     while True:
         store_path = os.getcwd() + '/result'
         detect_result_dir = os.listdir(path=store_path)
         for file_name in detect_result_dir:
            if('.png' in file_name):
                channel_id = 1285874554044157973
                target_channel = bot.get_channel(channel_id)
                detect_time = file_name.split('.')[0]
                embed = discord.Embed(
                    title = 'Alert',
                    description = f'Someone show up\nTime: {detect_time}', 
                    color = discord.Color.random()
                )
                try:
                     await target_channel.send(embed=embed, file=discord.File(f'{store_path}/{file_name}'))
                     os.remove(f'{store_path}/{file_name}')
                except Exception as e:
                     print(e)
            else:
                pass
         await asyncio.sleep(5)
'''
@bot.event
async def on_ready():
    #await bot.add_cog(TaskDetect(bot))
    print(f'login identity: {bot.user}')
    channel_id = 1296137183606079518
    target_channel = bot.get_channel(channel_id)
    embed = discord.Embed(
        title = 'Service Online', 
        description = 'Now conf is 0.6',
        color = discord.Color.random()
    )
    await target_channel.send(embed=embed) 

async def main():
    await asyncio.gather(
        bot.start('MTI5NjEzNzE4MzYwNjA3OTUxOA.GOV_Nl.Wt-A54TTbydF_f_s3Jp0cX4XoltUciPaRrFk3I', reconnect=True),
        human_detect()
        #, alert_message()
    )
asyncio.run(main())
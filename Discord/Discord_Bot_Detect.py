from ultralytics import YOLO
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from torch import nn
import cv2
import torch

from discord.ext import commands
import discord
import asyncio
import json


with open('Discord/setting.json', 'r', encoding='UTF8') as file:
    jfile = json.load(file)

intent = discord.Intents.all()
bot = commands.Bot(command_prefix='$', intents=intent)

class NeuralNet(nn.Module):
    def __init__(self, input_size=34, hidden_size1=64, hidden_size2=32, num_classes=2):
        super(NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(34, 64),
            # nn.BatchNorm1d(64),  
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(64, 32), 
            # nn.BatchNorm1d(32),
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, 2)
        )
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)
    
pose_model = YOLO('model/yolov8m-pose.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_inference = NeuralNet().to(device)
model_inference.load_state_dict(torch.load('Senior_code/classification.pth', map_location=device))
model_inference.eval()

encoder = LabelEncoder()
encoder.fit(['fall', 'not fallen'])

async def detect_fall():
    monitor = cv2.VideoCapture('rtsp://Program:000000@192.168.0.102/stream1')
    last_detect: datetime = None
    delta = datetime.now() - datetime.now()
    channel_id = jfile['project_channel']
    project_channel = await bot.fetch_channel(channel_id)
    
    while monitor.isOpened():
        ret, frame = monitor.read()
        if ret:
            results = pose_model.predict(frame, verbose=False, stream=True, imgsz=(640, 384), stream_buffer=True)
            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy().flatten()
                if keypoints.size == 34:
                    input_tensor = torch.from_numpy(keypoints).to(device)
                    with torch.no_grad():
                        out = model_inference(input_tensor)
                        if out.dim() == 1:
                            out = out.unsqueeze(0)
                        _, predicted_class = torch.max(out, 1)

                    predicted_label = encoder.inverse_transform(predicted_class.cpu().numpy())

                    delta = datetime.now() - last_detect if last_detect else datetime.now() - datetime.now()
                    if 'fall' in predicted_label and ((delta.seconds > 15) or (last_detect is None)):
                        print(f'last: {last_detect}')
                        print(f'delta:{delta}')
                        
                        embed = discord.Embed(
                            title='有人跌倒囉！',
                            description=f"時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                            color=discord.Color.blue()
                        )
                        embed.set_image(url=jfile['picture'])
                        await project_channel.send(embed=embed)
                        last_detect = datetime.now()
                        '''
                        try:
                            byte_img = cv2.imencode('.png', img=frame)[1]
                            io_img = io.BytesIO(byte_img.tobytes())
                            await ctx.send(embed=embed, file=discord.File(fp=io_img, filename=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png"))
                            last_detect = datetime.now()
                        except Exception as e:
                            print(e)
                        '''

            await asyncio.sleep(0.1)
        else:
            pass

@bot.event
async def on_ready():
    print(f'Bot Identify --> {bot.user}')
    channel_id = jfile['project_channel']
    target_channel = bot.get_channel(channel_id)
    embed = discord.Embed(
        title = '>> 服務已上線 <<',
        description = "已開始偵測",  
        color = discord.Color.green()
    )
    await target_channel.send(embed=embed)

async def main():
    await asyncio.gather(
        bot.start(jfile['TOKEN'], reconnect=True), 
        detect_fall()
    )

asyncio.run(main())
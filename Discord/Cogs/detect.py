from sklearn.preprocessing import LabelEncoder
from torch import nn
from discord.ext import commands
from ultralytics import YOLO
from datetime import datetime

import numpy as np
import cv2
import discord
import json
import torch
import asyncio
import io

with open('Discord/setting.json', 'r', encoding='UTF8') as file:
    jfile = json.load(file)

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

class Detect(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.monitor_active = False  # 控制相機運行狀態標志
        self.monitor = None  # 初始化摄像头对象      

    @commands.command()
    async def close(self, ctx: commands.Context):
        if self.monitor and self.monitor.isOpened():
            self.monitor.release()  # 释放摄像头资源
            cv2.destroyAllWindows()  # 关闭窗口
            self.monitor_active = False  # 重置标志
            await ctx.send("相機已關閉")
        else:
            await ctx.send("相機未開啟")

    @commands.command()
    async def detect_fall(self, ctx: commands.Context):
        if self.monitor_active:  # 检查是否已经在运行，避免重复启动
            await ctx.send("相機已經在運行")
            return

        try:
            self.monitor = cv2.VideoCapture('rtsp://Program:000000@192.168.0.102/stream1')
            self.monitor_active = True  # 启动标志
            last_detect: datetime = None
            delta = datetime.now() - datetime.now()

            if self.monitor.isOpened():
                await ctx.send("相機已開啟")
            else:
                await ctx.send("Error：無法開啟相機")
                self.monitor_active = False  # 確保在失敗時重置標志
                return

            while self.monitor.isOpened() and self.monitor_active:
                ret, frame = self.monitor.read()
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

                            #for (x, y) in keypoints.reshape(-1, 2):
                                #cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                            #cv2.putText(frame, f'Predicted: {predicted_label[0]}', (900, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            if 'fall' in predicted_label and ((delta.seconds > 15) or (last_detect is None)):
                                print(f'last: {last_detect}')
                                print(f'delta:{delta}')
                                embed = discord.Embed(
                                    title='有人跌倒囉！',
                                    description=f"時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                    color=discord.Color.blue()
                                )
                                embed.set_image(url=jfile['picture'])
                                await ctx.send(embed=embed)
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
                    #cv2.imshow('Yolov8 with Classfier', frame)
                    await asyncio.sleep(0.1) 
                else:
                    await ctx.send("相機連接發生錯誤")
                    break

        except Exception as e:
            await ctx.send(f"檢測錯誤：{str(e)}")
        finally:
            # 确保资源在任意情况下都被释放
            if self.monitor:
                self.monitor.release()
            cv2.destroyAllWindows()
            self.monitor_active = False  # 关闭监控后，重置标志


async def setup(bot):
    await bot.add_cog(Detect(bot))
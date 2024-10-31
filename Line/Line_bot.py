from flask import Flask, request, abort
from pyngrok import ngrok

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO
from datetime import datetime
from torch import nn
import numpy as np
import torch
import json
import cv2
import time

with open('Line/setting.json', 'r', encoding='UTF8') as file:
    jfile = json.load(file)

with open('Line/user.json', 'r', encoding='UTF8') as file:
    juser = json.load(file)

app = Flask(__name__) 

port = "5000"

line_bot_api = LineBotApi(jfile['token'])              # 確認 token 是否正確
handler = WebhookHandler(jfile['secret'])              # 確認 secret 是否正確

user_dict = {}

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

@app.route("/", methods=['POST'])
def linebot():
    signature = request.headers['X-Line-Signature']      # 加入回傳的 headers
    body = request.get_data(as_text=True)                    # 取得收到的訊息內容
    try:
        handler.handle(body, signature)                      # 綁定訊息回傳的相關資訊
    except InvalidSignatureError:
        abort(400)
    return 'OK'    

@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id
    profile = line_bot_api.get_profile(user_id)
    display_name =profile.display_name
    # 儲存 user_id 或做其他處理
    print(f"新加入的使用者ID: {user_id}")
    user_dict['user_id'][f'user_id_{display_name}'] = user_id
    with open('Line/user.json', 'w', encoding='utf8') as file:
        json.dump(user_dict, file, indent=4, ensure_ascii=False)

    # 回應訊息給使用者
    welcome_message = f"感謝{display_name}加我為好友！！！"
    message = "打出 >> 開始偵測 <<，即開始偵測。"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome_message))
    line_bot_api.push_message(user_id, TextSendMessage(text=message))
    line_bot_api.push_message(user_id, StickerSendMessage(package_id=11538, sticker_id=51626494))

@handler.add(MessageEvent, message=TextMessage)
def Detect_Fall(event):
    user_id = event.source.user_id
    profile = line_bot_api.get_profile(user_id)
    display_name = profile.display_name
    if user_id in juser["user_id"].values():
        pass
    else:
        user_dict['user_id'][f'user_id_{display_name}'] = user_id
        with open('Line/user.json', 'w', encoding='utf8') as file:
            json.dump(user_dict, file, indent=4, ensure_ascii=False)
    
    if event.message.text == "開始偵測":
        message = "已開始偵測"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))
        monitor = cv2.VideoCapture('rtsp://Program:000000@192.168.0.102/stream1')
        last_detect:datetime = None
        delta = datetime.now() - datetime.now()
        try:
            while monitor.isOpened():
                ret, frame = monitor.read()
                work_time = datetime.now()
                if ret:
                    results = pose_model.predict(frame, verbose=False)

                    if(last_detect == None):
                        pass
                    else:
                        delta = datetime.now() - last_detect

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

                            if 'fall' in predicted_label and (delta.seconds > 30 or last_detect == None):
                                print(f'last: {last_detect}')
                                print(f'delta:{delta}')
                                line_bot_api.push_message(user_id, TextSendMessage(text=f"有人跌倒了！\n時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
                                line_bot_api.push_message(user_id, ImageSendMessage(original_content_url=jfile['picture'], preview_image_url=jfile['picture']))
                                last_detect = datetime.now()
                            
                    #cv2.imshow('Yolov8m-pose with Classification', frame) 
                else:
                    print('>> Monitor Error <<')
                    break
        except Exception as e:
            print(e)
        finally:           
            monitor.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    public_url = ngrok.connect(port).public_url
    print(f"Ngrok Tunnel: {public_url} --> http://127.0.0.1:{port}")
    app.run()
from flask import Flask, request, abort
from pyngrok import ngrok

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO
from datetime import datetime
from torch import nn
import torch
import json
import cv2
import threading

with open('Line/setting.json', 'r', encoding='UTF8') as file:
    jfile = json.load(file)

with open('Line/user.json', 'r', encoding='UTF8') as file:
    juser = json.load(file)
    

app = Flask(__name__) 

port = "5000"

line_bot_api = LineBotApi(jfile['token'])
handler = WebhookHandler(jfile['secret'])

message_sent = False

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
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    global message_sent
    if not message_sent:
        user_id = juser["user_id"]["user_id_許晏益"]
        online_message = f">> 服務已上線 <<\n已開始偵測"
        
        '''
        for user_id in juser['user_id'].values():
            line_bot_api.push_message(user_id, TextSendMessage(text=online_message))
            line_bot_api.push_message(user_id, StickerSendMessage(package_id=11538, sticker_id=51626530))
        ''' 

        line_bot_api.push_message(user_id, TextSendMessage(text=online_message))
        line_bot_api.push_message(user_id, StickerSendMessage(package_id=11538, sticker_id=51626530))
        message_sent = True
        #threading.Thread(target=Detect_Fall).start()
           
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'    

@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id
    profile = line_bot_api.get_profile(user_id)
    display_name =profile.display_name

    print(f"新加入的使用者ID: {display_name} : {user_id}")
    juser['user_id'][f'user_id_{display_name}'] = user_id
    with open('Line/user.json', 'w', encoding='utf8') as file:
        json.dump(juser, file, indent=4, ensure_ascii=False)

    welcome_message = f"感謝{display_name}加我為好友！！！"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome_message))
    line_bot_api.push_message(user_id, StickerSendMessage(package_id=11538, sticker_id=51626494))

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    profile = line_bot_api.get_profile(user_id)
    display_name =profile.display_name
    message = f"{display_name}\n謝謝你願意找我聊天\n但我不會聊天喔！"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))
    line_bot_api.push_message(user_id, StickerSendMessage(package_id=11539, sticker_id=52114110))
    juser['user_id'][f'user_id_{display_name}'] = user_id
    with open('Line/user.json', 'w', encoding='utf8') as file:
        json.dump(juser, file, indent=4, ensure_ascii=False)
'''
def Detect_Fall():
    monitor = cv2.VideoCapture('rtsp://Program:000000@192.168.0.102/stream1')
    last_detect:datetime = None
    delta = datetime.now() - datetime.now()
    user_id = juser["user_id"]["user_id_許晏益"]
    while monitor.isOpened():
        ret, frame = monitor.read()
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

                    if 'fall' in predicted_label and (delta.seconds > 30 or last_detect == None):
                        print(f'last: {last_detect}')
                        print(f'delta:{delta}')
                        line_bot_api.push_message(user_id, TextSendMessage(text=f"有人跌倒了！\n時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
                        line_bot_api.push_message(user_id, ImageSendMessage(original_content_url=jfile['picture'], preview_image_url=jfile['picture']))
                        last_detect = datetime.now()       
        else:
            pass
'''

if __name__ == "__main__":
    public_url = ngrok.connect(port).public_url
    print(f"Ngrok Tunnel: {public_url} --> http://127.0.0.1:{port}")
    app.run()
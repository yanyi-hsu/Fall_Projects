import numpy as np
import pandas as pd
import torch
import cv2
from torch import nn
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder

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
encoder.fit(['fallen', 'not fall'])

monitor = cv2.VideoCapture('rtsp://Program:000000@192.168.0.102/stream1')

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

                for (x, y) in keypoints.reshape(-1, 2):
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(frame, f'Predicted: {predicted_label[0]}', (900, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow('Yolov8m-pose with Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Monitor error')
        break

monitor.release()
cv2.destroyAllWindows()

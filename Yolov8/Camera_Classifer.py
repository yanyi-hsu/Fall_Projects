import numpy as np
import pandas as pd
import torch
import cv2
from torch import nn
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

pose_model = YOLO('yolov8m-pose.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 34
hidden_size = 256
num_classes = 2
model_inference = NeuralNet(input_size, hidden_size, num_classes).to(device)
model_inference.load_state_dict(torch.load('Classifer.pt', map_location=device))
model_inference.eval()

encoder = LabelEncoder()
encoder.fit(['fall', 'not fallen'])

monitor = cv2.VideoCapture('rtsp://Program:000000@192.168.0.102/stream1')
video_width = int(monitor.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(monitor.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('classifer_video.mp4', fourcc, 24.0, (video_width, video_height))


while monitor.isOpened():
    ret, frame = monitor.read()
    if ret:
        results = pose_model.predict(frame, verbose=False, stream=True, imgsz=(640, 384), stream_buffer=True)

        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy().astype(np.float32)
            if keypoints.shape[0] > 0:
                keypoints = keypoints.reshape(-1)

                if keypoints.size == input_size:
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

            image_array = result.plot(im_gpu=1080)
            
        video_writer.write(frame)
        cv2.imshow('Yolov8m-pose with Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Monitor error')
        break

monitor.release()
video_writer.release()
cv2.destroyAllWindows()

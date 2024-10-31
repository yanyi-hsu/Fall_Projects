import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from pydantic import BaseModel
from PIL import Image
import matplotlib.pyplot as plt

class Keypoints(BaseModel):
    NOSE:int = 0, 
    LEFT_EYE:int = 1, 
    RIGHT_EYE:int = 2
    LEFT_EAR:int = 3
    RIGHT_EAR:int = 4
    LEFT_SHOULDER:int = 5
    RIGHT_SHOULDER:int = 6
    LEFT_ELBOW:int = 7
    RIGHT_ELBOW:int = 8
    LEFT_WRIST:int = 9
    RIGHT_WRIST:int = 10
    LEFT_HIP:int = 11
    RIGHT_HIP:int = 12
    LEFT_KNEE:int = 13
    RIGHT_KNEE:int = 14
    LEFT_ANKLE:int = 15
    RIGHT_ANKLE:int = 16

get_keypoint = Keypoints()

def extract_keypoints(keypoint) -> list:
    # nose
    nose_x, nose_y = keypoint[get_keypoint.NOSE]
    # eye
    left_eye_x, left_eye_y = keypoint[get_keypoint.LEFT_EYE]
    right_eye_x, right_eye_y = keypoint[get_keypoint.RIGHT_EYE]
    # ear
    left_ear_x, left_ear_y = keypoint[get_keypoint.LEFT_EAR]
    right_ear_x, right_ear_y = keypoint[get_keypoint.RIGHT_EAR]
    # shoulder
    left_shoulder_x, left_shoulder_y = keypoint[get_keypoint.LEFT_SHOULDER]
    right_shoulder_x, right_shoulder_y = keypoint[get_keypoint.RIGHT_SHOULDER]
    # elbow
    left_elbow_x, left_elbow_y = keypoint[get_keypoint.LEFT_ELBOW]
    right_elbow_x, right_elbow_y = keypoint[get_keypoint.RIGHT_ELBOW]
    # wrist
    left_wrist_x, left_wrist_y = keypoint[get_keypoint.LEFT_WRIST]
    right_wrist_x, right_wrist_y = keypoint[get_keypoint.RIGHT_WRIST]
    # hip
    left_hip_x, left_hip_y = keypoint[get_keypoint.LEFT_HIP]
    right_hip_x, right_hip_y = keypoint[get_keypoint.RIGHT_HIP]
    # knee
    left_knee_x, left_knee_y = keypoint[get_keypoint.LEFT_KNEE]
    right_knee_x, right_knee_y = keypoint[get_keypoint.RIGHT_KNEE]
    # ankle
    left_ankle_x, left_ankle_y = keypoint[get_keypoint.LEFT_ANKLE]
    right_ankle_x, right_ankle_y = keypoint[get_keypoint.RIGHT_ANKLE]
    
    return [
        nose_x, nose_y,
        left_eye_x, left_eye_y,
        right_eye_x, right_eye_y,
        left_ear_x, left_ear_y,
        right_ear_x, right_ear_y,
        left_shoulder_x, left_shoulder_y,
        right_shoulder_x, right_shoulder_y,
        left_elbow_x, left_elbow_y,
        right_elbow_x, right_elbow_y,
        left_wrist_x, left_wrist_y,
        right_wrist_x, right_wrist_y,
        left_hip_x, left_hip_y,
        right_hip_x, right_hip_y,
        left_knee_x, left_knee_y,
        right_knee_x, right_knee_y,        
        left_ankle_x, left_ankle_y,
        right_ankle_x, right_ankle_y
    ]

df = pd.read_csv('./keypoint.csv')

# class NeuralNet(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = torch.nn.Linear(input_size, hidden_size) 
#         self.l2 = torch.nn.Linear(hidden_size, num_classes)
#         self.relu = torch.nn.ReLU()
        
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         return out
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

# hidden_size = 256
encoder = LabelEncoder()
y_label = df['label']
# print(y_label)
y = encoder.fit_transform(y_label)
# print(encoder.classes_)
# print(y)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)#num_classes
# print(len(class_weights))

model_inference = NeuralNet()

model_inference.load_state_dict(torch.load(f='./classification.pth', weights_only=True))

model = YOLO('yolo11s-pose.pt')
# img:cv2.typing.MatLike = cv2.imread('./fall-18-cam0-rgb/fall-18-cam0-rgb-039.png')
# img1:cv2.typing.MatLike = cv2.imread('./fall-21-cam0-rgb/fall-21-cam0-rgb-052.png')
img = Image.open('./test014.jpg')
result = model.predict(img.resize(size=(640, 480)), verbose=False)[0]
print(f'is have visible?: {result.keypoints.has_visible}')
result_keypoints = result.keypoints.xyn.cpu().numpy()

feature = []
for kp in result_keypoints:
    if len(kp) != 0:
        keypoints_list = extract_keypoints(kp)
        feature.append(keypoints_list)

input = torch.from_numpy(np.array(feature[0]))
with torch.no_grad():
    class_name = ['fallen', 'not_fall']
    for i in feature:
        input = torch.from_numpy(np.array(i))
        outputs = model_inference(input)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        print(probabilities)
        print(f'result: {class_name[predictions]}')

        plt.title(f'{class_name[predictions]}')
        plt.imshow(img)

        plt.show()
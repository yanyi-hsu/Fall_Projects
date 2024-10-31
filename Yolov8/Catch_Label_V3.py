import cv2
import os
import math
import shutil
import csv
from ultralytics import YOLO
from pydantic import BaseModel

class GetKeypoint(BaseModel):
        NOSE:           int = 0
        LEFT_EYE:       int = 1
        RIGHT_EYE:      int = 2
        LEFT_EAR:       int = 3
        RIGHT_EAR:      int = 4
        LEFT_SHOULDER:  int = 5
        RIGHT_SHOULDER: int = 6
        LEFT_ELBOW:     int = 7
        RIGHT_ELBOW:    int = 8
        LEFT_WRIST:     int = 9
        RIGHT_WRIST:    int = 10
        LEFT_HIP:       int = 11
        RIGHT_HIP:      int = 12
        LEFT_KNEE:      int = 13
        RIGHT_KNEE:     int = 14
        LEFT_ANKLE:     int = 15
        RIGHT_ANKLE:    int = 16

get_keypoint = GetKeypoint()   

def extract_keypoint(keypoint):
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

    return [nose_x, nose_y,
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
            right_ankle_x, right_ankle_y]

def remove_root(labels_path, results_path):
    if os.path.exists(labels_path) or os.path.exists(results_path):
        shutil.rmtree(labels_path)
        shutil.rmtree(results_path)
    os.makedirs(labels_path)
    os.makedirs(results_path)

def detect_not_fallen(source_list):
    head = source_list[0:2]
    left_shoulder = source_list[10:12]
    right_shoulder = source_list[12:14]
    left_wrist = source_list[18:20]
    right_wrist = source_list[20:22]
    left_hip = source_list[22:24]
    right_hip = source_list[24:26]
    left_knee = source_list[26:28]
    right_knee = source_list[28:30]
    left_ankle = source_list[30:32]
    right_ankle = source_list[32:34]
        
    def meanorleftright(left, right):
        if left[1] != 0 and right[1] != 0:
            return [round((left[0]+right[0])/2, 8), round((left[1]+right[1])/2, 8)]
        else:
            return [max(left[0], right[0]), max(left[1], right[1])]
        
    def three_point_angle(hip, knee, ankle):
        inner_product = (hip[0] - knee[0]) * (ankle[0] - knee[0]) +  (hip[1] - knee[1]) * (ankle[1] - knee[1])
        AB_mag = math.sqrt((knee[0] - hip[0])**2 + (knee[1] -hip[1])**2)
        BC_mag = math.sqrt((ankle[0] - knee[0])**2 + (ankle[1] - knee[1])**2)
        cos_theta = inner_product / (AB_mag * BC_mag)
        cos_theta = max(-1, min(1, cos_theta))
        angle = math.acos(cos_theta)
        angle = math.degrees(angle)
        return round(angle)
    
    def calculate_angle(shoulder, hip):
        if right_hip[0]+left_hip[0] < right_shoulder[0]+left_shoulder[0] :
            delta_x = shoulder[0] - hip[0]
        else:
            delta_x = hip[0] - shoulder[0]
        delta_y = hip[1] - shoulder[1]
        angle = math.atan2(delta_y, delta_x)
        angle = math.degrees(angle)
        return round(angle)
    
    shoulder = meanorleftright(left_shoulder, right_shoulder)
    wrist = meanorleftright(left_wrist, right_wrist)
    hip = meanorleftright(left_hip, right_hip)
    knee = meanorleftright(left_knee, right_knee)
    ankle = meanorleftright(left_ankle, right_ankle)

    data = [head[1], shoulder[1], wrist[1], hip[1], knee[1], ankle[1]]
    zero_data = [zero for zero in data if zero == 0]

    angle2 = calculate_angle(shoulder, hip)

    if ankle[1] == 0 and knee[1] != 0:
        ankle[0] = knee[0]
        ankle[1] = knee[1]+0.2
        angle1 = three_point_angle(hip, knee, ankle)
    elif ankle[1] == 0 and knee[1] == 0:
        angle1 = 0
    else:
        angle1 = three_point_angle(hip, knee, ankle)

    if head[1] != 0 and ankle[1] !=0 and head[1] > ankle[1]:
        return "fall"
    elif len(zero_data) >= 3:
        return "fall"
    elif hip[1] < (knee[1]-0.05) and 157 <= angle1 <= 190 and angle2 > 73:
        return "not fallen" 
    elif (knee[1]-0.2) <= hip[1] <= (knee[1]+0.3) and angle1 <= 164 and wrist[1] < ankle[1] and angle2 >= 42:
        return "not fallen" 
    elif angle1 == 0 and angle2 > 60 and wrist[1] < hip[1]:
        return "not fallen" 
    else:
        return "fall"
       
dataset_csv = []

model = YOLO('yolov8m-pose.pt')

dataset_root = 'Yolov8/fall_dataset/archive/train/images'
label_root = 'Yolov8/fall_dataset/archive/train/labels'
result_root = 'Yolov8/fall_dataset/archive/results/train'

remove_root(label_root, result_root)
    
for images in os.listdir(dataset_root):
    image_path = os.path.join(dataset_root, images)

    image_info = cv2.imread(image_path)
    height, width = image_info.shape[:2]

    results = model.predict(image_info, save=False)[0]
    results_keypoint = results.keypoints.xyn.cpu().numpy()
    results_conf = results.boxes.conf.cpu().numpy()

    for result_keypoint, result_conf in zip(results_keypoint, results_conf):
        if len(result_keypoint) == 17 and result_conf >= 0.7:
            label_list = extract_keypoint(result_keypoint)
            label_list.insert(0, detect_not_fallen(label_list))
            dataset_csv.append(label_list)

header = [
    'label',
    # nose
    'nose_x',
    'nose_y',
    # left eye
    'left_eye_x',
    'left_eye_y',
    # right eye
    'right_eye_x',
    'right_eye_y',
    # left ear
    'left_ear_x',
    'left_ear_y',
    # right ear
    'right_ear_x',
    'right_ear_y',
    # left shoulder
    'left_shoulder_x',
    'left_shoulder_y',
    # right sholder
    'right_shoulder_x',
    'right_shoulder_y',
    # left elbow
    'left_elbow_x',
    'left_elbow_y',
    # rigth elbow
    'right_elbow_x',
    'right_elbow_y',
    # left wrist
    'left_wrist_x',
    'left_wrist_y',
    # right wrist
    'right_wrist_x',
    'right_wrist_y',
    # left hip
    'left_hip_x',
    'left_hip_y',
    # right hip
    'right_hip_x',
    'right_hip_y',
    # left knee
    'left_knee_x',
    'left_knee_y',
    # right knee
    'right_knee_x',
    'right_knee_y',
    # left ankle
    'left_ankle_x',
    'left_ankle_y',
    # right ankle
    'right_ankle_x',
    'right_ankle_y'
]

with open('Yolov8/Classifer_Keypoint.csv', 'w', encoding='UTF8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(dataset_csv)

        
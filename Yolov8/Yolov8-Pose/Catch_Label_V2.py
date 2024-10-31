import cv2
import os
import math
import shutil
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
    head = source_list[4:6]
    left_shoulder = source_list[14:46]
    right_shoulder = source_list[16:18]
    left_wrist = source_list[22:24]
    right_wrist = source_list[24:26]
    left_hip = source_list[26:28]
    right_hip = source_list[28:30]
    left_knee = source_list[30:32]
    right_knee = source_list[32:34]
    left_ankle = source_list[34:36]
    right_ankle = source_list[36:38]
        
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
        return 0
    elif len(zero_data) >= 3:
        return 0
    elif hip[1] < (knee[1]-0.05) and 157 <= angle1 <= 190 and angle2 > 73:
        return 1
    elif (knee[1]-0.2) <= hip[1] <= (knee[1]+0.3) and angle1 <= 164 and wrist[1] < ankle[1] and angle2 >= 42:
        return 1
    elif angle1 == 0 and angle2 > 60 and wrist[1] < hip[1]:
        return 1 
    else:
        return 0
        
root_list = ['train', 'valid']
error_data = []

model = YOLO('yolov8m-pose.pt')

for root in root_list:
    dataset_root = 'Yolov8/fall_dataset/archive/' + root + '/images'
    label_root = 'Yolov8/fall_dataset/archive/' + root + '/labels'
    result_root = 'Yolov8/fall_dataset/archive/results/' + root

    remove_root(label_root, result_root)
    
    for images in os.listdir(dataset_root):
        image_path = os.path.join(dataset_root, images)
        label_name = image_path.split('/')[-1]
        label_name = label_name.split('.jpg')[0]
        image_name = label_name.split('_')[-1]
        for name in image_name:
            if name.isdigit():
                image_name = image_name.replace(name, '')

        image_info = cv2.imread(image_path)
        height, width = image_info.shape[:2]

        results = model.predict(image_info, save=False)[0]
        results.save(f'{result_root}/{label_name}.jpg')

        results_box = results.boxes.xywhn.cpu().numpy()
        results_keypoint = results.keypoints.xyn.cpu().numpy()
        results_conf = results.boxes.conf.cpu().numpy()

        for result_keypoint, result_box, result_conf in zip(results_keypoint, results_box, results_conf):
            conf = round(result_conf, 2)
            if conf >= 0.7:
                label_list = result_box.tolist()
                label_list.extend(extract_keypoint(result_keypoint))

                if (image_name == "fall" and detect_not_fallen(label_list) == 1):
                    error_data.append(f"{label_name}/Class: {detect_not_fallen(label_list)}")
                elif (image_name == "not fallen" and detect_not_fallen(label_list) == 0):
                    error_data.append(f"{label_name}/Class: {detect_not_fallen(label_list)}")

                label_list.insert(0, detect_not_fallen(label_list))

                with open( f'{label_root}/{label_name}.txt', 'a', encoding='UTF8') as file:
                    list_to_str = ' '.join(str(value) for value in label_list)
                    file.write(list_to_str+'\n')


with open('error_dataset.txt', 'w', encoding='UTF8') as file:
    for error in error_data:
        file.write(error + '\n')


        
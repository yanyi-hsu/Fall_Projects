import cv2
import os
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

def extract_box(x, y, w, h):
    return [round(x / width, 8), round(y / height, 8), round(w / width, 8), round(h / height, 8)]

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

def remove_file(labels_path, results_path):
    for label, result in zip(os.listdir(labels_path), os.listdir(results_path)):
        os.remove(os.path.join(labels_path, label))
        os.remove(os.path.join(results_path, result))


root_list = ['train', 'test', 'valid']

for i in range(3):
    dataset_root = 'Yolo v8/fall_dataset/archive/'+ root_list[i] + '/images'
    label_root = 'Yolo v8/fall_dataset/archive/' + root_list[i] + '/labels'
    result_root = 'Yolo v8/fall_dataset/archive/results/' + root_list[i]

    remove_file(label_root, result_root)

    model = YOLO('yolov8m-pose.pt')
  
    for images in os.listdir(dataset_root):
        image_path = os.path.join(dataset_root, images)
        label_name = image_path.split('/')[-1]
        label_name = label_name.split('.jpg')[0]
        image_name = label_name
        label_name = label_name.split('_')[-1]
        
        for label in label_name:
            if label.isdigit():
                label_name = label_name.replace(label, '')
            
        image_info = cv2.imread(image_path)
        height, width = image_info.shape[:2]

        results = model.predict(image_info, save=False)[0]
        results.save(filename=f'{result_root}/{image_name}.jpg')

        results_box = results.boxes.xywh.cpu().numpy()
        results_keypoint = results.keypoints.xyn.cpu().numpy()

        for result_keypoint, result_box in zip(results_keypoint, results_box):
            #if len(result_keypoint) == 17:
            x, y, w, h = result_box
            label_list = extract_box(x, y, w, h)
            label_list.extend(extract_keypoint(result_keypoint))

            if label_name == "fall":
                label_list.insert(0, 0)
            elif label_name == "not fallen":
                label_list.insert(0, 1)
                    
            with open( f'{label_root}/{image_name}.txt', 'a', encoding='UTF8') as file:
                list_to_str = ' '.join(str(value) for value in label_list)
                file.write(list_to_str+'\n')



        
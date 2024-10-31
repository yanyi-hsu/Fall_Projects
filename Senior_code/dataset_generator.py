from skimage.metrics import structural_similarity as ssim
import numpy
import cv2
from ultralytics import YOLO
from pydantic import BaseModel
import csv
import os
from tqdm import tqdm
from PIL import Image, ImageFile
import imagehash

def ssim_compare(standard_img:cv2.typing.MatLike, fall_img:cv2.typing.MatLike) -> numpy.float64:
    width = standard_img.shape[1]
    height = standard_img.shape[0]

    fall_img_width = fall_img.shape[1]
    fall_img_height = fall_img.shape[0]
    
    if(fall_img_height < height):
        fall_img = cv2.rotate(fall_img, cv2.ROTATE_90_CLOCKWISE)
    else:
        pass

    fall_img_resize = cv2.resize(fall_img, (width, height))

    ssim_score = ssim(standard_img, fall_img_resize, full=False)
    return ssim_score

def hamming_distance_phash(standard_img:ImageFile, fall_img:ImageFile) -> int:
    standard_hash = imagehash.phash(standard_img)
    fall_hash = imagehash.phash(fall_img)

    distance = standard_hash - fall_hash
    
    return distance

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

fall_dir_list:list = []
for dir in os.listdir():
    if dir.startswith('fall'):
        fall_dir_list.append(dir)
    else:
        continue
fall_dir_list.sort()

dataset_csv:list = []

wait_for_predict:list = []

model = YOLO('yolo11s-pose.pt')

#get all dir of dataset

for dir in tqdm(fall_dir_list):
    standard_img = Image.open(f'./{dir}/{dir}-001.png')
    standard_predict_result = model.predict(standard_img, verbose=False)[0]
    boxes_standard = tuple(standard_predict_result.boxes.xyxy.tolist()[0])
    crop_standard = standard_img.crop(box=boxes_standard)
    # not_fall_img:cv2.typing.MatLike = cv2.imread(f'./{dir}/{dir}-001.png')
    # not_fall_result = model.predict(not_fall_img, verbose=False)[0]
    # not_fall_img_gray = cv2.cvtColor(not_fall_img, cv2.COLOR_BGR2GRAY)
    # not_fall_pt = [int(i) for i in not_fall_result.boxes.xyxy.tolist()[0]]
    # crop_not_fall_img = not_fall_img[not_fall_pt[1]:not_fall_pt[3], not_fall_pt[0]:not_fall_pt[2]]
    # crop_not_fall_img_gray = cv2.cvtColor(crop_not_fall_img, cv2.COLOR_BGR2GRAY)

    #foreach image in dir
    img_list = os.listdir(f'./{dir}')
    img_list.sort()
    for img in img_list:
        # fall_img:cv2.typing.MatLike = cv2.imread(f'./{dir}/{img}')
        # fall_result = model.predict(fall_img, verbose=False)[0]
        fall_img = Image.open(f'./{dir}/{img}')
        fall_predict_result = model.predict(fall_img, verbose=False)[0]
        if fall_predict_result.keypoints.has_visible:
            boxes_fall = tuple(fall_predict_result.boxes.xyxy.tolist()[0])
            crop_fall = fall_img.crop(box=boxes_fall)
            # fall_pt = [int(i) for i in fall_result.boxes.xyxy.tolist()[0]]

            # crop_fall_img = fall_img[fall_pt[1]:fall_pt[3], fall_pt[0]:fall_pt[2]]
            # crop_fall_img_gray = cv2.cvtColor(crop_fall_img, cv2.COLOR_BGR2GRAY)
            # fall_img_gray = cv2.cvtColor(fall_img, cv2.COLOR_BGR2GRAY)
            # ssim_score:numpy.float64 = ssim_compare(standard_img=not_fall_img_gray, fall_img=fall_img_gray)
            # label = ''
            # if(ssim_score >= 0.29):
            #     label = 'not fall'
            # else:
            #     label = 'fall'
            label = ''
            hamming_distance_full = hamming_distance_phash(standard_img=standard_img, fall_img=fall_img)
            hamming_distance_crop = hamming_distance_phash(standard_img=crop_standard, fall_img=crop_fall)
            if hamming_distance_full >= 8:
                if hamming_distance_crop >= 29:
                    label = 'fallen'
                else:
                    label = 'not fall'
            else:
                if hamming_distance_crop >= 29:
                    label = 'fallen'
                else:
                    label = 'not fall'

            result_keypoints = fall_predict_result.keypoints.xyn.cpu().numpy()
            # print(result_keypoints)
            for kp in result_keypoints:
                keypoints_list = extract_keypoints(kp)
                keypoints_list.insert(0, label)
                keypoints_list.insert(1, img)
                dataset_csv.append(keypoints_list)
                    
header = [
    'label',
    'image',
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

with open('keypoint.csv', 'w', encoding='UTF-8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    writer.writerows(dataset_csv)
import os
import cv2
import shutil
import albumentations as A

# 定義一個函數來進行圖片增強
def augment_image(dataset_path, augment_path):
    # 設定使用的增強方法（如調整大小、亮度對比、色相、飽和度等）
    transform = A.Compose([
        A.Resize(640, 640),  # 將圖片大小調整為 640x640

        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),  # 隨機調整亮度和對比度

        A.HueSaturationValue(hue_shift_limit=21, sat_shift_limit=0, val_shift_limit=0, p=1.0),  # 調整色相值
        
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=0, p=1.0),  # 調整飽和度值
        
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0, p=1.0),  # 再次隨機調整亮度，對比度保持不變
        
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0, p=0.5),  # 再次隨機調整亮度，機率為50%
        
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)  # 加入隨機高斯噪聲，機率為50%
    ])

    # 用來區分不同圖片的計數器，分別處理摔倒和未摔倒的奇數和偶數圖片
    fall_odd = 1
    notfallen_odd = 1
    fall_even = 2
    notfallen_even = 2
    
    # 遍歷資料集路徑中的所有圖片
    for images in os.listdir(dataset_path):
        images_path = os.path.join(dataset_path, images)  # 取得圖片完整路徑
        images_name = images_path.split('/')[-1]  # 取得圖片名稱
        images_name = images_name.split('.jpg')[0]  # 去掉圖片的副檔名 ".jpg"

        # 移除圖片名稱中的數字，保留文字部分以識別是摔倒還是未摔倒
        for image_name in images_name:
            if image_name.isdigit():
                images_name = images_name.replace(image_name, '')

        image = cv2.imread(images_path)  # 讀取圖片
        
        # 使用增強方法生成兩張增強後的圖片（奇數與偶數）
        augmented_odd = transform(image=image)['image']
        augmented_even = transform(image=image)['image']

        # 根據圖片名稱來決定是摔倒還是未摔倒，並存儲增強後的圖片
        if images_name == 'fall':
            cv2.imwrite(augment_path + '/augment_' + images_name + str(fall_odd) + '.jpg', augmented_odd)  # 存儲奇數圖片
            cv2.imwrite(augment_path + '/augment_' + images_name + str(fall_even) + '.jpg', augmented_even)  # 存儲偶數圖片
            fall_odd += 2  # 奇數圖片的序號增加2
            fall_even += 2  # 偶數圖片的序號增加2
        elif images_name == 'not fallen':
            cv2.imwrite(augment_path + '/augment_' + images_name + str(notfallen_odd) + '.jpg', augmented_odd)
            cv2.imwrite(augment_path + '/augment_' + images_name + str(notfallen_even) + '.jpg', augmented_even)
            notfallen_odd += 2
            notfallen_even += 2

# 定義一個函數來複製圖片文件到目標路徑
def copy_file(source_path, arrive_path):
    image_list = os.listdir(source_path)  # 獲取來源路徑中的所有文件
    for image in image_list:
        shutil.copy(os.path.join(source_path, image), arrive_path)  # 將圖片複製到目標路徑

# 定義一個函數來刪除指定的路徑，如果路徑不存在，則重新創建該路徑
def remove_root(images_path):
    if os.path.exists(images_path):
        shutil.rmtree(images_path)  # 刪除指定的路徑及其內容
    os.makedirs(images_path)  # 創建新路徑

# 設定資料集、增強圖片、訓練集、驗證集的根路徑
dataset_root = 'Yolo v8/fall_dataset/archive/dataset'
augment_root = 'Yolo v8/fall_dataset/archive/augment'
train_root = 'Yolo v8/fall_dataset/archive/train/images'
val_root = 'Yolo v8/fall_dataset/archive/val/images'
valid_root = 'Yolo v8/fall_dataset/archive/valid/images'

# 刪除並重建增強圖片、訓練集和驗證集的路徑
remove_root_list = [augment_root, train_root, valid_root]
for root in remove_root_list:
    remove_root(root)

# 對資料集和驗證集進行圖片增強處理
augment_image(dataset_root, augment_root)
augment_image(val_root, valid_root)

# 將原始資料集和增強後的圖片複製到訓練集路徑中
copy_file(dataset_root, train_root)
copy_file(augment_root, train_root)

# 將驗證集中的圖片複製到驗證集路徑中
for valid in os.listdir(val_root):
    shutil.copy(os.path.join(val_root + '/', valid), valid_root)

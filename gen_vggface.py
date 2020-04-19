"""
Created on Sat Mar  7 12:21:41 2020

@author: a
"""

import os
import cv2
import numpy as np
from detect_align import process_data
from tqdm import tqdm
import threading

ori_datapath = '/media/a/HDD/Face_Proj/vgg_face_dataset/images'
new_datapath = '/media/a/HDD/Face_Proj/vgg_face_dataset/new_images'

files = os.listdir(ori_datapath)



def download_and_save(path, warped):
    cv2.imwrite(path, warped)
        
        
for i in tqdm(range(len(files))):
    file_name = files[i]
    if not os.path.exists(os.path.join(new_datapath, file_name)):
        os.makedirs(os.path.join(new_datapath, file_name))  
    imgs = os.listdir(os.path.join(ori_datapath, file_name))
    for j in range(len(imgs)):
        img_name = os.path.join(ori_datapath, file_name, imgs[j])

        try:
            warped = process_data(cv2.imread(img_name))
            path = os.path.join(new_datapath, file_name, imgs[j])
            t = threading.Thread(target=download_and_save, args=(path, warped))        
            t.start()
#            cv2.imwrite(, warped)
        except Exception as e:
            continue

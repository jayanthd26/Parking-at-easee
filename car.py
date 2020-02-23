
import os
import numpy as np
import cv2
import mrcnn.config
 
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

VIDEO_SOURCE = "test_images/parking.mp4"


model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())


model.load_weights(COCO_MODEL_PATH, by_name=True)


parked_car_boxes = None




frame = cv2.imread('y1.jpg') 
#print(frame.shape)

rgb_image = frame[:, :, ::-1]


results = model.detect([rgb_image], verbose=0)


r = results[0]


car_boxes = get_car_boxes(r['rois'], r['class_ids'])

print("Cars found in frame of video:")
car_boxes=car_boxes[0:6]
box_l=[]
box_r=[]
for box in car_boxes:
    print("Car: ", box)

    y1, x1, y2, x2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0,255), 3)
    y,x,r=rgb_image.shape
    if x1<x/2:
        box_l.append(box)
    else:
        box_r.append(box)
lineThickness=4
for i in range(len(box_l)-1):
    print(i)
    if box_l[i][2]<box_l[i+1][1]:
        print("l")
        cv2.line(frame, (box_l[i][1], box_l[i][2]), (box_l[i+1][1], box_l[i+1][2]), (0,255,0), lineThickness)
        cv2.line(frame, (box_l[i][1], box_l[i][2]), (box_l[i][3], box_l[i][2]), (0,255,0), lineThickness)
        cv2.line(frame, (box_l[i+1][3], box_l[i+1][2]), (box_l[i][3], box_l[i][2]), (0,255,0), lineThickness)
        cv2.line(frame, (box_l[i+1][3], box_l[i+1][2]), (box_l[i+1][1], box_l[i+1][2]), (0,255,0), lineThickness)
        
for i in range(len(box_r)-1):
    if box_r[i][1]>box_r[i+1][2]:
        print("r")
        cv2.line(frame, (box_r[i][1], box_r[i][2]), (box_r[i+1][1], box_r[i+1][2]), (0,255,0), lineThickness)
        cv2.line(frame, (box_r[i][1], box_r[i][2]), (box_r[i][3], box_r[i][2]), (0,255,0), lineThickness)
        cv2.line(frame, (box_r[i+1][3], box_r[i+1][2]), (box_r[i][3], box_r[i][2]), (0,255,0), lineThickness)
        cv2.line(frame, (box_r[i+1][3], box_r[i+1][2]), (box_r[i+1][1], box_r[i+1][2]),(0,255,0), lineThickness)
        
    

   
cv2.imwrite('fr.png',frame)
cv2.destroyAllWindows()# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.scatter(x11,y11)
plt.scatter(x22,y22)

plt.show()


box=np.sort(car_boxes,axis=0)


frame1 = cv2.imread('x1.png') 
y1, x1, y2, x2 = car_boxes[0]
cv2.rectangle(frame1, (x1, 0), (x2, y2), (0, 255, 0), 1)
y1, x1, y2, x2 = car_boxes[4]
cv2.rectangle(frame1, (x1 ,y1), (x2, y2), (0, 255, 0), 1)
cv2.imwrite('fr1.png',frame1)
cv2.destroyAllWindows()# 




import matplotlib.pyplot as plt
import glob
path="/home/ubuntu/ml/infosys/objects_2011_a/labeldata/*"
files=glob.glob(path)
for file  in files:
    frame = cv2.imread(file) 
    print(frame.shape)
    plt.imshow(frame)
    plt.show()
    i=int(input("1 or 0"))
    if i==1:
        rgb_image = frame[:, :, ::-1]
        
        
        results = model.detect([rgb_image], verbose=0)
        
        
        r = results[0]
        
        
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        
        print("Cars found in frame of video:")
        
           
        for box in car_boxes:
            print("Car: ", box)
        
            y1, x1, y2, x2 = box
        
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
           
        plt.imshow(frame)
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()# -*- coding: utf-8 -*-
        

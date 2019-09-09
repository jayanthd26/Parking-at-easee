# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 04:20:35 2019

@author: Nanda Krishna K S
"""


import os
import numpy as np
import cv2
import mrcnn.config
 
 
import requests
import cv2
import numpy as np

url="http://192.168.137.140:8080/shot.jpg"

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

if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)
# Directory of images to run detection on
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
model.load_weights(COCO_MODEL_PATH, by_name=True)
parked_car_boxes = None





from flask import Flask,request,render_template,jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process',methods= ['POST'])
def process():
    Info = (request.form['info']).lstrip()
    print("string:"+Info)
    print("helloo")
    
    print("uosh")
    global parked_car_boxes
    global url
    global model
    
    
    while True:
        img_resp=requests.get(url)
        img_arr=np.array(bytearray(img_resp.content))
        img=cv2.imdecode(img_arr,-1)
        frame=img.copy()
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        rgb_image = img[:, :, ::-1]
    
        
        results = model.detect([rgb_image], verbose=0)
    
        
        r = results[0]
    
        
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])
    
        print("Cars found in frame of video:")
    
       
        for box in car_boxes:
            print("Car: ", box)
            y1, x1, y2, x2 = box    
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        mid_x=int(frame.size/(len(frame)*3*2))
        l_over_lap=0
        left=[]
        right=[]
        r_over_lap=0
        for box in car_boxes:
                print("Car: ", box)
                y1, x1, y2, x2 = box
                if(x1<mid_x):
                    if(len(left)!=0):
                        if(left[0][2]<x1):
                            l_over_lap=1;
                    left.append([x1,y1,x2,y2])
                else:
                    if(len(right)!=0):
                        if(right[0][0]>x2):
                            r_over_lap=1;
                    right.append([x1,y1,x2,y2])
        print(l_over_lap,r_over_lap)
        
        
        if(r_over_lap==1):
            px1,py1,px2,py2=right[0][0],right[0][3],right[0][2],right[0][3]
            px3,py3,px4,py4=right[1][0],right[1][3],right[1][2],right[1][3]
            pts = np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]], dtype=np.int32)
        
            polygon=np.array([[(px1, py1), (px2, py2), (px4, py3), (px3, py4)]])
            cv2.fillPoly(frame,polygon,(0,255,0))
            cv2.imwrite("static/n1.png",frame)
            break
            return jsonify({})
            
        if(l_over_lap==1):
            px1,py1,px2,py2=left[0][0],left[0][3],left[0][2],left[0][3]
            px3,py3,px4,py4=left[1][0],left[1][3],left[1][2],left[1][3]
            pts = np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]], dtype=np.int32)
        
            polygon=np.array([[(px1, py1), (px2, py2), (px4, py3), (px3, py4)]])
            cv2.fillPoly(frame,polygon,(0,255,0))
            cv2.imwrite("static/n1.png",frame)
            break
            return jsonify({})
    
        print("END")    
    
    return jsonify({})

    
    
    
    
    
if __name__ == '__main__':
    app.run()

























       
       
'''      
    rgb_image = img[:, :, ::-1]

   
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
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0,255), 3)
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
            cv2.line(img, (box_l[i][1], box_l[i][2]), (box_l[i+1][1], box_l[i+1][2]), (0,255,0), lineThickness)
            cv2.line(img, (box_l[i][1], box_l[i][2]), (box_l[i][3], box_l[i][2]), (0,255,0), lineThickness)
            cv2.line(img, (box_l[i+1][3], box_l[i+1][2]), (box_l[i][3], box_l[i][2]), (0,255,0), lineThickness)
            cv2.line(img, (box_l[i+1][3], box_l[i+1][2]), (box_l[i+1][1], box_l[i+1][2]), (0,255,0), lineThickness)
           
    for i in range(len(box_r)-1):
        if box_r[i][1]>box_r[i+1][2]:
            print("r")
            cv2.line(img, (box_r[i][1], box_r[i][2]), (box_r[i+1][1], box_r[i+1][2]), (0,255,0), lineThickness)
            cv2.line(img, (box_r[i][1], box_r[i][2]), (box_r[i][3], box_r[i][2]), (0,255,0), lineThickness)
            cv2.line(img, (box_r[i+1][3], box_r[i+1][2]), (box_r[i][3], box_r[i][2]), (0,255,0), lineThickness)
            cv2.line(img, (box_r[i+1][3], box_r[i+1][2]), (box_r[i+1][1], box_r[i+1][2]),(0,255,0), lineThickness)
           
       
 for box in car_boxes:
        print("Car: ", box)
        y1, x1, y2, x2 = box    
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
   
    mid_x=int(frame.size/(len(frame)*3*2))
    l_over_lap=0
    left=[]
    right=[]
    r_over_lap=0
    for box in car_boxes:
            print("Car: ", box)
            y1, x1, y2, x2 = box
            if(x1<mid_x):
                if(len(left)!=0):
                    if(left[0][2]<x1):
                        l_over_lap=1;
                left.append([x1,y1,x2,y2])
            else:
                if(len(right)!=0):
                    if(right[0][0]>x2):
                        r_over_lap=1;
                right.append([x1,y1,x2,y2])
    print(l_over_lap,r_over_lap)
   
   
    if(r_over_lap==1):
        px1,py1,px2,py2=right[0][0],right[0][3],right[0][2],right[0][3]
        px3,py3,px4,py4=right[1][0],right[1][3],right[1][2],right[1][3]
        pts = np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]], dtype=np.int32)
   
        polygon=np.array([[(px1, py1), (px2, py2), (px4, py3), (px3, py4)]])
        cv2.fillPoly(frame,polygon,(0,255,0))
       
    if(l_over_lap==1):
        px1,py1,px2,py2=left[0][0],left[0][3],left[0][2],left[0][3]
        px3,py3,px4,py4=left[1][0],left[1][3],left[1][2],left[1][3]
        pts = np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]], dtype=np.int32)
   
        polygon=np.array([[(px1, py1), (px2, py2), (px4, py3), (px3, py4)]])
        cv2.fillPoly(frame,polygon,(0,255,0))


   
'''

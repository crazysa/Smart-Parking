import RPi.GPIO as GPIO
import socket
import numpy as np
import threading
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


def imagedetection():
	

	# Root directory of the project
	ROOT_DIR = os.path.abspath("../")

	# Import Mask RCNN
	sys.path.append(ROOT_DIR)  # To find local version of the library
	import mrcnn.utils
	import mrcnn.model as modellib
	from mrcnn import visualize
	# Import COCO config
	sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
	import coco

	%matplotlib inline 

	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	# Download COCO trained weights from Releases if needed
	if not os.path.exists(COCO_MODEL_PATH):
		utils.download_trained_weights(COCO_MODEL_PATH)

	# Directory of images to run detection on
	IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    
	class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = InferenceConfig()
	config.display()
	
	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)
	
	
		# COCO Class names
#	 Index of the class in the list is its ID. For example, to get ID of
	# the teddy bear class, use: class_names.index('teddy bear')
	class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
			   
	# Load a random image from the images folder
	file_names = next(os.walk(IMAGE_DIR))[2]
	filename = os.path.join(IMAGE_DIR ,"car.jpg")

	#image = skimage.io.imread(filename)
	image = cv2.imread(filename,color)
	# Run detection
	results = model.detect([image], verbose=1)

	# Visualize results
	r = results[0]
	
	overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, result['rois'])
	for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
		max_IoU_overlap = np.max(overlap_areas)
		
		if max_IoU_overlap < 0.85:
					return 0
					
					
					
	visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
	
def initialidealimage():
	# Root directory of the project
	ROOT_DIR = os.path.abspath("../")

	# Import Mask RCNN
	sys.path.append(ROOT_DIR)  # To find local version of the library
	import mrcnn.utils
	import mrcnn.model as modellib
	from mrcnn import visualize
	# Import COCO config
	sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
	import coco

	%matplotlib inline 

	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	# Download COCO trained weights from Releases if needed
	if not os.path.exists(COCO_MODEL_PATH):
		utils.download_trained_weights(COCO_MODEL_PATH)

	# Directory of images to run detection on
	IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    
	class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = InferenceConfig()
	config.display()
	
	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)
	
	
		# COCO Class names
#	 Index of the class in the list is its ID. For example, to get ID of
	# the teddy bear class, use: class_names.index('teddy bear')
	class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
			   
	# Load a random image from the images folder
	file_names = next(os.walk(IMAGE_DIR))[2]
	filename = os.path.join(IMAGE_DIR ,"car.jpg")

	#image = skimage.io.imread(filename)
	image = cv2.imread(filename,color)
	# Run detection
	results = model.detect([image], verbose=1)

	# Visualize results
	r = results[0]
	
	return r['rois']
	


def check():
    
    while True:
        
        loop_counter=0
        for i in a:
            state=GPIO.input(i[2])
            if state is 1:
                pname='P1'+str(loop_counter+1)
                torun='select led status from Slots where slot_no='+str(pname)
                s.send(torun.encode())
                status=s.recv(1024).decode()
                print('Status',status)
                if status is 0:
                    
                    proper=imagedetection()
                    
                    if proper is 1:
                        print('name',pname)
                        torun='update Slots set led_status=1 where slot_no='+str(pname)
                        s.send(torun.encode())
                        GPIO.output(i[0],True)
                    else:
                        print('inproper name',pname)
                        torun='update Slots set led_status=2 where slot_no='+str(pname)
                        s.send(torun.encode())
                        GPIO.output(i[1],True)
                elif status>0:
                    print('bye',pname)
                    torun='update Slots set led_status=0 where slot_no='+str(pname)
                    s.send(torun.encode())
                    GPIO.output(i[0],False)
                    GPIO.output(i[1],False)
            loop_counter=loop_counter+1

if __name__== "__main__":
    
	image = cv2.imread("path to the ideal image")
	
	parked_car_boxes = initialidealimage()#get_car_boxes#get_car_boxes(r['rois'], r['class_ids']) #provide a function here than for ideal box of parked car
    
    GPIO.setmode(GPIO.board)
    a=np.array([
            [0,1,2,3,4,5],
            [6,7,8,100,100,100]
            ])
   
    outlist=[0,1,6,7,5]
    inlist=[2,3,4,8]
    GPIO.setup(outlist,GPIO.OUT)
    GPIO.setup(inlist,GPIO.IN)
    
    ref=['P11','P12']
    loop_count=0
    
    t=threading.Thread(target=check)
    
    print('Initiating...')
    s=socket.socket()
    host='192.168.1.3 or whatever ip it is'
    port=12345
    s.connect((host,port))
    
    s.send('delete from NWorking'.encode())
    '''for i in a:
        state=GPIO.input(i[2])
        if state is 0:
            print(ref[loop_count])
            torun='insert into NWorking values('+str(ref[loop_count])+')'
            s.send(torun.encode())
            loop_count=loop_count+1'''
    
    t.start()
    
    while True:
        state1=GPIO.input(3)
        state2=GPIO.input(4)
        while state1 == state2:
            print('Idle')
        
        if(state1 is 1 and state2 is 0):
            print('Vehicle entered')
            s.send('select counter from Lots'.encode())
            count=s.recv(1024)
            count=count+1
            torun='update Lots set counter ='+str(count)+' where lot_no ='+str(1)
            s.send(torun.encode())
            
        elif(state2 is 1 and state1 is 0):
            print('Vehicle left')
            s.send('select counter from Lots'.encode())
            count=s.recv(1024)
            count=count-1
            torun='update Lots set counter ='+str(count)+' where lot_no ='+str(1)
            s.send(torun.encode())
    
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:26:13 2020

@author: tejashree
"""
# python encode_faces_yolo.py --dataset dataset --encodings encodings.pickle --detection-method hog

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import numpy as np
from utils import *
from sklearn.ensemble import RandomForestClassifier

def createXY(data):
    x=[]
    y=[]
    for name,enc in data.items():
        for i in enc:
            x.append(i)
            y.append(name)
    return x,y

def yolo_face_detection(frame):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    reboxes=[]
    for j in faces:
        reboxes.append([j[1],j[0]+j[2],j[1]+j[3],j[0]])
    return reboxes

def resnet50_encodings(img,boxes):
	out=[]
	for box in boxes:
		top,right,bottom,left=box
		face=img[top:bottom,left:right]
		face=cv2.resize(face,(224,224))
		face=face.reshape(224,224,3)
		face=face.astype('float32')
		samples = np.expand_dims(face, axis=0)
		# prepare the face for the model, e.g. center pixels
		samples = preprocess_input(samples, version=2)
		out.append(model.predict(samples))
	return out
		

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="yolo",
	help="face detection model to use: either `hog` or `cnn` or yolo")
ap.add_argument('--model-cfg', type=str, default='modules/cfg/yolov3-face.cfg',
                    help='path to config file')
ap.add_argument('--model-weights', type=str,
                    default='modules/model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
args = vars(ap.parse_args())

if args['detection_method']=="yolo":
    net = cv2.dnn.readNetFromDarknet(args['model_cfg'], args['model_weights'])
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
data = {}
if os.path.isfile(args['encodings']):
	data=pickle.loads(open(args['encodings'],'rb').read())

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
	if args['detection_method']!="yolo":
		boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
	else:
		boxes = yolo_face_detection(image) 

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes,num_jitters=2)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		if data.get(name,0)==0:
			data[name]=[]
		data[name].append(encoding)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] creating RandomForest classifier...")
clf = RandomForestClassifier()
x,y = createXY(data)
clf.fit(x,y)
print("[INFO] writing model to file...")
path = args['encodings']
index = path.rfind('/')+1
cpath = path[:index]+'clf/'+path[index:]
pickle.dump(clf,open(cpath,'wb'))
print("[INFO] Done written to:",cpath)

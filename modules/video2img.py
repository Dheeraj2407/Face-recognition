import os
import cv2
import sys
import time
import argparse

# Arguement parsing
######################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--video',required=True, type=str,help='path to video file')
parser.add_argument('--name',required=True, type=str,help='name of person in video')
parser.add_argument('--rotate',default=None,type=int,help='rotation if needed, (0 = ROTATE_90_CLOCKWISE, 1 = ROTATE_180_CLOCKWISE, 2 = ROTATE_90_COUNTERCLOCKWISE)')

args = parser.parse_args()

######################################################################

print('[i] Path to video file: ', args.video)

######################################################################
# Checking video path
if not os.path.exists(args.video):
	print('!!!!Invalid video path terminating process....')
	sys.exit(0)

######################################################################
# Checking person name uniqueness
if os.path.exists('tempdataset/'+args.name):
	print('!!!!Name already exists terminating process....')
	sys.exit(0)
else:
	os.system('mkdir tempdataset/'+args.name)
######################################################################
# Checking for roatation parameter
if args.rotate!=None:
	print('[i] Image will be rotated')
else:
	print('[i] Image won\'t be rotated')
######################################################################
# Opening Video
cap=cv2.VideoCapture(args.video)

# Reading frames and saving them to folder

while True:
	has_frame,frame=cap.read()
	if not has_frame:
		break	
	if args.rotate!=None:
		#print(args.rotate)
		frame=cv2.rotate(frame,args.rotate)
			
	fname=str(int(time.time()))+'.jpg'
	cv2.imwrite('tempdataset/'+args.name+'/'+fname,frame)
	time.sleep(0.05)

print("==>Done")


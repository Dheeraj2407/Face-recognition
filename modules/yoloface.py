# *******************************************************************

# Usage example:  python yoloface.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python yoloface.py --video samples/subway.mp4 --output-dir outputs/
#                 python yoloface.py --src 1 --output-dir outputs/


import argparse
import sys
import os
import imutils
from utils import *


#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='modules/cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='modules/model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default=None,
                    help='path to image file')
parser.add_argument('--video', type=str, default=None,
                    help='path to video file')
parser.add_argument('--rotate', type=int, default=None,
					help='rotation if needed, (0 = ROTATE_90_CLOCKWISE, 1 = ROTATE_90_CLOCKWISE, 2 = 						ROTATE_90_COUNTERCLOCKWISE)')
parser.add_argument('--encodings', type=str, default='encodings/encodings.pickle',required=True,
                    help='path to face encodings file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()
#####################################################################
# loading facial encodings for recognition
load_encodings(args.encodings)

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('[i] Path to image file: ', args.image)
print('[i] Path to video file: ', args.video)
print('###########################################################\n')

# check outputs directory
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def _main():
    waitingTime=0
    wind_name = 'Face Recognition'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    output_file = ''

    if args.image:
        if not os.path.isfile(args.image):
            print("[!] ==> Input image file {} doesn't exist".format(args.image))
            sys.exit(1)
        cap = cv2.VideoCapture(args.image)
        output_file = args.image[:-4].rsplit('/')[-1] + '_yoloface.jpg'
    elif args.video:
        if not os.path.isfile(args.video):
            print("[!] ==> Input video file {} doesn't exist".format(args.video))
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4].rsplit('/')[-1] + '_yoloface.avi'
    else:
        # Get data from the camera
        cap = cv2.VideoCapture(args.src)

    # Get the video writer initialized to save the output video
    if not args.image:
        video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                       cv2.VideoWriter_fourcc(*'MJPG'),
                                       cap.get(cv2.CAP_PROP_FPS), (
                                           round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),True)
        print('Videowriter:',video_writer.isOpened())
        waitingTime=1

    while True:

        has_frame, frame = cap.read()
        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
            cv2.waitKey(1000)
            break
        #frame=imutils.resize(frame,800)
        # Rotate frame if needed
        if args.rotate!=None:
        	print('[i] Rotating')
        	frame=cv2.rotate(frame,args.rotate)
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces)),"dims:",frame.shape)
        print('#' * 60)
        
        # Normalize frame
        #frame=cv2.normalize(frame,None,25,255,cv2.NORM_MINMAX)
        # recognize faces
        recognize_face(frame,faces)
        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)

        # Save the output to file
        if args.image:
            cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
        elif video_writer!=None:
            print("Frame written")
            video_writer.write(frame.astype(np.uint8))
        if frame.shape[1]>frame.shape[0]:
            frame=imutils.resize(frame,width=640)
        else:
            frame=imutils.resize(frame,height=480)
        cv2.imshow(wind_name, frame)

        key = cv2.waitKey(waitingTime)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()
    if not args.image:video_writer.release()
    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()

import datetime
import numpy as np
import cv2
import pickle
import face_recognition
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

# face encoding data
data = None
# RandomForest Classifier
clf = None
# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,COLOR_WHITE, 1)


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        #left, top, right, bottom = refined_box(left, top, width, height)
        # draw_predict(frame, confidences[i], left, top, left + width,
        #              top + height)
        #draw_predict(frame, confidences[i], left, top, right, bottom)
    return final_boxes


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._num_frames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._num_frames / self.elapsed()

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom

#dlib face recognition
def load_encodings(fname):
    global data,clf
    data = pickle.loads(open(fname, "rb").read())
    path = fname
    index = path.rfind('/')+1
    cpath = path[:index]+'clf/'+path[index:]
    clf = pickle.load(open(cpath,"rb"))
def recognize_face(frame,boxes):
    #Converting boxes according to face_recognition
    reboxes = []
    nboxes = []
    wrange = range(frame.shape[1])
    hrange = range(frame.shape[0])
    for j in boxes:
        if j[0] in wrange and j[0]+j[2] in wrange and j[1] in hrange and j[1]+j[3] in hrange:
            reboxes.append([j[1],j[0]+j[2],j[1]+j[3],j[0]])
            nboxes.append(j)
    #Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, reboxes,num_jitters=5)
    names = []
    # loop over the facial embeddings
    for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
        count = {}
        prob_list = list(clf.predict_proba([encoding])[0])
        sorted_proba = prob_list.copy()
        sorted_proba.sort(reverse=True)
        name = 'Unknown'
        for i in sorted_proba:
        	if i<0.5:
        		break
        	index = prob_list.index(i)
        	matches = face_recognition.compare_faces(data[clf.classes_[index]],encoding,0.55)
        	if matches.count(True)>0.6*len(data[clf.classes_[index]]):
        		name = clf.classes_[index]
        		break	
        names.append(name)

	# loop over the recognized faces
    for ((x, y, w, h), name) in zip(nboxes, names):
		# draw the predicted face name on the image
        cv2.rectangle(frame, (x, y), (x+w, y+h),
			(0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 35), (x+w, y), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y - 6), font, 1, (255, 255, 255), 1)
    print("\nFaces recognized:",names)
    
    

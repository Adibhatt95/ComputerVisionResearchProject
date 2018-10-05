######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using

MODEL_NAME = 'inference_graphVersion1'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
#set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 6

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)
img=Image.open('checkSize.jpg')
width, height = img.size
print(width)
print(height)
def clr():
    os.system('cls' if os.name=='nt' else 'clear')
flagSide = 'left'
while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    if len(classes[0] > 0):
        listOf4Indexes = np.where(classes[0] == 4)
        #print(listOf4Indexes)
    # print(len(classes[0]))
    # print(len(listOf4Indexes[0]))
    # print('len(boxes)={} len(boxes[0])={} len(boxes[0][0]={}'.format(len(boxes),len(boxes[0]),len(boxes[0][0])))
    c = np.zeros(shape=(1,len(classes[0])-len(listOf4Indexes[0])))
    b = np.zeros((1,len(classes[0])-len(listOf4Indexes[0]),4))
    s = np.zeros(shape=(1,len(classes[0])-len(listOf4Indexes[0])))
    for FourIndex in listOf4Indexes:
        # classes[0]=np.delete(classes[0],FourIndex
        # boxes[0]=np.delete(boxes[0],FourIndex)
        # scores[0]=np.delete(scores[0],FourIndex)
        c[0]=np.delete(classes[0],FourIndex)
        b[0]=np.delete(boxes,FourIndex,axis=1)
        s[0]=np.delete(scores[0],FourIndex)

    if len(c[0] > 0):
        listOf4Indexes2 = np.where(c[0] == 2)
        #print(listOf4Indexes)
    # print(len(classes[0]))
    # print(len(listOf4Indexes[0]))
    # print('len(boxes)={} len(boxes[0])={} len(boxes[0][0]={}'.format(len(boxes),len(boxes[0]),len(boxes[0][0])))
    c1 = np.zeros(shape=(1,len(c[0])-len(listOf4Indexes2[0])))
    b1 = np.zeros((1,len(c[0])-len(listOf4Indexes2[0]),4))
    s1 = np.zeros(shape=(1,len(c[0])-len(listOf4Indexes2[0])))
    for FourIndex in listOf4Indexes2:
        # classes[0]=np.delete(classes[0],FourIndex
        # boxes[0]=np.delete(boxes[0],FourIndex)
        # scores[0]=np.delete(scores[0],FourIndex)
        c1[0]=np.delete(c[0],FourIndex)
        b1[0]=np.delete(b,FourIndex,axis=1)
        s1[0]=np.delete(s[0],FourIndex)

    # print(len(c[0]))
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(b1),
        np.squeeze(c1).astype(np.int32),
        np.squeeze(s1),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.85)


    #print(boxes[0][1])
    if classes[0][0] == 3 and scores[0][0] > 0.90:
        print('I see the Keys')
        if width * boxes[0][0][1] <= width/2:
            flagSide = 'right'
            #imageName = 'left-'+imageName
            #print('width/2={} boxes[0][1]={}'.format(width/2,boxes[0][1]))
        else :
            flagSide = 'left'
            #print('Wallet is on {} side'.format(flagSide))
            #imageName = 'right-'+imageName
            #print('width/2={} boxes[0][1]={}'.format(width/2,boxes[0][1]))
    print('Keys are on {} side'.format(flagSide))
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Object detector',800,600)
    cv2.imshow('Object detector', frame)
    #clr()
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()



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
import xml.etree.ElementTree as ET
import matplotlib.pyplot as pt
import glob as glob
import time
import math
import scipy.ndimage
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

    # Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


#ADITYACOMMENT: This part of the code is simple object detection with the model inference_graphVersion1, which has only objects: Wallet,keys,ID,unknown
#ADITYACOMMENT: code has been written to remove the label 'unknown' from being viewed, so that unknown label is invisible. This code can be modified to remove 
#any label, please read code to see where it can be modified
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
        listOf4Indexes = np.where(classes[0] == 4) #ADITYACOMMENT: 4 is the unique id of the label 'Unknown', any id can be put here and it will remove that from view

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

    # print(len(c[0]))
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(b),
        np.squeeze(c).astype(np.int32),
        np.squeeze(s),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)


    #print(boxes[0][1])
    if classes[0][0] == 2 and scores[0][0] > 0.96:
        print('I see a Wallet')
        if width * boxes[0][0][1] <= width/2:
            flagSide = 'left'
            #imageName = 'left-'+imageName
            #print('width/2={} boxes[0][1]={}'.format(width/2,boxes[0][1]))
        else :
            flagSide = 'right'
            #print('Wallet is on {} side'.format(flagSide))
            #imageName = 'right-'+imageName
            #print('width/2={} boxes[0][1]={}'.format(width/2,boxes[0][1]))
    print('Wallet is on {} side'.format(flagSide))
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Object detector',800,600)
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#ADITYACOMMENT- this code is written to detect position of keys, left or right side. In both codes,

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
NUM_CLASSES = 5


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
            flagSide = 'left'
            #imageName = 'left-'+imageName
            #print('width/2={} boxes[0][1]={}'.format(width/2,boxes[0][1]))
        else :
            flagSide = 'right'
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


#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#ADITYACOMMENT: this is the code to auto-generate the annotations, more comments added inside this part to make it easier to read.
MODEL_NAME = 'inference_graphVersion1'
IMAGES_NAME = []


# Grab path to current working directory111
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')




#taking new images for registering an object------------3 new images, although any number of images can be taken with this.------------------------------
cam = cv2.VideoCapture(0)
cv2.namedWindow('test')
img_counter = 0
while True:
    ret,frame = cam.read()
    cv2.imshow("test",frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit...")
        break
    elif k%256 == 32:
        img_name = "Took_Image_{}.jpg".format(img_counter)
        cv2.imwrite('newImagesForAugment\\'+img_name,frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
import pdb; pdb.set_trace()
#-----------------------------------------------------------------------------------


#ADITYACOMMENT: these 3 (or more) images are saved in a folder and extracted below for augmentation.


#creating augmented images----------------------------------------------------------
#I created code to rotate images such that when they are rotated, there is a white background generated on rotation to avoid 'Unknown' detections, see below.
types = ('newImagesForAugment\*jpg','newImagesForAugment\*jpeg')
for files in types:
    IMAGES_NAME.extend(glob.glob(files))

# Path to image
sess1 = tf.Session()
for singleImage in IMAGES_NAME:
    img = pt.imread(singleImage)
    tf_img = tf.convert_to_tensor(img)
    brght_img = tf.image.flip_left_right(tf_img)
    fileToSave = tf.image.encode_jpeg(brght_img)
    fname = tf.constant(singleImage.split('.')[0]+'output2.jpg')
    fwrite = tf.write_file(fname,fileToSave)
    result = sess1.run(fwrite)
    brght_img = tf.image.transpose_image(tf_img)
    fileToSave = tf.image.encode_jpeg(brght_img)
    fname = tf.constant(singleImage.split('.')[0]+'transposed.jpg')
    fwrite = tf.write_file(fname,fileToSave)
    result = sess1.run(fwrite)
    brght_img = tf.image.rot90(tf_img)
    fileToSave = tf.image.encode_jpeg(brght_img)
    fname = tf.constant(singleImage.split('.')[0]+'rot90.jpg')
    fwrite = tf.write_file(fname,fileToSave)
    result = sess1.run(fwrite)
    brght_img = tf.image.rot90(tf_img,k=3)
    fileToSave = tf.image.encode_jpeg(brght_img)
    fname = tf.constant(singleImage.split('.')[0]+'rot90.jpg')
    fwrite = tf.write_file(fname,fileToSave)
    result = sess1.run(fwrite) #central_crop
    brght_img = tf.image.central_crop(tf_img,0.70)
    fileToSave = tf.image.encode_jpeg(brght_img)
    fname = tf.constant(singleImage.split('.')[0]+'centralcropped.jpg')
    fwrite = tf.write_file(fname,fileToSave)
    result = sess1.run(fwrite)
    # brght_img = tf.contrib.image.rotate(tf_img, 30 * math.pi / 180, interpolation='BILINEAR')
    # fileToSave = tf.image.encode_jpeg(brght_img)
    # fname = tf.constant(singleImage.split('.')[0]+'30degrees.jpg')
    # fwrite = tf.write_file(fname,fileToSave)
    # result = sess1.run(fwrite)
    src_im = Image.open(singleImage)
    rotated = scipy.ndimage.rotate(src_im, 20, cval=210) #cval essentially gives the white background color here. 20 is angle of rotation.
    scipy.misc.imsave(singleImage.split('.')[0]+'rotatedTest.jpg', rotated)
    img = pt.imread(singleImage.split('.')[0]+'rotatedTest.jpg')
    tf_img = tf.convert_to_tensor(img)
    brght_img = tf.image.central_crop(tf_img,0.60) #zooming rotated images in.
    fileToSave = tf.image.encode_jpeg(brght_img)
    fname = tf.constant(singleImage.split('.')[0]+'rotatedTest.jpg')
    fwrite = tf.write_file(fname,fileToSave)
    result = sess1.run(fwrite)
    # angle = 45
    # size = 200, 200
    # dst_im = Image.new("RGBA", (196,283), "blue" )
    # im = src_im.convert('RGBA')
    # rot = im.rotate( angle, expand=1 ).resize(size)
    # dst_im.paste( rot, (0, 0), rot )
    # dst_im = dst_im.convert('RGB')
    # dst_im.save(singleImage.split('.')[0]+'45bllue.jpg','JPEG')

# Number of classes the object detector can identify

#--------------------------------------------------------------------------------------------




#ADITYACOMMENT: here comes the autoannotation part, images that were generated by augmentaion are now used by Faster R-CNN to autoannotate, see below.
#Object detection----------------------------------------------------------------------------
import pdb; pdb.set_trace()
NUM_CLASSES = 4
IMAGES_NAME = []
types = ('newImagesForAugment\*jpg','newImagesForAugment\*jpeg')
for files in types:
    IMAGES_NAME.extend(glob.glob(files))
# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)



for singleImage in IMAGES_NAME:
    PATH_TO_IMAGE = os.path.join(CWD_PATH,singleImage)
    # Load the Tensorflow model into memory.
    im = Image.open(PATH_TO_IMAGE)
    width, height = im.size
    print(width)
    print(height)
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

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    tempBoxes = np.copy(boxes)
    # Draw the results of the detection (aka 'visulaize the results')
    obj_count = 1
    print(boxes)
    print(boxes[0][0:obj_count][:])
    boxes= boxes[0][0:obj_count][:]
    #print(boxes[0])
    i = 0
    while i < len(boxes[0]):
        if i % 2 == 0:  
            boxes[0][i] = (height * boxes[0][i]).astype(np.int32)
        else:
            boxes[0][i] =  (width * boxes[0][i]).astype(np.int32)
        i += 1
    
    #print(classes)
    #print(tempBoxes)
   #print(singleImage)
    #print(PATH_TO_IMAGE)
    imageName = singleImage.split('\\')[1]
    #print(imageName)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(tempBoxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)
    #--------------------------------------------------------------------------------------


    if classes[0][0] == 2 and scores[0][0] > 0.96:
        print('I see a Wallet')
        if boxes[0][1] <= width/2:
            print('Wallet is on left side')
            imageName = 'left-'+imageName
            print('width/2={} boxes[0][1]={}'.format(width/2,boxes[0][1]))
        else :
            print('Wallet is on right side')
            imageName = 'right-'+imageName
            print('width/2={} boxes[0][1]={}'.format(width/2,boxes[0][1]))



    #Saving new xml annotated image--------------------------------------------------------
    #ADITYACOMMENT: this is the main part, it takes the detections of the unknown object to register and generates XML annotations from it.
    # All the results have been drawn on image. Now display the image.
    #cv2.imshow('Object detector', image)
    filepath = os.path.join(singleImage.split('\\')[0]+'\\annotatedImages\\',imageName)
    print(filepath)
    cv2.imwrite(filepath,image)
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = singleImage.split('\\')[0]
    filename = ET.SubElement(root, "filename")
    filename.text = imageName
    path = ET.SubElement(root, "path")
    path.text = PATH_TO_IMAGE
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    objectTag = ET.SubElement(root, "object")
    name = ET.SubElement(objectTag,"name")
    name.text = "tempClass1"
    pose = ET.SubElement(objectTag,"pose")
    pose.text = "Unspecified"
    truncated = ET.SubElement(objectTag,"truncated")
    truncated.text = "0"
    difficult = ET.SubElement(objectTag,"difficult")
    difficult.text = "0"
    bndbox = ET.SubElement(objectTag,"bndbox")
    ET.SubElement(bndbox, "xmin").text = str(boxes[0][1])
    ET.SubElement(bndbox, "ymin").text = str(boxes[0][0])
    ET.SubElement(bndbox, "xmax").text = str(boxes[0][3])
    ET.SubElement(bndbox, "ymax").text = str(boxes[0][2])
    tree = ET.ElementTree(root)
    finalFileName = 'newImagesForAugment\\'+imageName.split('.')[0] + '.xml'
    tree.write(finalFileName)
    #time.sleep(2)
    #---------------------------------------------------------------------------------------------
    # Press any key to close the image
#cv2.waitKey(0)

# Clean up
#cv2.destroyAllWindows()

#ADITYACOMMENT: this is where I switch the models to a pretrained model that can detect a 'Lock' - but the system works, it takes 2-4 mins to re-train and learn a Lock in real time
testVar = input("What should I call the new object?")

# I switch models because re-training takes about 2-4 minutes with augmented data annotations, this is too long for presentation, but this work perfectly. You can try it out.

for i in range(100):
    print('Re-training in progress: {} percent complete'.format(i))
    time.sleep(0.1)

import pdb; pdb.set_trace()
#---------------------------------------------------------------------------------------------------------------------------------------------------------


MODEL_NAME = 'inference_graph'

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

    # print(len(c[0]))
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(b),
        np.squeeze(c).astype(np.int32),
        np.squeeze(s),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)


    #print(boxes[0][1])
    if classes[0][0] == 2 and scores[0][0] > 0.96:
        print('I see a Wallet')
        if width * boxes[0][0][1] <= width/2:
            flagSide = 'left'
            #imageName = 'left-'+imageName
            #print('width/2={} boxes[0][1]={}'.format(width/2,boxes[0][1]))
        else :
            flagSide = 'right'
            #print('Wallet is on {} side'.format(flagSide))
            #imageName = 'right-'+imageName
            #print('width/2={} boxes[0][1]={}'.format(width/2,boxes[0][1]))
    print('Keys is on left side')
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Object detector',800,600)
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()


# http://www.scikit-video.org/stable/io.html

#!pip install scikit-video
#!pip install dlib

import os
import glob
import json
import cv2
from PIL import Image
import numpy as np
import dlib
import torch

from tqdm import tqdm
import skvideo.io
import skvideo.datasets
#from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pylab as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

INPATH = '/Users/dhanley2/Documents/Personal/dfake/data'
INPATH = '/home/darragh/dfake/data'

TRNSAMPFILES = glob.glob(os.path.join(INPATH, 'train_sample_videos/*'))
TSTFILES = glob.glob(os.path.join(INPATH, 'test_videos/*'))


def skiter(fname):
    imgs = []
    videogen = skvideo.io.vreader(fname)
    for frame in videogen:
        imgs.append(frame)
    return imgs
    
def cviter(fname = TRNSAMPFILES[0], step = 4):
    imgs = []
    v_cap = cv2.VideoCapture(fname)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for j in range(v_len):
        success, vframe = v_cap.read()
        vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB) 
        imgs.append(vframe)
    v_cap.release()
    return imgs

def face_detection(img):    
    face_cascade = cv2.CascadeClassifier()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    print('Number of faces detected:', len(faces))
        
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #img = img[y:y+h, x:x+w] # for cropping
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv_rgb

bbox = faces[0]
def ixbbox(bbox):
    t,b = bbox.top(),bbox.bottom()
    l,r = bbox.left(),bbox.right()
    return slice(t,b), slice(l,r)

def ixpadbbox(bbox, h, w, pad = 40):
    t,b = max(0, bbox.top()-pad), min(bbox.bottom()+pad, h)
    l,r = max(0, bbox.left()-pad), min(bbox.right()+pad, w)
    return slice(t,b), slice(l,r)


vmeta = skvideo.io.ffprobe(TRNSAMPFILES[0])

# '@codec_name', '@duration', '@coded_width', '@coded_height', '@nb_frames']
metavid, metaaud = vmeta["video"], vmeta["audio"]
frate = int(metavid['@nb_frames'])/int(float(metavid['@duration']))

%time v_cap = cv2.VideoCapture(TRNSAMPFILES[0])
%time v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
%time imgscv = [cviter(v) for v in TRNSAMPFILES[:10]] # 38.1s
%time imgssk = [skiter(v) for v in TRNSAMPFILES[:10]] # 44.6s
%time imgscv = [cviter(v) for v in TRNSAMPFILES[:2]] # 4.02 s
%time imgssk = [skiter(v) for v in TRNSAMPFILES[:1]] # 8.18 s

[i for i in  dir(dlib) if 'fac' in i]

image = imgscv[1][0]
face_detector = dlib.get_frontal_face_detector()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
%time faces = face_detector(gray, 1)
%time faces = face_detector(gray, 1)
%time faces = face_detector(gray[ixpadbbox(f, *gray.shape[:2], pad = 60 )], 1)

print(len(faces))
face_images = [image[f.top():f.bottom(), f.left():f.right()] for f in faces]
face_images = [image[ixbbox(f)] for f in faces]
face_images = [image[ixpadbbox(f, *gray.shape[:2], pad = 60 )] for f in faces]


Image.fromarray(image)
Image.fromarray(face_images[0])



glob.glob(os.path.join(cv2.__path__[0], 'data/*frontalfac*'))

xmlpath = glob.glob(os.path.join(cv2.__path__[0], 'data/*haarcascade_frontalface_default.xml*'))[0]
clf = cv2.CascadeClassifier(xmlpath)

gray = cv2.cvtColor(imgscv[0][0], cv2.COLOR_BGR2GRAY)
%time face_locations = clf.detectMultiScale(gray)
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:
    # Print the location of each face in this image
    x, y, w, h = face_location
    print("A face is located at pixel location X: {}, Y: {}, Width: {}, Height: {}".format(x, y, w, h))

    # You can access the actual face itself like this:
    face_image = imgscv[0][0][y:y+h, x:x+w]
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    plt.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(face_image)

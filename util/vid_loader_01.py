# http://www.scikit-video.org/stable/io.html
# https://github.com/abewley/sort

#!pip install scikit-video
#!pip install dlib
#!pip install filterpy

import os
import sys
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

INPATH = '/Users/dhanley2/Documents/Personal/dfake'
INPATH = '/home/darragh/dfake'
sys.path.append(INPATH)
from utils.sort import *

TRNSAMPFILES = glob.glob(os.path.join(INPATH, 'data/train_sample_videos/*'))
TSTFILES = glob.glob(os.path.join(INPATH, 'data/test_videos/*'))
FACEWEIGHTS = os.path.join(INPATH, 'weights/mmod_human_face_detector.dat')

def vid2imgls(fname, FPS=8):
    imgs = []
    v_cap = cv2.VideoCapture(TRNSAMPFILES[0])
    vnframes, vh, vw, vfps = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
            int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(round(v_cap.get(cv2.CAP_PROP_FPS)))
    vcap = cv2.VideoCapture(fname)
    for t in range(vnframes):
        ret = vcap.grab()
        if t % int(round(vfps/FPS)) == 0:
            ret, frame = vcap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)
    vcap.release()
    return imgs

face_detector = dlib.get_frontal_face_detector()
face_detector = dlib.cnn_face_detection_model_v1(FACEWEIGHTS)

def face_bbox(image, fn = face_detector, RESIZE = (400,260) ):
    ih, iw = image.shape[:2]
    RESIZE = tuple((min(RESIZE[0], iw), min(RESIZE[1], ih)))
    facesls = []
    while len(facesls)==0 and RESIZE[0]<iw:
        HEIGHT_DOWNSIZE, WIDTH_DOWNSIZE = ih/RESIZE[1], iw/RESIZE[0] 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        gray = cv2.resize(gray, RESIZE, interpolation=cv2.INTER_CUBIC)
        faces = fn(gray, 1)
        facesls += [{'t' : int(round(f.rect.top()*HEIGHT_DOWNSIZE)), 
                  'b' : int(round(f.rect.bottom()*HEIGHT_DOWNSIZE)), 
                  'l': int(round(f.rect.left()*WIDTH_DOWNSIZE)), 
                  'r': int(round(f.rect.right()*WIDTH_DOWNSIZE)),
                  'conf' : f.confidence} for f in faces]
        RESIZE = tuple(int(t*1.5) for t in RESIZE)
    return facesls


FPS = 8
%time imgls1 = vid2imgls(os.path.join(INPATH, 'data/train_sample_videos/aapnvogymq.mp4'), FPS)
%time faces = [face_bbox(i) for t, i in tqdm(enumerate(imgls1)) if t % FPS ==0 ]
faces

mot_tracker = Sort()
trackmat = []
def mbbox(bbdict):
    return [bbdict[i] for i in ['l','t', 'r', 'b', 'conf']]
frameobjs = set()
for t, frame in enumerate(faces):
    dets = np.array([mbbox(b) for b in frame])
    trackers = mot_tracker.update(dets)
    for tt, d in enumerate(trackers):
        faces[t][tt]['obj'] = int(d[-1])
        frameobjs.add(int(d[-1]))
        
[[f1 for f1 in f if f1['obj']==21] for f in faces ]
        
trackmat = np.around(np.array(trackmat), 0).astype(int)
trackmat[trackmat[:,-1].argsort()]

image = imgls1[10]
Image.fromarray(image)
faces = face_bbox(image)
face_images = [image[f['t']:f['b'], f['l']:f['r']] for f in faces]
Image.fromarray(face_images[1])
Image.fromarray(face_images[0])

Image.fromarray(gray)
Image.fromarray(image)
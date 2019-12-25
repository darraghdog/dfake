# http://www.scikit-video.org/stable/io.html
# https://github.com/abewley/sort
# https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv/50757596
# https://github.com/danmohaha/DSP-FWA
# http://krasserm.github.io/2018/02/07/deep-face-recognition/

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
import pandas as pd
import dlib
import torch
from itertools import product
from time import time
import datetime
from align import AlignDlib
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
from utils.logs import get_logger
from utils.utils import dumpobj, loadobj
# Print info about environments
logger = get_logger('VIDEO SETUP', 'INFO') 

TRNSAMPFILES = glob.glob(os.path.join(INPATH, 'data/train_sample_videos/*'))
TSTFILES = glob.glob(os.path.join(INPATH, 'data/test_videos/*'))
FACEWEIGHTS = os.path.join(INPATH, 'weights/mmod_human_face_detector.dat')

face_detector = dlib.get_frontal_face_detector()
face_detector = dlib.cnn_face_detection_model_v1(FACEWEIGHTS)

def vid2imgls(fname, FPS=8):
    imgs = []
    v_cap = cv2.VideoCapture(fname)
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

def face_bbox(image, fn = face_detector, RESIZE_MAXDIM = 500 ):
    ih, iw = image.shape[:2]
    RESIZE_RATIO = RESIZE_MAXDIM / max(ih, iw)
    RESIZE = tuple((int(RESIZE_RATIO * iw), int(RESIZE_RATIO * ih)))
    facesls = []
    while len(facesls)==0 and RESIZE[0]<iw:
        HEIGHT_DOWNSIZE, WIDTH_DOWNSIZE = ih/RESIZE[1], iw/RESIZE[0] 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        gray = cv2.resize(gray, RESIZE, interpolation=cv2.INTER_CUBIC)
        faces = fn(gray, 1)
        facesls += [[f.rect.left()*WIDTH_DOWNSIZE, 
                  f.rect.top()*HEIGHT_DOWNSIZE, 
                  f.rect.right()*WIDTH_DOWNSIZE,
                  f.rect.bottom()*HEIGHT_DOWNSIZE,
                  f.confidence] for f in faces]
        RESIZE = tuple(int(t*1.5) for t in RESIZE)
    return facesls

# Make tracker for box areas
def sortbbox(faces, thresh = 3, max_age = 1):
    mot_tracker = Sort(max_age = 1)
    trackmat = []
    for t, frame in enumerate(faces):
        dets = np.array( frame)
        trackers = mot_tracker.update(dets)
        trackers = np.hstack((trackers, np.ones((trackers.shape[0], 1))*t*ANCHORFRAMES))
        trackmat += trackers.tolist()
    cols =  ['x1', 'y1', 'x2', 'y2', 'obj', 'frame']
    trackmat = pd.DataFrame(trackmat, columns = cols).astype(np.int)#[:,0,:]
    trackmat = trackmat[ trackmat.groupby('obj')['obj'].transform('count') >= thresh]
    return trackmat


# Get anchor frames for boxes
VNAMES = os.listdir(os.path.join(INPATH, 'data/train_sample_videos'))[:3]
VNAME = VNAMES[0]
logls = []
FPS = 8
ANCHORFRAMES = FPS//2
logger.info('Process {}'.format(VNAME))
imgls1 = vid2imgls(os.path.join(INPATH, 'data/train_sample_videos/{}'.format(VNAME)), FPS)
faces = [face_bbox(i) for t, i in tqdm(enumerate(imgls1)) if t % ANCHORFRAMES ==0 ]

l=list(map(int, faces[0][0]))
Image.fromarray(imgls1[0])
fimg = imgls1[0][l[1]:l[3],l[0]:l[2]]
Image.fromarray(fimg)

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')    
align_image(fimg)


for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


trackmat = sortbbox(faces, thresh = 2)
logger.info('Tracker length (1st try) {}'.format(len(trackmat)))

if len(trackmat)<len(faces)//2 and len(faces)>8:
    if int((datetime.datetime.now()-START).total_seconds())<360:
        logger.info('Tracker length - 2nd try needed')
        ANCHORFRAMES //= 2
        faces = [face_bbox(i) for t, i in enumerate(imgls1) if t % ANCHORFRAMES ==0 ]
        trackmat = sortbbox(faces, thresh = 2)
logger.info('Face ct : {}; Track ct : {}'.format(sum(len(l) for l in faces), len(trackmat)))
# Image.fromarray(imgls1[0])

trackfull = pd.DataFrame(list(product(trackmat.obj.unique(), range(len(imgls1) ))), \
             columns=['obj', 'frame'])
trackfull = trackfull.merge(trackmat, how = 'left')

trackfull = pd.concat([trackfull.query('obj==@o')\
                       .interpolate(method='piecewise_polynomial').dropna() \
     for o in trackfull['obj'].unique()], 0) \
        .astype(np.int).sort_values(['frame', 'obj'], 0) \
        .reset_index(drop=True)
trackfull.obj = trackfull.obj.astype('category').cat.codes

# Visualise it all
H, W, _ = imgls1[0].shape
NOBJ = 1+trackfull.obj.max()
MAXOBJ = trackfull.obj.value_counts().max()
# Pad the boundary box
trackfull['w'] = trackfull.x2 - trackfull.x1
trackfull['h'] = trackfull.y2 - trackfull.y1
trackfull['maxw'] = trackfull.groupby(['obj'])['w'].transform(max)
trackfull['maxh'] = trackfull.groupby(['obj'])['h'].transform(max)

imgdict = dict((o, []) for o in range(NOBJ)) 
for (t, row) in trackfull.iterrows():
    obj = row.obj
    frame = imgls1[row.frame]
    imgdict[obj].append(frame[row.y1:row.y1+row.maxh, row.x1:row.x1+row.maxw])

imgdict = dict((k, np.vstack(v)) for k, v in imgdict.items())
imgdim0 = max([i.shape[0] for i in imgdict.values()])
imgdict = dict((k, np.vstack((v, \
                np.zeros((imgdim0 - v.shape[0], v.shape[1], 3),dtype=np.uint8 ) )) ) \
               for k, v in imgdict.items())
imgall = np.hstack(list(imgdict.values()))
FOUT = os.path.join(INPATH, 'check/train_sample_check/{}'.format(VNAME.replace('mp4', 'jpg')))
logger.info('Write image out')
Image.fromarray(imgall)
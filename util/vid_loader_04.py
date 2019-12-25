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

def vid1imgls(fname, FPS=8):
    imgs = []
    vcap = cv2.VideoCapture(fname)
    vnframes, vh, vw, vfps = \
                int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)), \
                int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
                int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                int(round(vcap.get(cv2.CAP_PROP_FPS)))
    for t in range(vnframes):
        ret = vcap.grab()
        if t % int(round(vfps/FPS)) == 0:
            ret, frame = vcap.retrieve()
            imgs.append(frame)
    vcap.release()
    return imgs

# Get anchor frames for boxes
VNAMES = glob.glob(os.path.join(INPATH, 'data/train_sample_videos/*'))
FPS = 8
%time imgls1 = [vid1imgls(f, FPS) for f in tqdm(VNAMES[:10])]

from multiprocessing import Pool
from multiprocessing import cpu_count
def poolload():
    pool = Pool(processes=4)
    imgls2 = pool.map(vid1imgls, tqdm(VNAMES[:10]))
    pool.close()
    pool.join()
    return imgls2
%time imgls2 = poolload()

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
VNAMES = os.listdir(os.path.join(INPATH, 'data/train_sample_videos'))
logls = []
for VNAME in VNAMES[:50]:
    START = datetime.datetime.now()
    try:
        FPS = 8
        ANCHORFRAMES = FPS//2
        logger.info('Process {}'.format(VNAME))
        %time imgls1 = vid2imgls(os.path.join(INPATH, 'data/train_sample_videos/{}'.format(VNAME)), FPS)
        faces = [face_bbox(i) for t, i in enumerate(imgls1) if t % ANCHORFRAMES ==0 ]
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
        
        trackvid = pd.DataFrame(list(product(trackmat.obj.unique(), range(len(imgls1) ))), \
                     columns=['obj', 'frame'])
        trackvid = trackvid.merge(trackmat, how = 'left')
        
        trackvid = pd.concat([trackvid.query('obj==@o')\
                               .interpolate(method='piecewise_polynomial').dropna() \
             for o in trackvid['obj'].unique()], 0) \
                .astype(np.int).sort_values(['frame', 'obj'], 0) \
                .reset_index(drop=True)
        trackvid.obj = trackvid.obj.astype('category').cat.codes
        trackvid['video']=VNAME
        END = datetime.datetime.now()
        STATUS = 'sucess'
    except:
        END = datetime.datetime.now()
        STATUS = 'fail'
    DURATION = int((END-START).total_seconds())
    logls.append([VNAME, DURATION, STATUS])
    
    
    pd.DataFrame(logls, columns = ['video', 'duration', 'status']) \
        .to_csv(os.path.join(INPATH, 'check/train_sample_check/log.txt'), index = False)
        
        # Visualise it all
        H, W, _ = imgls1[0].shape
        NOBJ = 1+trackvid.obj.max()
        MAXOBJ = trackvid.obj.value_counts().max()
        # Pad the boundary box
        trackvid['w'] = trackvid.x2 - trackvid.x1
        trackvid['h'] = trackvid.y2 - trackvid.y1
        trackvid['maxw'] = trackvid.groupby(['obj'])['w'].transform(max)
        trackvid['maxh'] = trackvid.groupby(['obj'])['h'].transform(max)
        
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
        Image.fromarray(imgall).save(FOUT)
        
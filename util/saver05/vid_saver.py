# http://www.scikit-video.org/stable/io.html
# https://github.com/abewley/sort
# https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv/50757596
# https://github.com/danmohaha/DSP-FWA
# http://krasserm.github.io/2018/02/07/deep-face-recognition/
# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=videocapture

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
import collections
from tqdm import tqdm
import skvideo.io
import skvideo.datasets
import random
import optparse
import itertools
#from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings("ignore")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

# Print info about environments
parser = optparse.OptionParser()
parser.add_option('-a', '--seed', action="store", dest="seed", help="model seed", default="1234")
parser.add_option('-b', '--fold', action="store", dest="fold", help="Fold for split", default="0")
parser.add_option('-c', '--rootpath', action="store", dest="rootpath", help="root directory", default="")
parser.add_option('-d', '--vidpath', action="store", dest="vidpath", help="root directory", default="data/mount/video/train")
parser.add_option('-e', '--imgpath', action="store", dest="imgpath", help="root directory", default="data/mount/npimg/train")
parser.add_option('-f', '--wtspath', action="store", dest="wtspath", help="root directory", default="weights")
parser.add_option('-g', '--fps', action="store", dest="fps", help="Frames per second", default="8")
parser.add_option('-i', '--size', action="store", dest="size", help="image size", default="224")


options, args = parser.parse_args()
INPATH = options.rootpath

sys.path.append(os.path.join(INPATH, 'utils' ))
from logs import get_logger
from utils import dumpobj, loadobj
from sort import *

# Print info about environments
logger = get_logger('Video to image :', 'INFO') 
logger.info('Cuda set up : time {}'.format(datetime.datetime.now().time()))

device=torch.device('cuda')
logger.info('Device : {}'.format(torch.cuda.get_device_name(0)))
logger.info('Cuda available : {}'.format(torch.cuda.is_available()))
n_gpu = torch.cuda.device_count()
logger.info('Cuda n_gpus : {}'.format(n_gpu ))


logger.info('Load params : time {}'.format(datetime.datetime.now().time()))
for (k,v) in options.__dict__.items():
    logger.info('{}{}'.format(k.ljust(20), v))

SEED = int(options.seed)
SIZE = int(options.size)
FOLD = int(options.fold)
TRNFILES = glob.glob(os.path.join(INPATH, options.vidpath, '*'))
WTSFILES = os.path.join(INPATH, options.wtspath)
FACEWEIGHTS = os.path.join(INPATH, WTSFILES, 'mmod_human_face_detector.dat')
face_detector = dlib.cnn_face_detection_model_v1(FACEWEIGHTS)
OUTDIR = os.path.join(INPATH, options.imgpath)
FPS = int(options.fps)

logger.info(TRNFILES[:5])


def vid2imgls(fname, FPS=8):
    imgs = []
    v_cap = cv2.VideoCapture(fname)
    vnframes, vh, vw, vfps = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
            int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(round(v_cap.get(cv2.CAP_PROP_FPS)))
    vcap = cv2.VideoCapture(fname)
    for t in range(vnframes//2):
        ret = vcap.grab()
        if t % int(round(vfps/FPS)) == 0:
            ret, frame = vcap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)
    vcap.release()
    return imgs

def face_bbox(image, fn = face_detector, RESIZE_MAXDIM = 500 ):
    warnings.filterwarnings("ignore")
    try:
        ih, iw = image.shape[:2]
        RESIZE_RATIO = RESIZE_MAXDIM / max(ih, iw)
        RESIZE = tuple((int(RESIZE_RATIO * iw), int(RESIZE_RATIO * ih)))
        facesls = []
        HEIGHT_DOWNSIZE, WIDTH_DOWNSIZE = ih/RESIZE[1], iw/RESIZE[0] 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        gray = cv2.resize(gray, RESIZE, interpolation=cv2.INTER_CUBIC)
        faces = fn(gray, 1)
        facesls += [[f.rect.left()*WIDTH_DOWNSIZE, 
                  f.rect.top()*HEIGHT_DOWNSIZE, 
                  f.rect.right()*WIDTH_DOWNSIZE,
                  f.rect.bottom()*HEIGHT_DOWNSIZE,
                  f.confidence] for f in faces]
        return facesls
    except:
        logger.info('Bad image returned')
        return []
    

# Make tracker for box areas
def sortbbox(faces, anchorframes, thresh = 3, max_age = 1):
    mot_tracker = Sort(max_age = 1)
    trackmat = []
    for t, frame in enumerate(faces):
        dets = np.array( frame)
        trackers = mot_tracker.update(dets)
        trackers = np.hstack((trackers, np.ones((trackers.shape[0], 1))*t*anchorframes))
        trackmat += trackers.tolist()
    cols =  ['x1', 'y1', 'x2', 'y2', 'obj', 'frame']
    trackmat = pd.DataFrame(trackmat, columns = cols).astype(np.int)#[:,0,:]
    trackmat = trackmat[ trackmat.groupby('obj')['obj'].transform('count') >= thresh]
    return trackmat

def gettrack(imgls, anchorframes, maxdim):
    faces = [face_bbox(i, RESIZE_MAXDIM = maxdim) for t, i in enumerate(imgls) if t % anchorframes ==0 ]
    trackmat = sortbbox(faces, anchorframes, thresh = 2)  
    logger.info('Detector dimension {} Anchor frames {} Tracker length {} Faces count {}'.format(maxdim, anchorframes, len(trackmat), len(list(itertools.chain(*faces)))))
    return trackmat, faces


# Get anchor frames for boxes
logls = []
trackls = []
counter = 0 
for tt, VNAME in enumerate(TRNFILES[:1000]):
    START = datetime.datetime.now()
    try:
        logger.info('Process image {} : {}'.format(tt, VNAME.split('/')[-1]))
        imgls = vid2imgls(VNAME, FPS)
        H, W, _ = imgls[0].shape
        probebbox, MAXDIM = [], 500.0
        probels = random.sample(imgls, k = 2)
        while (len(probebbox)==0) and MAXDIM < max(H,W):
            probebbox = list(itertools.chain(*[face_bbox(p, RESIZE_MAXDIM = MAXDIM) for p in probels]))
            if len(probebbox)==0 : MAXDIM *= 1.3
        if len(probebbox)==0:
            raise Exception('Cannot find faces')
        trackmat, faces = gettrack(imgls, FPS//2, MAXDIM)
        if len(trackmat)<4:
            trackmat, faces = gettrack(imgls, FPS//4, min(max(H,W),1000)) 
        trackvid = pd.DataFrame(list(product(trackmat.obj.unique(), range(len(imgls) ))), \
                     columns=['obj', 'frame'])
        trackvid = trackvid.merge(trackmat, how = 'left')
        
        trackvid = pd.concat([trackvid.query('obj==@o')\
                               .interpolate(method='piecewise_polynomial').dropna() \
             for o in trackvid['obj'].unique()], 0) \
                .astype(np.int).sort_values(['frame', 'obj'], 0) \
                .reset_index(drop=True)
        trackvid.obj = trackvid.obj.astype('category').cat.codes
        trackvid['video']=VNAME
        trackvid['maxdim'] = pd.concat([trackvid.x2-trackvid.x1, trackvid.y2-trackvid.y1], axis=1).max(axis=1)
        imgdict = collections.OrderedDict((o, []) for o in range(1+trackvid.obj.max()))         
        for (t, row) in trackvid.iterrows():
            obj = row.obj
            frame = imgls[row.frame]
            face = frame[row.y1:row.y1+row.maxdim, row.x1:row.x1+row.maxdim]
            face = cv2.resize(face, (SIZE,SIZE), interpolation=cv2.INTER_CUBIC)
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) 
            imgdict[obj].append(face)
        trackfaces = np.array(sum(list(imgdict.values()), []))
        trackvid = trackvid.sort_values(['obj', 'frame'], 0).reset_index(drop=True)
        N_OBJ, N_FACES = len(trackvid.obj.unique()), len(trackvid)
        trackls.append(trackvid)
        np.savez_compressed(os.path.join(OUTDIR, VNAME.split('/')[-1].replace('mp4', 'npz')), trackfaces)
        # Image.fromarray(trackfaces[2])
        END = datetime.datetime.now()
        STATUS = 'sucess'
    except:
        END = datetime.datetime.now()
        N_OBJ, N_FACES = 0, 0
        STATUS = 'fail'
    DURATION = int((END-START).total_seconds())
    logls.append([VNAME.split('/')[-1], N_OBJ, N_FACES, DURATION, STATUS])
    
    
logdf = pd.DataFrame(logls, columns = ['video', 'objectct', 'framect', 'duration', 'status'])
trackdf = pd.concat(trackls, 0)
trackdf['video'] = trackdf['video'].apply(lambda x: x.split('/')[-1])
logdf.to_csv(os.path.join(OUTDIR, 'log.txt'), index = False)
trackdf.to_csv(os.path.join(OUTDIR, 'tracker.txt'), index = False)


        

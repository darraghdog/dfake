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
import torch
from itertools import product, chain
from time import time
import datetime
import collections
from tqdm import tqdm
import skvideo.io
import skvideo.datasets
import random
import optparse
import itertools
import matplotlib.pylab as plt
import warnings
from torch.utils.data import Dataset, DataLoader
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
parser.add_option('-j', '--metafile', action="store", dest="metafile", help="Meta file", default="trainmeta.csv.gz")
parser.add_option('-k', '--startsize', action="store", dest="startsize", help="Starting video resize dim", default="640")
parser.add_option('-l', '--batchsize', action="store", dest="batchsize", help="Batch size", default="64")
parser.add_option('-m', '--loadseconds', action="store", dest="loadseconds", help="Seconds to load from video", default="12")


options, args = parser.parse_args()
INPATH = options.rootpath

sys.path.append(os.path.join(INPATH))
from utils.sort import *
from utils.logs import get_logger
from utils.utils import dumpobj, loadobj, chunks
from utils.utils import cfg_re50, cfg_mnet, decode_landm, decode, PriorBox
from utils.utils import py_cpu_nms, load_fd_model, remove_prefix
from utils.retinaface import RetinaFace
warnings.filterwarnings("ignore")

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
    
'''
INPATH='/Users/dhanley2/Documents/Personal/dfake'
SIZE=224
FOLD=0
MAXLOADSECONDS=3
METAFILE = os.path.join(INPATH, 'data', options.metafile)
'''

SEED = int(options.seed)
SIZE = int(options.size)
FOLD = int(options.fold)
MAXLOADSECONDS = int(options.loadseconds )
METAFILE = os.path.join(INPATH, 'data', options.metafile)
WTSFILES = os.path.join(INPATH, options.wtspath)
# FACEWEIGHTS = os.path.join(INPATH, WTSFILES, 'mmod_human_face_detector.dat')
# face_detector = dlib.cnn_face_detection_model_v1(FACEWEIGHTS)
OUTDIR = os.path.join(INPATH, options.imgpath)
FPS = int(options.fps)
STARTSIZE = int(options.startsize)
BATCHSIZE = int(options.batchsize)


# METAFILE='/Users/dhanley2/Documents/Personal/dfake/data/trainmeta.csv.gz'


def imresizels(imgls, im_height, im_width, MAXDIM = 1000):
    if MAXDIM > max(im_height, im_width):
        return imgls, im_height, im_width
    RATIO = MAXDIM/ max(im_height, im_width)
    w, h = (int(im_width*RATIO), int(im_height*RATIO))
    outls = [cv2.resize(i, dsize=(w, h), interpolation=cv2.INTER_LINEAR) for i in imgls]
    return outls, h, w


def bboxes(loc, conf, scale, prior_data, cfg):
    boxes = decode(loc.data, prior_data, cfg['variance'])
    boxes = boxes * scale 
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]
    
    # ignore low scores
    inds = np.where(scores > cfg['confidence'])[0]
    boxes = boxes[inds]
    scores = scores[inds]
    
    # keep top-K before NMS
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, cfg['nms_threshold'])
    dets = dets[keep, :]
    
    # convert to boxes
    dets = [d[:4].astype(np.int32).clip(0, 999999).tolist() + [d[4]] for d in dets]
    return dets

def vid2imgls(fname, FPS=8, MAXLOADSECONDS = 20):
    imgs = []
    v_cap = cv2.VideoCapture(fname)
    vnframes, vh, vw, vfps = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
            int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(round(v_cap.get(cv2.CAP_PROP_FPS)))
    vcap = cv2.VideoCapture(fname)
    maxframes = vfps*MAXLOADSECONDS
    for t in range(vnframes):
        if t>maxframes: continue
        ret = vcap.grab()
        if t % int(round(vfps/FPS)) == 0:
            ret, frame = vcap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)
    vcap.release()
    return imgs, vh, vw

def face_bbox(imgls, im_height, im_width):    
    batch = np.float32(np.array([i for i in imgls]))
    # logger.info(batch.shape)
    batch -= (104, 117, 123)
    batch = torch.tensor(batch)
    batch = batch.permute(0, 3, 1, 2)
    scale = torch.Tensor([im_width, im_height]*2)
    batch = batch.to(device)
    scale = scale.to(device)
    loc, conf, landms = net(batch)
    #logger.info(loc[0])
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    dets = [bboxes(loc[i], conf[i], scale, prior_data, cfg) for i in range(batch.shape[0])]
    #logger.info(dets)
    return dets

# Make tracker for box areas
def sortbbox(faces, anchorframes, thresh = 3, max_age = 1):
    mot_tracker = Sort(max_age = 1)
    trackmat = []
    #logger.info(faces)
    for t, frame in enumerate(faces):
        dets = np.array( frame)
        trackers = mot_tracker.update(dets)
        trackers = np.hstack((trackers, np.ones((trackers.shape[0], 1))*t*anchorframes))
        trackmat += trackers.tolist()
    cols =  ['x1', 'y1', 'x2', 'y2', 'obj', 'frame']
    trackmat = pd.DataFrame(trackmat, columns = cols).astype(np.int)#[:,0,:]
    trackmat = trackmat[ trackmat.groupby('obj')['obj'].transform('count') >= thresh]
    return trackmat

def gettrack(imgls, anchorframes, im_height, im_width, bsize = BATCHSIZE):
    imgl = [i for t, i in enumerate(imgls) if t % anchorframes ==0 ]
    faces = [face_bbox(l, im_height, im_width) for l in chunks(imgl, bsize)]
    faces = list(chain(*faces))
    #logger.info(faces)
    trackmat = sortbbox(faces, anchorframes, thresh = 2)  
    logger.info('Video dimension {} {} Frames {} Anchor frames {} Tracker length {} Faces count {}'.format(\
                    im_height, im_width, len(imgls), anchorframes, len(trackmat), len(list(itertools.chain(*faces)))))
    return trackmat, faces


class DFakeVideoLoad(Dataset):
    def __init__(self, metafile, vidpath, fold, mnpath, modpath, config, \
                 maxloadsec, fps, batchsize, outdir, framesize = SIZE,\
                 initresize = 1024, confthresh=0.95, nmsthresh=0.4, saveimg = False):

        logger.info('Set up processing params')
        self.fold = fold
        self.initresize = initresize
        self.maxloadseconds = maxloadsec
        self.fps = fps
        self.batchsize = batchsize
        self.saveimg = saveimg
        self.outdir = outdir
        self.size = framesize
        
        logger.info('Set up model')
        self.mnpath = mnpath
        self.modpath = modpath
        self.cfg = config
        self.cfg['confidence'] = confthresh
        self.cfg['nms_threshold'] = nmsthresh
        self.net = RetinaFace(self.cfg, phase = 'test', mnpath = self.mnpath)
        self.net = load_fd_model(self.net, self.modpath, True)
        self.net = self.net.to(device)
        self.net.eval()
        
        logger.info('Set up files to load')
        df = pd.read_csv(METAFILE)
        df['video_path'] = vidpath+'/'+df['folder']+'/'+df['video']
        logger.info('Full video file shape {} {}'.format(*df.shape))
        self.metadf = df.reset_index(drop=True)
        self.vidfiles = self.metadf.video_path.tolist()[:100]
        
        logger.info('Set up logging params')
        self.logls = []
        self.trackls = []
        
    def __len__(self):
        return len(self.vidfiles)
    
    def __getitem__(self, idx):

        vname = self.vidfiles[idx]
        logger.info('Process image {} : {}'.format(idx, vname.split('/')[-1]))
        try:
            imgls, H, W = vid2imgls(vname, self.fps, self.maxloadseconds)
            # For most images we do not need the full size to find the boxes
            thumbls, h, w = imresizels(imgls, H, W, MAXDIM = self.initresize)
            trackmat, faces = gettrack(thumbls, self.fps//4, h, w, self.batchsize*2)
            # If downsizing does not work, try with the original image 
            if len(trackmat)<5 and max(H,W)<2000:
                trackmat, faces = gettrack(imgls, self.fps//4, H, W, self.batchsize)
                if len(trackmat)<5 and max(H,W)<2000:
                    trackmat, faces = gettrack(imgls, self.fps//8, H, W, self.batchsize)  
            else:
                trackmat[['x1', 'x2']] = (trackmat[['x1', 'x2']]*(W/w)).astype(np.int32)
                trackmat[['y1', 'y2']] = (trackmat[['y1', 'y2']]*(H/h)).astype(np.int32)
            trackmat[['x1', 'x2']] = trackmat[['x1', 'x2']].clip(0, W)
            trackmat[['y1', 'y2']] = trackmat[['y1', 'y2']].clip(0, H)
            trackvid = pd.DataFrame(list(product(trackmat.obj.unique(), range(len(imgls) ))), \
                         columns=['obj', 'frame'])
            trackvid = trackvid.merge(trackmat, how = 'left')
            trackvid = pd.concat([trackvid.query('obj==@o')\
                                   .interpolate(method='piecewise_polynomial').dropna() \
                 for o in trackvid['obj'].unique()], 0) \
                    .astype(np.int).sort_values(['frame', 'obj'], 0) \
                    .reset_index(drop=True)
            trackvid.obj = trackvid.obj.astype('category').cat.codes
            trackvid['video']=vname
            trackvid['maxdim'] = pd.concat([trackvid.x2-trackvid.x1, 
                                            trackvid.y2-trackvid.y1], axis=1).max(axis=1)
            imgdict = collections.OrderedDict((o, []) for o in range(1+trackvid.obj.max()))         
            for (t, row) in trackvid.iterrows():
                obj = row.obj
                frame = imgls[row.frame]
                face = frame[row.y1:row.y1+row.maxdim, row.x1:row.x1+row.maxdim]
                face = cv2.resize(face, (self.size,self.size), interpolation=cv2.INTER_CUBIC)
                imgdict[obj].append(face)
            trackfaces = np.array(sum(list(imgdict.values()), []))
            trackvid = trackvid.sort_values(['obj', 'frame'], 0).reset_index(drop=True)
            N_OBJ, N_FACES = len(trackvid.obj.unique()), len(trackvid)
            self.trackls.append(trackvid)
            if self.saveimg:
                np.savez_compressed(os.path.join(OUTDIR, VNAME.split('/')[-1].replace('mp4', 'npz')), trackfaces)
            self.logls.append([vname.split('/')[-1], N_OBJ, N_FACES])
            return {'idx': idx, 'frames' : trackfaces}
        except Exception:
            logger.exception("Fatal error in main loop")

def collatefn(batch):
    # Remove error reads
    batch = [b for b in batch if b is not None]
    seqlen = torch.tensor([l['frames'].shape[0] for l in batch])
    ids = torch.tensor([l['idx'] for l in batch])

    maxlen = seqlen.max()    
    # get shapes
    d0,d1,d2,d3 = batch[0]['frames'].shape
        
    # Pad with zero frames
    x_batch = [l['frames'] if l['frames'].shape[0] == maxlen else \
         torch.cat((l['frames'], torch.zeros((maxlen-sl,d1,d2,d3))), 0) 
         for l,sl in zip(batch, seqlen)]
    x_batch = torch.cat([x.unsqueeze(0) for x in x_batch])
    
    return {'frames': x_batch, 'ids': ids, 'seqlen': seqlen, 'labels': y_batch}


INPATH

alldataset = DFakeVideoLoad(
                        mnpath = os.path.join(WTSFILES, 'retinaface/mobilenetV1X0.25_pretrain.tar'),
                        config = cfg_mnet,
                        confthresh = 0.95,
                        nmsthresh = 0.4,
                        modpath = os.path.join(WTSFILES, 'retinaface/mobilenet0.25_Final.pth'),
                        metafile = METAFILE,
                        vidpath = os.path.join(INPATH, options.vidpath),
                        maxloadsec = MAXLOADSECONDS,
                        fps = FPS,
                        batchsize = BATCHSIZE,
                        saveimg = True,
                        outdir = OUTDIR)
allloader = DataLoader(alldataset, 
                       batch_size=4, 
                       shuffle=False, 
                       num_workers=8, 
                       collate_fn=collatefn)  
    
for step, batch in enumerate(allloader):
    x = batch['frames']#.to(device, dtype=torch.float)
    y = batch['labels']#.to(device, dtype=torch.float)

logdf = pd.DataFrame(alldataset.logls, columns = ['video', 'objectct', 'framect', 'duration', 'status'])
trackdf = pd.concat(alldataset.trackls, 0)
trackdf['video'] = trackdf['video'].apply(lambda x: x.split('/')[-1])
logdf.to_csv(os.path.join(alldataset.outdir, 'log_fold{}.txt'.format(alldataset.fold)), index = False)
trackdf.to_csv(os.path.join(alldataset.outdir, 'tracker_fold{}.txt'.format(alldataset.fold)), index = False)


        

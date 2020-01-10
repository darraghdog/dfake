import os
import sys
import glob
import json
import cv2
from PIL import Image
import numpy as np
import pandas as pd
#import dlib
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

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip, RandomBrightnessContrast, 
                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                           GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                           Blur, ToGray, ToSepia, MultiplicativeNoise, JpegCompression, Lambda, NoOp, CenterCrop, Resize
                           )
import albumentations as A
from tqdm import tqdm
from apex import amp


from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier


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
parser.add_option('-k', '--batchsize', action="store", dest="batchsize", help="Batch size", default="8")
parser.add_option('-l', '--epochs', action="store", dest="epochs", help="epochs", default="10")
parser.add_option('-m', '--lr', action="store", dest="lr", help="learning rate", default="0.0001")
parser.add_option('-n', '--decay', action="store", dest="decay", help="Weight Decay", default="0.0")
parser.add_option('-o', '--lrgamma', action="store", dest="lrgamma", help="Scheduler Learning Rate Gamma", default="1.0")
parser.add_option('-p', '--start', action="store", dest="start", help="Start epochs", default="0")
parser.add_option('-q', '--infer', action="store", dest="infer", help="root directory", default="TRN")
parser.add_option('-r', '--accum', action="store", dest="accum", help="accumulation steps", default="1")


options, args = parser.parse_args()
INPATH = options.rootpath

#INPATH='/Users/dhanley2/Documents/Personal/dfake'
sys.path.append(os.path.join(INPATH, 'utils' ))
from logs import get_logger
from utils import dumpobj, loadobj, chunks, pilimg, SpatialDropout
from sort import *
from sppnet import SPPNet

# Print info about environments
logger = get_logger('Video to image :', 'INFO') 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Device : {}'.format(torch.cuda.get_device_name(0)))
logger.info('Cuda available : {}'.format(torch.cuda.is_available()))
n_gpu = torch.cuda.device_count()
logger.info('Cuda n_gpus : {}'.format(n_gpu ))


logger.info('Load params')
for (k,v) in options.__dict__.items():
    logger.info('{}{}'.format(k.ljust(20), v))

SEED = int(options.seed)
SIZE = int(options.size)
FOLD = int(options.fold)
BATCHSIZE = int(options.batchsize)
METAFILE = os.path.join(INPATH, 'data', options.metafile)
WTSFILES = os.path.join(INPATH, options.wtspath)
WTSPATH = os.path.join(INPATH, options.wtspath)
IMGDIR = os.path.join(INPATH, options.imgpath)
EPOCHS = int(options.epochs)
START = int(options.start)
LR=float(options.lr)
LRGAMMA=float(options.lrgamma)
DECAY=float(options.decay)
INFER=options.infer
ACCUM=int(options.accum)

SEED = int(options.seed)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# METAFILE='/Users/dhanley2/Documents/Personal/dfake/data/trainmeta.csv.gz'
metadf = pd.read_csv(METAFILE)
logger.info('Full video file shape {} {}'.format(*metadf.shape))


metadf = pd.read_csv(METAFILE)
logger.info('Full meta file shape {} {}'.format(*metadf.shape))
TRKFILES = glob.glob(os.path.join(INPATH,IMGDIR) + '/track*')
trkdf = pd.concat([pd.read_csv(f) for f in TRKFILES], 0)
trkdf = trkdf[['video', 'boxdim']].drop_duplicates().reset_index(drop=True)
metadf = pd.merge(metadf, trkdf)
logger.info('Full meta file after boxdim added {} {}'.format(*metadf.shape))


def sortDfDim(df, size_clip = 256, bsize = 16):  
    '''
    Sort the dataframe to get similar sized boxes
    ''' 
    max_dim = df.boxdim.max()
    n_batches = int(np.floor(len(df)/float(bsize)))
    batch_steps = np.array(random.sample(range(n_batches), n_batches))
    sorted_ix = df.sort_values('boxdim').index.values
    batchidx = np.repeat(batch_steps, bsize)
    loaderidx = sorted_ix[batchidx.argsort()]
    df = df.iloc[loaderidx].reset_index(drop=True)
    df['batch'] =     batchidx
    df['boxgrpmax'] = df.groupby(['batch'])['boxdim'].transform(max)
    df['boxgrpmax'] = df['boxgrpmax'].clip(1, size_clip )
    return df

# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class SPPSeqNet(nn.Module):
    def __init__(self, backbone, embed_size, pool_size=(1, 2, 6), pretrained=True, \
                 dense_units = 256, dropout = 0.2):
        # Only resnet is supported in this version
        super(SPPSeqNet, self).__init__()
        self.sppnet = SPPNet(backbone=34, pool_size=pool_size, folder=WTSPATH)
        self.dense_units = dense_units
        self.lstm1 = nn.LSTM(embed_size, self.dense_units, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(self.dense_units*2, self.dense_units*2)
        self.linear_out = nn.Linear(self.dense_units*2, 1)
        self.embedding_dropout = SpatialDropout(dropout)
    
    def forward(self, x):
        # Input is batch of image sequences
        batch_size, seqlen = x.size()[:2]
        # Flatten to make a single long list of frames
        x = x.view(batch_size * seqlen, *x.size()[2:])
        # Pass each frame thru SPPNet
        emb = self.sppnet(x.permute(0,3,1,2))
        # Split back out to batch
        emb = emb.view(batch_size, seqlen, emb.size()[1])
        emb = self.embedding_dropout(emb)
        
        # Pass batch thru sequential model(s)
        h_lstm1, _ = self.lstm1(emb)
        max_pool, _ = torch.max(h_lstm1, 1)
        h_pool_linear = F.relu(self.linear1(max_pool))
        
        # Max pool and linear layer
        hidden = max_pool + h_pool_linear
        
        # Classifier
        out = self.linear_out(hidden)
        return out

# IMGDIR='/Users/dhanley2/Documents/Personal/dfake/data/npimg'
# https://www.kaggle.com/alexanderliao/image-augmentation-demo-with-albumentation/notebook
# https://albumentations.readthedocs.io/en/latest/augs_overview/image_only/image_only.html#blur

'''
In this dataset, no video was subjected to more than one augmentation.
- reduce the FPS of the video to 15
- reduce the resolution of the video to 1/4 of its original size
- reduce the overall encoding quality.
'''

def snglaugfn(imgdim):
    rot = random.randrange(-10, 10)
    dim1 = random.uniform(0.7, 1.0)
    dim2 = random.randrange(SIZE//2, SIZE)
    return Compose([
        ShiftScaleRotate(p=0.5, rotate_limit=(rot,rot)),
        CenterCrop(int(imgdim*dim1), int(imgdim*dim1), always_apply=False, p=0.5), 
        Resize(dim2, dim2, interpolation=1,  p=0.5),
        Resize(imgdim, imgdim, interpolation=1,  p=1),
        ])

mean_img = [0.4258249 , 0.31385377, 0.29170314]
std_img = [0.22613944, 0.1965406 , 0.18660679]

p1 = 0.1
trn_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.Downscale(scale_min=0.5, scale_max=0.9, interpolation=0, always_apply=False, p=0.5),
            ]),
        A.OneOf([
            A.GaussNoise(var_limit=(100.0, 600.0), p=p1),
            A.ISONoise(color_shift=(0.2, 0.25), intensity=(0.2, 0.25), p=p1),
            A.MultiplicativeNoise(multiplier=[0.7, 1.6], elementwise=False, per_channel=False, p=p1),
            A.NoOp(p=p1*3),
            ]),
        A.OneOf([
            A.Blur(blur_limit=15, p=p1),
            A.GaussianBlur(blur_limit=15, p=p1), 
            A.MotionBlur(blur_limit=(15), p=p1), 
            A.MedianBlur(blur_limit=10, p=p1),
            A.NoOp(p=p1*3),
            ]),
        A.OneOf([
             A.RandomGamma(gamma_limit=(50, 150), p=p1),
             A.RandomBrightness(limit=0.4, p=p1),
             A.RandomContrast(limit=0.4, p=p1),
             A.NoOp(p=p1*3),
            ]),
        A.OneOf([
             A.JpegCompression(quality_lower=30, quality_upper=100, always_apply=False, p=p1),
             A.ImageCompression(quality_lower=30, quality_upper=100, always_apply=False, p=p1),
             A.NoOp(p=p1*2),
            ]),
        A.OneOf([
             A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=p1),
             A.RandomShadow( p=p1),
             A.NoOp(p=p1*12),
            ]),
        A.OneOf([
            A.CoarseDropout(max_holes=50, max_height=20, max_width=20, min_height=6, min_width=6, p=p1),
            A.Cutout(num_holes=12, max_h_size=24, max_w_size=24, fill_value=255, p=p1),
            A.CLAHE(clip_limit=2.0, p=p1),
            A.NoOp(p=p1*12),
            ]),
    ])

val_transforms = Compose([
    NoOp(),
    #JpegCompression(quality_lower=50, quality_upper=50, p=1.0),
    ])

transform_norm = Compose([
    #JpegCompression(quality_lower=75, quality_upper=75, p=1.0),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
    ])
    
class DFakeDataset(Dataset):
    def __init__(self, df, imgdir, aug_ratio = 5, train = False, val = False, labels = False, maxlen = 32):
        self.data = df.copy()
        self.data.label = (self.data.label == 'FAKE').astype(np.int8)
        self.imgdir = imgdir
        self.framels = sorted(os.listdir(imgdir))
        self.labels = labels
        self.data = self.data[self.data.video.str.replace('.mp4', '.npz').isin(self.framels)]
        self.data = pd.concat([self.data.query('label == 0')]*5+\
                               [self.data.query('label == 1')])
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        # self.data = pd.concat([ self.data[self.data.video.str.contains('qirlrtrxba')],  self.data[:500].copy() ]).reset_index(drop=True)
        self.maxlen = maxlen
        logger.info('Expand the REAL class {} {}'.format(*self.data.shape))
        self.snglaug = snglaugfn
        self.train = train
        self.val = val
        self.norm = transform_norm
        self.transform = trn_transforms if not val else val_transforms
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vid = self.data.loc[idx]
        # Apply constant augmentation on combined frames
        fname = os.path.join(self.imgdir, vid.video.replace('mp4', 'npz'))
        try:
            frames = np.load(fname)['arr_0']
            # Cut the frames to max 37 with a sliding window
            d0,d1,d2,d3 = frames.shape
            if self.train and (d0>self.maxlen):
                xtra = frames.shape[0]-self.maxlen
                shift = random.randint(0, xtra)
                frames = frames[xtra-shift:xtra-shift+self.maxlen]
            else:
                frames = frames[:self.maxlen]
            d0,d1,d2,d3 = frames.shape
            augsngl = self.snglaug
            # Standard augmentation on each image
            augfn = self.snglaug(d2)
            if self.train : frames = np.stack([augfn(image=f)['image'] for f in frames])
            frames = frames.reshape(d0*d1, d2, d3)
            if self.train or self.val:
                augmented = self.transform(image=frames)
                frames = augmented['image']               
            '''
            augmented = self.norm(image=frames)
            frames = augmented['image']
            frames = frames.resize_(d0,d1,d2,d3)
            '''
            logger.info(frames.shape)
            logger.info(type(frames))
            if self.train:
                labels = torch.tensor(vid.label)
                return {'frames': frames, 'idx': idx, 'labels': labels}    
            else:      
                return {'frames': frames, 'idx': idx}
        except Exception:
            logger.exception('Failed to load numpy array {}'.format(fname))
               
def collatefn(batch):
    # Remove error reads
    batch = [b for b in batch if b is not None]
    maxdim = max([b['frames'].shape[1] for b in batch])
    batchlen = [b['frames'].shape[0]//b['frames'].shape[1] for b in batch]

    for d, bl in zip(batch, batchlen):
        if d['frames'].shape[1] != maxdim:
            d['frames'] = cv2.resize(d['frames'], size=(maxdim, maxdim*bl), interpolation=cv2.INTER_LINEAR) 

        augmented = transform_norm(image=frames)
        d['frames'] = augmented['image']
        d['frames'] = d['frames'].resize_(batchlen,maxdim,maxdim,3)

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
    
    if 'labels' in batch[0]:
        y_batch = torch.tensor([l['labels'] for l in batch])
        return {'frames': x_batch, 'ids': ids, 'seqlen': seqlen, 'labels': y_batch}
    else:
        return {'frames': x_batch, 'ids': ids, 'seqlen': seqlen}        
   

 
logger.info('Create loaders...')
# IMGDIR='/Users/dhanley2/Documents/Personal/dfake/data/npimg'
# BATCHSIZE=2
trndf = metadf.query('fold != @FOLD').reset_index(drop=True)
valdf = metadf.query('fold == @FOLD').reset_index(drop=True)#.head(1024)

trndataset = DFakeDataset(trndf, IMGDIR, train = True, val = False, labels = True, maxlen = 32)
valdataset = DFakeDataset(valdf, IMGDIR, train = False, val = True, labels = False, maxlen = 32)
trnloader = DataLoader(trndataset, batch_size=BATCHSIZE, shuffle=True, num_workers=8, collate_fn=collatefn)
valloader = DataLoader(valdataset, batch_size=BATCHSIZE, shuffle=False, num_workers=8, collate_fn=collatefn)


logger.info('Create model')
poolsize=(1, 2, 6)
embedsize = 512*sum(i**2 for i in poolsize)
bb=34
du=256
do=0.2
model = SPPSeqNet(backbone=bb, pool_size=poolsize, dense_units = du, \
                  dropout = do, embed_size = embedsize)
model = model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = optim.Adam(plist, lr=LR)
scheduler = StepLR(optimizer, 1, gamma=LRGAMMA, last_epoch=-1)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
criterion = torch.nn.BCEWithLogitsLoss()

ypredvalls = []
for epoch in range(EPOCHS):
    logger.info('Epoch {}/{}'.format(epoch, EPOCHS - 1))
    logger.info('-' * 10)
    model_file_name = 'weights/sppnet_epoch{}_lr{}_accum{}_fold{}.bin'.format(epoch, LR, ACCUM, FOLD)

    trnsortdf = sortDfDim(trndf, size_clip = SIZE, bsize = 8*16).head(64)#.tail(8).reset_index(drop=True)
    valsortdf = sortDfDim(valdf, size_clip = SIZE, bsize = 8*16).head(64)
    logger.info(trnsortdf.shape)
    logger.info(trnsortdf.head())
    trndataset = DFakeDataset(trnsortdf, IMGDIR, train = True, val = False, labels = True, maxlen = 32)
    valdataset = DFakeDataset(valsortdf, IMGDIR, train = True, val = False, labels = True, maxlen = 32)
    trnloader = DataLoader(trndataset, batch_size=BATCHSIZE, shuffle=False, num_workers=8, collate_fn=collatefn)
    valloader = DataLoader(valdataset, batch_size=BATCHSIZE, shuffle=False, num_workers=8, collate_fn=collatefn)


    if epoch<START:
        logger.info('Load checkpoint : {}'.format(model_file_name))
        model = SPPSeqNet(backbone=bb, pool_size=poolsize, dense_units = du, \
                  dropout = do, embed_size = embedsize)
        model.load_state_dict(torch.load(model_file_name))
        model.to(device)
        continue
    if INFER not in ['TST', 'EMB', 'VAL']:

        tr_loss = 0.
        for param in model.parameters():
            param.requires_grad = True
        model.train()  
        for step, batch in enumerate(trnloader):
            logger.info(x.shape)
            x = batch['frames'].to(device, dtype=torch.float)
            y = batch['labels'].to(device, dtype=torch.float)
            logger.info(x.shape)
            x = torch.autograd.Variable(x, requires_grad=True)
            y = torch.autograd.Variable(y)
            y = y.unsqueeze(1)
            out = model(x)
            # Get loss
            loss = criterion(out, y)
                  dropout = do, embed_size = embedsize)
        model.load_state_dict(torch.load(model_file_name))
        model.to(device)
        model.eval()
        ypredval = []
        valids = [] 
        with torch.no_grad():
            for step, batch in enumerate(valloader):
                x = batch['frames'].to(device, dtype=torch.float)
                out = model(x)
                out = torch.sigmoid(out)
                ypredval.append(out.cpu().detach().numpy())
                valids.append(batch['ids'].cpu().detach().numpy())
                if step%200==0:
                    logger.info('Val step {} of {}'.format(step, len(valloader)))    
                del x, out, batch
        ypredval = np.concatenate(ypredval).flatten()
        valids = np.concatenate(valids).flatten()
        ypredvalls.append(ypredval)
        yactval = valdataset.data.iloc[valids].label.values
        logger.info('Actuals {}'.format(yactval[:8]))
        logger.info('Preds {}'.format(ypredval[:8]))
        logger.info('Ids {}'.format(valids[:8]))
        for c in [.1, .01, .001] :
            valloss = log_loss(yactval, ypredval.clip(c,1-c))
            logger.info('Epoch {} val single; clip {:.3f} logloss {:.5f}'.format(epoch,c, valloss))
        for c in [.1, .01, .001] :
            BAGS=3
            ypredvalbag = sum(ypredvalls[-BAGS:])/len(ypredvalls[-BAGS:])
            valloss = log_loss(yactval, ypredvalbag.clip(c,1-c))
            logger.info('Epoch {} val bags {}; clip {:.3f} logloss {:.5f}'.format(epoch, len(ypredvalls[:BAGS]), c, valloss))
        del yactval, ypredval, valids
''' 
logger.info('Write out bagged prediction to preds folder')
yvaldf = valdataset.data.iloc[valids][['video', 'label']]
yvaldf['pred'] = ypredval 
yvaldf.to_csv('preds/dfake_sppnet_sub_epoch{}.csv.gz'.format(epoch), \
            index = False, compression = 'gzip')
'''

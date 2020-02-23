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
import itertools
import warnings
warnings.filterwarnings("ignore")
# sys.path.append('/share/dhanley2/dfake/scripts/opt/conda/lib/python3.6/site-packages')
# sys.path.append('/share/dhanley2/dfake/scripts/pim')
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
parser.add_option('-s', '--skip', action="store", dest="skip", help="Skip every k frames", default="1")
parser.add_option('-t', '--arch', action="store", dest="arch", help="Architecture", default="mixnet_l")
parser.add_option('-u', '--stop', action="store", dest="stop", help="Start epochs", default="13")
parser.add_option('-v', '--fakesamp', action="store", dest="fakesamp", help="Mix in ratio of fake samples", default="0")
parser.add_option('-w', '--origsamp', action="store", dest="origsamp", help="Max mix in ratio of original", default="0")
parser.add_option('-x', '--realsamp', action="store", dest="realsamp", help="Max mix in ratio of other reals", default="0")
parser.add_option('-y', '--jpegcomplwr', action="store", dest="jpegcomplwr", help="Lower bound for jpeg or image compression", default="30")
parser.add_option('-z', '--resizelwr', action="store", dest="resizelwr", help="Lower bound for resizing", default="0.3")


options, args = parser.parse_args()
INPATH = options.rootpath

#INPATH='/Users/dhanley2/Documents/Personal/dfake'
sys.path.append(os.path.join(INPATH, 'utils' ))
print(os.path.join(INPATH, 'utils' ))
from logs import get_logger
from utils import dumpobj, loadobj, chunks, pilimg, SpatialDropout
from sort import *
from sppnet import SPPNet
import timm
from splits import  load_folds, get_real_to_fake_dict_v2, get_cluster_data


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

SKIP = int(options.skip)
SEED = int(options.seed)
SIZE = int(options.size)
FOLD = int(options.fold)
BATCHSIZE = int(options.batchsize) * SKIP
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
ORIGSAMP=int(options.origsamp)
FAKESAMP=int(options.fakesamp)
REALSAMP=int(options.realsamp)
JPEGCOMPRESSION=int(options.jpegcomplwr)
RESIZERATIO=float(options.resizelwr)

# METAFILE='/Users/dhanley2/Documents/Personal/dfake/data/trainmeta.csv.gz'
metadf = pd.read_csv(METAFILE)
logger.info('Full video file shape {} {}'.format(*metadf.shape))


n_gpu = torch.cuda.device_count()
logger.info('Cuda n_gpus : {}'.format(n_gpu ))

METADATA_PATH=os.path.join(INPATH, 'data/splits')
REAL_VIDEO_TO_CLUSTER, FAKE_VIDEO_TO_CLUSTER, CLUSTER_TO_REAL_VIDEOS, \
            CLUSTER_TO_FAKE_VIDEOS, CLUSTER_TO_DESCRIPTOR = \
            get_cluster_data(METADATA_PATH)

ACTORS = [[i+'.mp4' for i in m] for m in CLUSTER_TO_REAL_VIDEOS.values()]
REALVIDSAMEACTOR = dict(itertools.chain(*[[(im, m) for im in  m] \
                                           for m in ACTORS]))

random.sample(REALVIDSAMEACTOR['eopcdsfkzm.mp4'], 3)
# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class SPPSeqNet(nn.Module):
    def __init__(self, backbone, embed_size, pool_size=(1, 2, 6), pretrained=True, \
                 architecture = 'resnet', dense_units = 256, dropout = 0.2):
        # Only resnet is supported in this version
        super(SPPSeqNet, self).__init__()
        logger.info('{} {} {}'.format(architecture, backbone, WTSPATH))
        self.sppnet = SPPNet(architecture=architecture, backbone=backbone, pool_size=pool_size, folder=WTSPATH)
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

'''i
In this dataset, no video was subjected to more than one augmentation.
- reduce the FPS of the video to 15
- reduce the resolution of the video to 1/4 of its original size
- reduce the overall encoding quality.
'''

def predictval(model, loader):
    model.eval()
    ypredval = []
    valids = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            x = batch['frames'].to(device, dtype=torch.float)
            out = model(x)
            out = torch.sigmoid(out)
            ypredval.append(out.cpu().detach().numpy())
            valids.append(batch['ids'].cpu().detach().numpy())
            if step%200==0:
                logger.info('Val step {} of {}'.format(step, len(loader)))
    ypredval = np.concatenate(ypredval).flatten()
    valids = np.concatenate(valids).flatten()
    yactval = loader.dataset.data.iloc[valids].label.values
    return ypredval, valids, yactval

def snglaugfn():
    rot = random.randrange(-10, 10)
    dim1 = random.uniform(0.7, 1.0)
    #dim2 = random.randrange(int(SIZE)*0.75, SIZE)
    return Compose([
        ShiftScaleRotate(p=0.5, rotate_limit=(rot,rot)),
        CenterCrop(int(SIZE*dim1), int(SIZE*dim1), always_apply=False, p=0.5), 
        #Resize(dim2, dim2, interpolation=1,  p=0.5),
     	   Resize(SIZE, SIZE, interpolation=1,  p=1),
        ])

#mean_img = [0.4258249 , 0.31385377, 0.29170314]
#std_img = [0.22613944, 0.1965406 , 0.18660679]
mean_img = [0.485, 0.456, 0.406]
std_img = [0.229, 0.224, 0.225]

p1 = 0.1
trn_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(100.0, 600.0), p=p1),
            A.ISONoise(color_shift=(0.2, 0.25), intensity=(0.2, 0.25), p=p1),
            A.MultiplicativeNoise(multiplier=[0.7, 1.6], elementwise=False, per_channel=False, p=p1),
            A.NoOp(p=p1*9),
            ]),
        A.OneOf([
            A.Blur(blur_limit=15, p=p1),
            A.GaussianBlur(blur_limit=15, p=p1), 
            A.MotionBlur(blur_limit=(15), p=p1), 
            A.MedianBlur(blur_limit=10, p=p1),
            A.NoOp(p=p1*9),
            ]),
        A.OneOf([
             A.RandomGamma(gamma_limit=(50, 150), p=p1),
             A.RandomBrightness(limit=0.4, p=p1),
             A.RandomContrast(limit=0.4, p=p1),
             A.NoOp(p=p1*9),
            ]),
        A.OneOf([
            A.OneOf([
                A.JpegCompression(quality_lower=JPEGCOMPRESSION, quality_upper=100, always_apply=False, p=p1),
                A.ImageCompression(quality_lower=JPEGCOMPRESSION, quality_upper=100, always_apply=False, p=p1),
                ]),
            A.OneOf([
                A.Downscale(scale_min=RESIZERATIO, scale_max=0.9, interpolation=cv2.INTER_NEAREST, always_apply=False, p=p1),
                A.Downscale(scale_min=RESIZERATIO, scale_max=0.9, interpolation=cv2.INTER_LINEAR, always_apply=False, p=p1),
                A.Downscale(scale_min=RESIZERATIO, scale_max=0.9, interpolation=cv2.INTER_CUBIC, always_apply=False, p=p1),
                A.Downscale(scale_min=RESIZERATIO, scale_max=0.9, interpolation=cv2.INTER_AREA, always_apply=False, p=p1),
                ]),
            ]),
    ])

val_transforms = Compose([
    NoOp(),
    #JpegCompression(quality_lower=50, quality_upper=50, p=1.0),
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

# Simulate downsize of 0.4
small_val_transforms = Compose([
    A.Downscale(scale_min=0.4, scale_max=0.4, interpolation=cv2.INTER_LINEAR, always_apply=False, p=p1),
    ])



def makevaldf(yactval, ypredval):
    vdf = pd.DataFrame({'label': yactval, 'pred': ypredval})
    vdf = pd.concat([vdf.query('label == 0')]*5+[vdf.query('label == 1')])
    return vdf    

class DFakeDataset(Dataset):
    def __init__(self, df, imgdir, framels, small=False, aug_ratio = 5, train = False, val = False, labels = False, maxlen = 32 // SKIP):
        self.data = df.copy()
        self.data.label = (self.data.label == 'FAKE').astype(np.int8)
        self.imgdir = imgdir
        self.framels = framels # os.listdir(imgdir)
        self.labels = labels
        self.data = self.data[self.data.video.str.replace('.mp4', '.npz').isin(self.framels)]
        if train:
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
        self.transform = trn_transforms if not val else (val_transforms if not small else small_val_transforms )
  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid = self.data.loc[idx]
        # Apply constant augmentation on combined frames
        fname = os.path.join(self.imgdir, vid.video.replace('mp4', 'npz'))
        frames = np.load(fname)['arr_0']
        if (SKIP>1) and (frames.shape[0] > SKIP*6):
            every_k = random.randint(0,SKIP-1)
            frames = np.stack([b for t,b in enumerate(frames) if t%SKIP==every_k])
        framesbackup = frames.copy()
        try:
            # if it is a FAKE label in trainnig, 60% apply mixing of augmentations
            if self.train:
                try:
                    # Get out a random sample of fake and real
                    ascdf = self.data[ self.data.original == vid.original].reset_index()
                    ascdf = ascdf[['video', 'original']].drop_duplicates().reset_index(drop=True)
                    
                    if vid.label==1:
                        mixinfake = min(random.randint(0,FAKESAMP), ascdf.shape[0])
                        mixinorig = min(random.randint(0,ORIGSAMP), ascdf.shape[0])
                        mixinreal = min(random.randint(0,REALSAMP), ascdf.shape[0])
                        ascls = [vid.video] + ascdf.sample(  mixinfake ).video.tolist() + \
                                        random.sample(REALVIDSAMEACTOR[vid.video], mixinreal) # ascdf.original[: mixinorig  ].tolist()

                    if vid.label==0:
                        mixinreal = min(random.randint(0,REALSAMP), ascdf.shape[0])
                        ascls = [vid.video] + random.sample(REALVIDSAMEACTOR[vid.video], mixinreal)
                    # logger.info('{} : {}'.format(vid.label, ' '.join(ascls)  ) )    
                    framesdict = dict((k, np.load(os.path.join(self.imgdir, \
                                        k.replace('.mp4', '.npz')))['arr_0']) \
                                        for k in set(ascls) )
                    frames = []
                    # Pick from a random video
                    if (SKIP>1) and (framesdict[vid.video].shape[0] > SKIP*6):
                        every_k = random.randint(0,SKIP-1)
                        for t,b in enumerate(framesdict[vid.video]):
                            if t%SKIP!=every_k:
                                continue
                            try:
                                randsamp = random.choice(ascls)
                                frames.append(framesdict[randsamp][t])
                            except:
                                frames.append(b)
                        frames = np.stack(frames)
                    else:
                        frames = framesbackup.copy()
                except:
                    frames = framesbackup.copy()
            else:
                frames = framesbackup.copy()
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
            augfn = self.snglaug()
            if self.train : frames = np.stack([augfn(image=f)['image'] for f in frames])
            frames = frames.reshape(d0*d1, d2, d3)
            if self.train or self.val:
                augmented = self.transform(image=frames)
                frames = augmented['image']               
            augmented = self.norm(image=frames)
            frames = augmented['image']
            frames = frames.resize_(d0,d1,d2,d3)
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
valdf = metadf.query('fold == @FOLD').reset_index(drop=True)

FRAMELS = pd.read_csv(os.path.join(IMGDIR, '../cropped_faces08.txt'), header=None).iloc[:,0].tolist()
trndataset = DFakeDataset(trndf.head(2000), IMGDIR, FRAMELS, train = True, val = False, labels = True, maxlen = 48 // SKIP)
valdataset = DFakeDataset(valdf.head(2000), IMGDIR, FRAMELS, train = False, val = True, labels = False, maxlen = 48 // SKIP)
smldataset = DFakeDataset(valdf.head(2000), IMGDIR, FRAMELS, train = False, val = True, labels = False, maxlen = 48 // SKIP, small=True)
trnloader = DataLoader(trndataset, batch_size=BATCHSIZE, shuffle=True, num_workers=16, collate_fn=collatefn)
valloader = DataLoader(valdataset, batch_size=BATCHSIZE*2, shuffle=False, num_workers=16, collate_fn=collatefn)
smlloader = DataLoader(smldataset, batch_size=BATCHSIZE*2, shuffle=False, num_workers=16, collate_fn=collatefn)



logger.info('Create model')
poolsize=(1, 2) # , 6)
embedsize = 512*sum(i**2 for i in poolsize)
# embedsize = 384*sum(i**2 for i in poolsize)

model = SPPSeqNet(backbone=50, pool_size=poolsize, dense_units = 256, \
                  architecture = options.arch, dropout = 0.2, embed_size = embedsize)
model = model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = optim.Adam(plist, lr=LR)
# scheduler = StepLR(optimizer, 1, gamma=LRGAMMA, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
criterion = torch.nn.BCEWithLogitsLoss()

if n_gpu > 0:
    model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

ypredvalls = []
ypredvalsmls=[]
for epoch in range(EPOCHS):
    LRate = scheduler.get_lr()[0]
    logger.info('Epoch {}/{} LR {:.9f}'.format(epoch, EPOCHS - 1, LRate))
    logger.info('-' * 10)
    model_file_name = 'weights/sppnet_{}_epoch{}_lr{}_accum{}_fold{}.bin'.format(options.arch, epoch, LR, ACCUM, FOLD)
    if epoch > int(options.stop):
        continue
    if epoch<START:
        if epoch == (START - 1):
            model = SPPSeqNet(backbone=50, pool_size=poolsize, dense_units = 256, \
                  architecture = options.arch, dropout = 0.2, embed_size = embedsize)
            if n_gpu > 0:
                model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
            model.load_state_dict(torch.load(model_file_name))
            model.to(device)
        scheduler.step()
        continue
    if INFER not in ['TST', 'EMB', 'VAL']:

        tr_loss = 0.
        for param in model.parameters():
            param.requires_grad = True
        model.train()  
        for step, batch in enumerate(trnloader):
            x = batch['frames'].to(device, dtype=torch.float)
            y = batch['labels'].to(device, dtype=torch.float)
            x = torch.autograd.Variable(x, requires_grad=True)
            y = torch.autograd.Variable(y)
            y = y.unsqueeze(1)
            out = model(x)
            # Get loss
            loss = criterion(out, y)
            tr_loss += loss.item()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if step % ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step%100==0:
                logger.info('Trn step {} of {} trn lossavg {:.5f}'. \
                        format(step, len(trnloader), (tr_loss/(1+step))))
            del x, y, out, batch
        torch.save(model.state_dict(), model_file_name)
        scheduler.step()
    else:
        model = SPPSeqNet(backbone=50, pool_size=poolsize, dense_units = 256, \
                  architecture = options.arch, dropout = 0.2, embed_size = embedsize)
        if n_gpu > 0:
            model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
        model.load_state_dict(torch.load(model_file_name))
        model.to(device)
    if INFER in ['VAL', 'TRN']:

        BAGS=3

        logger.info('Evaluate on 88')
        ypredval, valids, yactval = predictval(model, smlloader)
        ypredvalsmls.append(ypredval)
        vdf = makevaldf(yactval, smldataset.data.iloc[valids].label.values)
        vdfb = makevaldf(yactval, sum(ypredvalsmls[-BAGS:])/len(ypredvalsmls[-BAGS:])  )
        for c in [.01] :
            valloss = log_loss(vdf.label.values, vdf.pred.values.clip(c,1-c))
            logger.info('Epoch {} val  88 single; clip {:.3f} logloss {:.5f}'.format(epoch,c, valloss))
            valloss = log_loss(vdfb.label.values, vdfb.pred.values.clip(c,1-c))
            logger.info('Epoch {} val  88 bags {}; clip {:.3f} logloss {:.5f}'.format(epoch, len(ypredvalsmls[:BAGS]), c, valloss))

        logger.info('Evaluate on 224')      
        ypredval, valids, yactval = predictval(model, valloader)
        ypredvalls.append(ypredval)
        vdf = makevaldf(yactval, valdataset.data.iloc[valids].label.values)
        vdfb = makevaldf(yactval, sum(ypredvalls[-BAGS:])/len(ypredvalls[-BAGS:])  )
        for c in [.01] :
            valloss = log_loss(vdf.label.values, vdf.pred.values.clip(c,1-c))
            logger.info('Epoch {} val 224 single; clip {:.3f} logloss {:.5f}'.format(epoch,c, valloss))
            valloss = log_loss(vdfb.label.values, vdfb.pred.values.clip(c,1-c))
            logger.info('Epoch {} val 224 bags {}; clip {:.3f} logloss {:.5f}'.format(epoch, len(ypredvalls[:BAGS]), c, valloss))

        logger.info('Write out bagged prediction to preds folder')
        yvaldf = valdataset.data.iloc[valids][['video', 'label']]
        yvaldf['pred'] = ypredval 
        yvaldf.to_csv('preds/dfake_{}_sub_epoch{}_fold{}.csv.gz'.format(options.arch, epoch, FOLD), \
            index = False, compression = 'gzip')

        del yactval, ypredval, valids


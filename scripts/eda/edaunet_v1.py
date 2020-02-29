import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import glob
import json
import cv2
from math import pi
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision
from scipy import ndimage
import imutils

from torchvision import models, transforms
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
import segmentation_models_pytorch as smp
warnings.filterwarnings("ignore")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

from typing import Optional, Union, List
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import encoders
import torch.utils.model_zoo as model_zoo

INPATH = '/Users/dhanley2/Documents/Personal/dfake'
#INPATH='/Users/dhanley2/Documents/Personal/dfake'
sys.path.append(os.path.join(INPATH, 'utils' ))
from logs import get_logger
from utils import dumpobj, loadobj, chunks, pilimg, SpatialDropout
from utils import makegray, maskit
from sort import *
from sppnet import SPPNet
import segnet

# Print info about environments
logger = get_logger('Video to image :', 'INFO') 

class DblHeadUnet(nn.Module):
    def __init__(self, backbone, aux_params, inpath, pretrained=True, dense_units = 256, 
                 dropout = 0.5):
        # Only resnet is supported in this version
        super(DblHeadUnet, self).__init__()
        self.unet = segnet.Unet(backbone, encoder_weights='imagenet', classes=1, 
                                aux_params=aux_params, inpath = inpath)
        self.dense_units = dense_units
        self.in_channels = self.unet.encoder._conv_head.in_channels
        self.lstm1 = nn.LSTM(self.in_channels, self.dense_units, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(self.dense_units*2, self.dense_units*2)
        self.linear_out = nn.Linear(self.dense_units*2, 1)
        self.embedding_dropout = SpatialDropout(dropout)
    
    def forward(self, x):
        # Input is batch of image sequences
        batch_size, seqlen = x.size()[:2]
        # Flatten to make a single long list of frames
        x = x.view(batch_size * seqlen, *x.size()[2:])
        logger.info(x.shape)
        # Pass each frame thru SPPNet
        outmask, outgap = self.unet(x)#.permute(0,3,1,2))
        # Split back out to batch
        outgap = outgap.view(batch_size, seqlen, outgap.size()[1])
        outgap = self.embedding_dropout(outgap)
        
        # Pass batch thru sequential model(s)
        h_lstm1, _ = self.lstm1(outgap)
        max_pool, _ = torch.max(h_lstm1, 1)
        h_pool_linear = F.relu(self.linear1(max_pool))
        
        # Max pool and linear layer
        hidden = max_pool + h_pool_linear
        # Classifier
        out = self.linear_out(hidden)
        
        # Squeeze outmask
        outmask = outmask.squeeze(1)
        
        return out, outmask

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation=None,      # activation function, default is None
    classes=1,                 # define number of output labels
)

os.environ['TORCH_HOME'] = os.path.join(INPATH, 'weights')


unet = Unet('efficientnet-b0', encoder_weights='imagenet', classes=1, aux_params=aux_params)

model = DblHeadUnet(backbone='efficientnet-b1', dense_units = 256, dropout = 0.5, 
                    aux_params=aux_params, inpath = INPATH)

seqlen = torch.tensor([6,8,4,7])
maxlen = 8



INPATH = '/Users/dhanley2/Documents/Personal/dfake'
METAFILE = os.path.join(INPATH, 'data', 'trainmeta.csv.gz')
metadf = pd.read_csv(METAFILE)
MAPFILES = sorted(glob.glob(os.path.join( INPATH, 'data/map/map*' )))
IMGFILES = sorted([i for i in glob.glob(os.path.join( INPATH, 'data/map/*.npz' )) if 'map__' not in i] )
batch = np.load(IMGFILES[1])['arr_0']
batch = torch.tensor(np.float32(batch/255)) - 0.5
d0,d1,d2,d3 = batch.shape
batch = batch.reshape(d0, d3, d1, d2)
%time out, outmask = model(batch[:8].unsqueeze(0))

b = batch[:8].unsqueeze(0)

b.view(-1,  *b.size()[2:]).shape


outmask = out[1]
out = out


torch.nn.MSELoss()
y = np.load(MAPFILES[1])['arr_0']
y = maskit(y, ksize = 2, scale = 8)
y = torch.tensor(np.float32(y)) / 255
y.shape


criterion = torch.nn.L1Loss()

criterion(y[:8], out[0].squeeze(1))




ldict = dict((v.replace('.mp4', ''),l) for t,(v,l) 
                    in metadf[['video', 'label']].iterrows())

matmap = np.load(MAPFILES[1])['arr_0']
matimg = np.load(IMGFILES[1])['arr_0']

Image.fromarray(matimg[0])
Image.fromarray(matmap[0])

mapmat=matmap[0]

morph(matmap[0], 5).max()

makegray(mat)

maskit(mat, ksize =2, its=2)

ix = 1
matimg = np.load(IMGFILES[ix])['arr_0']
masks = np.load(MAPFILES[ix])['arr_0']
Image.fromarray(matimg[-1])
Image.fromarray(mat[-1])
Image.fromarray(maskit(masks)[0])


Image.fromarray(makegray(mat, scale = 8)[0] , 'L')

bmat = makegray(mat, scale = 8)[0]
its=k=3
kernel = np.ones((k, k), np.uint8)
Image.fromarray(cv2.dilate(cv2.erode(bmat, kernel, iterations = its), kernel, iterations = its))



Image.fromarray(morph(mat, ksize = 5, scale = 8)[0] , 'L')

f = MAPFILES[1]
Image.fromarray(morph(np.load(f)['arr_0'][0], ksize = 5, scale = 8), 'L')

Image.fromarray(makegray(matmap[0]), 'L')




maxdf = pd.DataFrame([(ldict[f.split('__')[-1].replace('.npz', '')], \
                             morph(np.load(f)['arr_0'][0], ksize = 3, scale = 8).max()) 
                            for f in tqdm(MAPFILES[:100]) ])
maxdf.columns =['label', 'pixmax']
    
maxdf[ 'pixmax' ].hist(by=maxdf['label'], bins = 50, figsize=(6,15))


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

from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,
                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                           GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                           RandomBrightnessContrast, Lambda, NoOp, CenterCrop, Resize
                           )

from tqdm import tqdm
from apex import amp

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier



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


options, args = parser.parse_args()
INPATH = options.rootpath

#INPATH='/Users/dhanley2/Documents/Personal/dfake'
sys.path.append(os.path.join(INPATH, 'utils' ))
from logs import get_logger
from utils import dumpobj, loadobj, chunks, pilimg
from sort import *
from sppnet import SPPNet

# Print info about environments
logger = get_logger('Video to image :', 'INFO') 
logger.info('Cuda set up : time {}'.format(datetime.datetime.now().time()))

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
METAFILE = os.path.join(INPATH, 'data', options.metafile)
WTSFILES = os.path.join(INPATH, options.wtspath)
OUTDIR = os.path.join(INPATH, options.imgpath)

# METAFILE='/Users/dhanley2/Documents/Personal/dfake/data/trainmeta.csv.gz'
metadf = pd.read_csv(METAFILE)
metadf['video_path'] = os.path.join(INPATH, options.vidpath) + '/' + metadf['folder'] + '/' + metadf['video']
logger.info('Full video file shape {} {}'.format(*metadf.shape))

 
l = [np.load(i)['arr_0'][:1] for i in tqdm(IMGFILES)]
l = np.concatenate(l)#.shape
np.mean(l, axis=tuple(range(l.ndim-1)))/255.



# Load and visualise
IMGFILES = glob.glob(os.path.join(INPATH, 'data/npimg/*'))
ix = 51
vid = IMGFILES[ix].split('/')[-1].replace('npz', 'mp4')
print('Label {}'.format( metadf.set_index('video').loc[vid].label  ))
frames = np.load(IMGFILES[ix])['arr_0']
Image.fromarray(pilimg(frames))
# Image diff
Image.fromarray(pilimg( np.stack([((i.astype(np.int8)-j.astype(np.int8))+128).astype(np.uint8) \
                          for i,j in zip(frames[1:], frames[:-1])   ])))


    
# 
pool_size = (1, 2, 6)
netspp = SPPNet(backbone=34, pool_size=pool_size)
embed_size = sum(map(lambda x: x**2, (1, 2, 6)))*512

# Taken of one frame from each of 600 images
mean_img = [0.4258249 , 0.31385377, 0.29170314]
std_img = [0.22613944, 0.1965406 , 0.18660679]
transform_train = Compose([
    #Transpose(p=0.5),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])

augmented = transform_train(image=frames)
augmented = augmented ['image'].permute(1, 0, 2, 3)
augmented = augmented .unsqueeze(0)
augmented.size()

emb = netspp(augmented)#.shape()
emb = emb.unsqueeze(0)

DENSE_HIDDEN_UNITS = 256
lstm1 = nn.LSTM(embed_size, DENSE_HIDDEN_UNITS, bidirectional=True, batch_first=True)
linear1 = nn.Linear(DENSE_HIDDEN_UNITS*2, DENSE_HIDDEN_UNITS*2)
linear_out = nn.Linear(DENSE_HIDDEN_UNITS*2, 1)


h_lstm1, _ = lstm1(emb)
avg_pool = torch.mean(h_lstm1, 1)
h_pool_linear = F.relu(linear1(avg_pool))
hidden = avg_pool + h_pool_linear 
linear_out(hidden)


# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class NeuralNet(nn.Module):
    def __init__(self, embed_size=trnemb.shape[-1]*3, LSTM_UNITS=64, DO = 0.3):
        super(NeuralNet, self).__init__()
        
        self.embedding_dropout = SpatialDropout(0.0) #DO)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear = nn.Linear(LSTM_UNITS*2, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x

        h_embadd = torch.cat((h_embedding[:,:,:2048], h_embedding[:,:,:2048]), -1)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embadd

        output = self.linear(hidden)
        #output = self.linear(h_lstm1)
        
        return output
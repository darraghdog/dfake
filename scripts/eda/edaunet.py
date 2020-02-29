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
from segmentation_models_pytorch.encoders import encoders
import torch.utils.model_zoo as model_zoo


INPATH = '/Users/dhanley2/Documents/Personal/dfake'

def get_encoder(name, in_channels=3, depth=5, weights=None):
    os.environ['TORCH_HOME'] = os.path.join(INPATH, 'weights')
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels)

    return encoder

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        super().__init__(pool, flatten)

class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)
    Returns:
        ``torch.nn.Module``: **Unet**
    .. _Unet:
        https://arxiv.org/pdf/1505.04597
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation=None,      # activation function, default is None
    classes=1,                 # define number of output labels
)

os.environ['TORCH_HOME'] = os.path.join(INPATH, 'weights')
model = Unet('efficientnet-b1', encoder_weights='imagenet', classes=1, aux_params=aux_params)
dir(model.classification_head)
model.classification_head


def makegray(mapmat, scale = 3):
    outmat = (abs(mapmat.mean(-1) - 128) * 2 )
    outmat[outmat < 4] = 0
    outmat[outmat > 251] = 0
    outmat = outmat * scale
    outmat = outmat.clip(0,255)
    return np.uint8(outmat )

def morph(mat, ksize = 10, scale = 3):
    mat = makegray(mat, scale)
    mat = np.stack(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones([ksize, ksize])) for m in mat)
    return mat
    
def catimg(mat):
    img = np.concatenate((mat, 
                          makegray(mat, scale = 8), 
                          morph(mat, ksize = 5, scale = 10)), 1)
    return img

def maskit(bmasks, ksize =2, its=2, scale = 8):
    bmasks = makegray(bmasks, scale)
    kernel = np.ones((ksize, ksize), np.uint8)
    bmasks = np.stack([ cv2.erode(b, kernel, iterations = its) for b in bmasks  ])
    bmasks = np.stack([ cv2.dilate(b, kernel, iterations = its) for b in bmasks  ])
    return bmasks

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation=None,      # activation function, default is None
    classes=1,                 # define number of output labels
)

model = smp.Unet('efficientnet-b0', encoder_weights='imagenet', classes=1, aux_params=aux_params)
dir(model.classification_head)
model.classification_head

def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))

batch = np.load(IMGFILES[1])['arr_0']
batch = torch.tensor(np.float32(batch/255)) - 0.5
d0,d1,d2,d3 = batch.shape
batch = batch.reshape(d0, d3, d1, d2)
%time out = model(batch[:8])

torch.nn.MSELoss()
y = np.load(MAPFILES[1])['arr_0']
y = maskit(y, ksize = 3, scale = 8)
y = torch.tensor(np.float32(y)) / 255
y.shape


criterion = torch.nn.L1Loss()

criterion(y[:8], out[0].squeeze(1))



INPATH = '/Users/dhanley2/Documents/Personal/dfake'
METAFILE = os.path.join(INPATH, 'data', 'trainmeta.csv.gz')
metadf = pd.read_csv(METAFILE)
MAPFILES = sorted(glob.glob(os.path.join( INPATH, 'data/map/map*' )))
IMGFILES = sorted([i for i in glob.glob(os.path.join( INPATH, 'data/map/*.npz' )) if 'map__' not in i] )

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


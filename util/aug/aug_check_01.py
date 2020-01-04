# https://github.com/biubug6/Pytorch_Retinaface

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

from tqdm import tqdm
import skvideo.io
import skvideo.datasets
import matplotlib.pylab as plt
import albumentations as A

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')
INPATH = '/home/darragh/dfake'
INPATH = '/Users/dhanley2/Documents/Personal/dfake'
sys.path.append(INPATH)
from utils.sort import *
from utils.logs import get_logger
from utils.utils import dumpobj, loadobj
from utils.utils import cfg_re50, cfg_mnet, decode_landm, decode, PriorBox
from utils.utils import py_cpu_nms, load_fd_model, remove_prefix
from utils.retinaface import RetinaFace



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
    return imgs, vnframes, vh, vw, vfps

VNAMES = glob.glob(os.path.join(INPATH, 'data/train_sample_videos/*'))
img = vid2imgls(VNAMES[7], FPS=8)[0][0]
H,W,C= img.shape
im = Image.fromarray(img)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im.size


trn_transforms = Compose([
    HorizontalFlip(p=0.5),
    RandomContrast(p=0.3),
    RandomBrightness(p=0.3),
    JpegCompression(quality_lower=50, quality_upper=80, p=0.5),
    HueSaturationValue(p=0.3),
    Blur(blur_limit=30, p=0.3),
    MultiplicativeNoise(multiplier=1.5, p=0.3),
    IAAAdditiveGaussianNoise(p=0.2),
    ])

# ['GaussNoise', 'ISONoise', 'MultiplicativeNoise']
# ['Blur', 'GaussianBlur', 'MedianBlur', 'MotionBlur']
# ['ChannelDropout', 'CoarseDropout']
# ['Downscale', 'RandomScale', ]
# ['BasicTransform','DualTransform','ElasticTransform','ImageOnlyTransform']

def snglaugfn():
    rot = random.randrange(-10, 10)
    dim1 = random.uniform(0.7, 1.0)
    dim2 = random.randrange(SIZE//3, SIZE)
    return Compose([
        ShiftScaleRotate(p=0.5, rotate_limit=(rot,rot)),
        CenterCrop(int(SIZE*dim1), int(SIZE*dim1), always_apply=False, p=0.5), 
        Resize(dim2, dim2, interpolation=1,  p=0.5),
        Resize(SIZE, SIZE, interpolation=1,  p=1),
        ])
    
w,h=im.size
aug = A.Compose([
    A.Resize(h//2, w//2, interpolation=1,  p=1),
    A.Resize(h, w, interpolation=1,  p=1),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

    
'''
Noise - one off - proba 0.5
'''
aug = A.Compose([
    A.GaussNoise(var_limit=(100.0, 600.0), p=1.0),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

aug = A.Compose([
    A.ISONoise(color_shift=(0.2, 0.25), intensity=(0.2, 0.25), p=1.0),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024]*2,Image.ANTIALIAS)
im

aug = A.Compose([
    A.MultiplicativeNoise(multiplier=[0.7, 1.6], elementwise=False, per_channel=False, p=1.0),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024]*2,Image.ANTIALIAS)
im

'''
Blur - one of; proba 0.5
'''

aug = A.Compose([
    A.Blur(blur_limit=15, p=1.0), 
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024]*2,Image.ANTIALIAS)
im


aug = A.Compose([
    A.GaussianBlur(blur_limit=15, p=1.0), 
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024]*2,Image.ANTIALIAS)
im

aug = A.Compose([
    A.MotionBlur(blur_limit=(15), p=1.0), 
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024]*2,Image.ANTIALIAS)
im

aug = A.Compose([
    A.MedianBlur(blur_limit=10, p=1.0), 
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024]*2,Image.ANTIALIAS)
im
        

'''
Trasnforms - this could be confused for a kind of deepfake
'''


'''
Dropout - 0.2 proba
'''
aug = A.Compose([
    A.CoarseDropout(max_holes=50, max_height=20, max_width=20, min_height=6, min_width=6, p=1),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

aug = A.Compose([
    A.Cutout(num_holes=12, max_h_size=24, max_w_size=24, fill_value=255, p=1),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

'''
Downscale - 0.5 proba
'''
aug = A.Compose([
    A.Downscale(scale_min=0.3, scale_max=0.9, interpolation=0, always_apply=False, p=1.0),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im


'''
Brightness - 0.5 proba
'''

aug = A.Compose([
    A.RandomGamma(gamma_limit=(50, 150), p=1.0),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

aug = A.Compose([
    A.RandomBrightness(limit=0.4, p=1.0),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

aug = A.Compose([
    A.RandomContrast(limit=0.4, p=1.0),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

'''
JPEGCompression - 0.5 proba
'''
aug = A.Compose([
    A.JpegCompression(quality_lower=30, quality_upper=100, always_apply=False, p=1.0),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

aug = A.Compose([
    A.ImageCompression(quality_lower=30, quality_upper=100, always_apply=False, p=1.0),
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

'''
Weather - 0.1 proba
'''

aug = A.Compose([
    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=1.0)
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

aug = A.Compose([
    A.RandomShadow( p=1.0)
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im

aug = A.Compose([
    A.CLAHE(clip_limit=2.0, p=1.0)
])
imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im


'''
All together
'''
p1 = 0.1
aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.Downscale(scale_min=0.3, scale_max=0.9, interpolation=0, always_apply=False, p=0.5),
            ]),
        A.OneOf([
            A.GaussNoise(var_limit=(100.0, 600.0), p=p1),
            A.ISONoise(color_shift=(0.2, 0.25), intensity=(0.2, 0.25), p=p1),
            A.MultiplicativeNoise(multiplier=[0.7, 1.6], elementwise=False, per_channel=False, p=p1),
            A.NoOp(p=p2*3),
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
             A.NoOp(p=p2*3),
            ]),
        A.OneOf([
             A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=p1),
             A.RandomShadow( p=p1),
             A.NoOp(p=p1*8),
            ]),
        A.OneOf([
            A.CoarseDropout(max_holes=50, max_height=20, max_width=20, min_height=6, min_width=6, p=p1),
            A.Cutout(num_holes=12, max_h_size=24, max_w_size=24, fill_value=255, p=p1),
            A.CLAHE(clip_limit=2.0, p=p1),
            A.NoOp(p=p1*8),
            ]),
    ])

imaug = aug(image=img)['image']  
im = Image.fromarray(imaug)
im.thumbnail([1024,1024],Image.ANTIALIAS)
im


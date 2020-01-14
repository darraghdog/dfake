import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import glob
import json
import cv2
from PIL import Image
import numpy as np
import pandas as pd
#import dlib
import torch
import torchvision
from torchvision import models
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from itertools import product
from time import time
import datetime

from tqdm import tqdm
import skvideo.io
import skvideo.datasets
import matplotlib.pylab as plt
from albumentations.pytorch import ToTensor
import albumentations as A


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')
INPATH = '/home/darragh/dfake'
INPATH = '/Users/dhanley2/Documents/Personal/dfake'
sys.path.append(INPATH)
from utils.sort import *
from utils.logs import get_logger
from utils.utils import dumpobj, loadobj,chunks, pilimg, SpatialDropout, GradualWarmupScheduler

from utils.utils import cfg_re50, cfg_mnet, decode_landm, decode, PriorBox
from utils.utils import py_cpu_nms, load_fd_model, remove_prefix
from utils.retinaface import RetinaFace
from utils.sppnet import SPPNet
from utils.temporal_shift import TSN

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

VNAMES = os.listdir(os.path.join(INPATH, 'data/train_sample_videos'))
logdf = pd.concat([pd.read_csv(os.path.join(INPATH, 'data/meta/log_fold{}.txt'.format(i))) \
        for i in range(5)])
faills = [i for i in logdf.query('status == "fail"').video.tolist() if i in set(VNAMES)]

MODPATH = '/Users/dhanley2/Downloads/Resnet50_Final.pth'
cfg = cfg_re50
MNPATH = '/Users/dhanley2/Downloads/mobilenetV1X0.25_pretrain.tar'
MODPATH = '/Users/dhanley2/Downloads/mobilenet0.25_Final.pth'
cfg = cfg_mnet
cfg['confidence'] = 0.95
cfg['nms_threshold'] = 0.4

net = RetinaFace(cfg, phase = 'test', mnpath = MNPATH)
net = load_fd_model(net, MODPATH, True)
device = torch.device("cpu")
net = net.to(device)
net.eval()

VNAMES = ['cdaxixbosp.mp4', 'btiysiskpf.mp4', 'clihsshdkq.mp4']
for VNAME in VNAMES[:1]: #faills:
    fname = os.path.join(INPATH, 'data/train_sample_videos/{}'.format(VNAME))
    imgls, vnframes, im_height, im_width, vfps = vid2imgls(fname, 8)
    

mean_img = [0.4258249 , 0.31385377, 0.29170314]
std_img = [0.22613944, 0.1965406 , 0.18660679]

trn_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomContrast(p=0.3),
    A.RandomBrightness(p=0.3),
    A.JpegCompression(quality_lower=20, quality_upper=100, p=1.0),
    A.HueSaturationValue(p=0.3),
    A.Blur(blur_limit=30, p=0.3),
    A.ToGray(p=0.05),
    A.ToSepia(p=0.05),
    A.MultiplicativeNoise(multiplier=1.5, p=0.3),
    A.IAAAdditiveGaussianNoise(p=0.2),
    ])
val_transforms = A.Compose([
    A.NoOp(), 
    ])
transform_norm = A.Compose([
    A.Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
    ])

'''
python main.py kinetics RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 128 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb
     
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
'''
npfiles = glob.glob(os.path.join(INPATH, 'data/npimg/*'))
frames = np.load(npfiles[0])['arr_0']
Image.fromarray(frames[0])
d0,d1,d2,d3 = frames.shape
frames = np.stack([transform_norm(image=f)['image'] for f in frames])
frames = frames.reshape(d0,d1,d2,d3)
input_var = torch.tensor(frames[:32]).unsqueeze(0)
input_var = input_var.permute(0, 1, 4, 2, 3)
input_var.shape
n_segments=16

for bb in [50]:    
    mod = ResNet(bb, num_class=2, pretrained=True, folder=folder)
model = TSN(num_class=1, num_segments=n_segments, modality='RGB', \
            base_model='resnet50', img_feature_dim=224, pretrain='imagenet')
'''
folder='/Users/dhanley2/Documents/Personal/dfake/weights'
output_model_file = '{}/tsnresnet{}.pth'.format(folder, '50')
torch.save(model.state_dict(), output_model_file)
'''


%time out = model(torch.cat((input_var,input_var))[:,:n_segments])
out
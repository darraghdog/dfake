import os,sys,glob,json,cv2
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
import random, optparse,itertools, warnings
import matplotlib.pylab as plt
from dill.source import getname
from torch.utils.data import Dataset, DataLoader
INPATH='/share/dhanley2/dfake/'
sys.path.append(os.path.join(INPATH))
from utils.sort import *
from utils.logs import get_logger
from utils.utils import dumpobj, loadobj, chunks
from utils.utils import cfg_re50, cfg_mnet, decode_landm, decode, PriorBox
from utils.utils import py_cpu_nms, load_fd_model, remove_prefix
from utils.retinaface import RetinaFace
from sklearn.metrics.pairwise import cosine_similarity
from utils.face_embedding import InceptionResnetV1
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

logger = get_logger('Embeddings :', 'INFO') 

def logshape(df, logger=logger):
    name =[x for x in globals() if globals()[x] is df][0]
    logger.info('{} shape : {} {}'.format(name, *metadf.shape))


BSIZE=64
NPPATH = os.path.join(INPATH, 'data/npimg08')
device=torch.device('cuda')

metadf = pd.read_csv(os.path.join(INPATH, 'data/trainmeta.csv.gz'))
logshape(metadf)
         
logger.info('limit to original videos')
metadf = metadf[metadf['video']==metadf['original']]
logshape(metadf)

logger.info('limit to files on disk')
npfiles = os.listdir(NPPATH)
idx = metadf.video.str.replace('.mp4', '.npz').isin(npfiles)
metadf = metadf[idx].reset_index(drop=True)
logshape(metadf)


logger.info('get the different obejcts per video')
trackfiles = glob.glob(os.path.join(INPATH, 'data/npimg08/tracker_fold*'))
trackdf = [pd.read_csv(i) for i in trackfiles]
trackdf = [d.loc[~d[['obj', 'video']].duplicated()] for d in trackdf]
trackdf = pd.concat(trackdf)
trackdf = trackdf[trackdf.video.isin(metadf.video)].reset_index(drop=True)
trackdf = trackdf[['video', 'obj', 'frame']]
logshape(trackdf)


logger.info('Load model')
resnet = InceptionResnetV1(num_classes=2, device=device)
input_model_file = os.path.join(INPATH, 'weights/vggfacenet.bin')
resnet.load_state_dict(torch.load(input_model_file))
resnet.to(device)
resnet.eval()

logger.info('Load Numpy arrays and get embeddings')
batch = []
embls = []
idxls = []
DIM = trackdf.index.max()
for t, row in trackdf.iterrows():
    video, obj, frame = row
    try:
        frames = np.load(os.path.join(NPPATH, video.replace('.mp4','.npz')))['arr_0']
        batch .append( frames[frame])
    except:
        continue
    idxls.append(t)
    if (t%BSIZE==BSIZE-1) or t == DIM:
        batch = (np.array(batch)- 127.5) * 0.0078125
        batch = torch.tensor(batch).permute(0, 3,1,2).to(dtype=torch.float32)
        emb = resnet(batch)
        embls.append(emb.detach().cpu().numpy())        
        batch = []
 
logger.info('Concat and write out')
allemb = np.concatenate(embls)
allemb = pd.DataFrame(allemb, index = idxls, columns = ['emb'+str(x) for x in range(512)])
outdf = pd.concat([trackdf, allemb], 1).dropna(0)
outdf.to_csv(os.path.join(INPATH, 'data/face_embeddings.csv.gz'), compression = 'gzip', index = False) 
logshape(outdf)

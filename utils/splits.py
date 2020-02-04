import json
import os
import pickle
from collections import defaultdict
from glob import glob

import cv2
import numpy as np
from tqdm.auto import tqdm

def load_folds(METADATA_PATH):
    """Load cross-validation folder names"""
    with open(f'{METADATA_PATH}/FOLDS.pickle', 'rb') as f:
        folds = pickle.load(f)
    return folds


def get_cluster_data(METADATA_PATH='MY_DATA/CLUSTER_DATA_v6'):
    """Return dicts with metadata :

    REAL_VIDEO_TO_CLUSTER  - real video_name -> cluster idx
    FAKE_VIDEO_TO_CLUSTER  - fake video_name -> cluster idx
    CLUSTER_TO_REAL_VIDEOS - cluster -> list of real videos in cluster
    CLUSTER_TO_FAKE_VIDEOS - cluster -> list of fake videos in cluster
    CLUSTER_TO_DESCRIPTOR  - cluster -> precomouted average (median descriptor) 
    """
    with open(f'{METADATA_PATH}/REAL_VIDEO_TO_CLUSTER.pickle', 'rb') as f:
        REAL_VIDEO_TO_CLUSTER = pickle.load(f)
    with open(f'{METADATA_PATH}/FAKE_VIDEO_TO_CLUSTER.pickle', 'rb') as f:
        FAKE_VIDEO_TO_CLUSTER = pickle.load(f)
    with open(f'{METADATA_PATH}/CLUSTER_TO_REAL_VIDEOS.pickle', 'rb') as f:
        CLUSTER_TO_REAL_VIDEOS = pickle.load(f)
    with open(f'{METADATA_PATH}/CLUSTER_TO_FAKE_VIDEOS.pickle', 'rb') as f:
        CLUSTER_TO_FAKE_VIDEOS = pickle.load(f)
    with open(f'{METADATA_PATH}/CLUSTER_TO_DESCRIPTOR.pickle', 'rb') as f:
        CLUSTER_TO_DESCRIPTOR = pickle.load(f)
    return REAL_VIDEO_TO_CLUSTER, FAKE_VIDEO_TO_CLUSTER, CLUSTER_TO_REAL_VIDEOS, CLUSTER_TO_FAKE_VIDEOS, CLUSTER_TO_DESCRIPTOR


def get_real_to_fake_dict_v2(root_data_path='MY_DATA/CLUSTER_DATA_v6', is_full=False):
    """Load pickled metadata : real, fake video names and mapping real -> fake names corresponding to real"""
    if not is_full:
        with open(f'{root_data_path}/real_set.pickle', 'rb') as f:
            real_set = pickle.load(f)

        with open(f'{root_data_path}/fake_set.pickle', 'rb') as f:
            fake_set = pickle.load(f)

        with open(f'{root_data_path}/real_to_fake.pickle', 'rb') as f:
            real_to_fake = pickle.load(f)

    else:
        with open(f'{root_data_path}/real_set_full.pickle', 'rb') as f:
            real_set = pickle.load(f)

        with open(f'{root_data_path}/fake_set_full.pickle', 'rb') as f:
            fake_set = pickle.load(f)

        with open(f'{root_data_path}/real_to_fake_full.pickle', 'rb') as f:
            real_to_fake = pickle.load(f)

    return real_set, fake_set, real_to_fake


def get_real_to_fake_dict_orig(root_data_path='data', with_extension=True):
    """Load original metadata from json files provided by organizers"""
    metadata_pathes = [f'{root_data_path}/dfdc_train_part_{i}/metadata.json' for i in range(50)]

    real_list, fake_list = [], []
    real_to_fake = defaultdict(list)
    fake_to_real = {}

    for meta_path in tqdm(metadata_pathes):

        with open(meta_path, 'r') as f:
            meta = json.load(f)
            for video_name in meta:
                if meta[video_name]['label'] == 'REAL':
                    real_list.append(video_name if with_extension else video_name[:-4])
                else:
                    fake_to_real[video_name] = meta[video_name]['original']
                    v = meta[video_name]['original']
                    real_to_fake[v if with_extension else v[:-4]].append(
                        video_name if with_extension else video_name[:-4])

    for key in real_to_fake:
        fake_list.extend(real_to_fake[key])

    real_set = set(real_list)
    fake_set = set(fake_list)
    return real_set, fake_set, real_to_fake


def get_name_to_path(root_folder='data'):
    """Return video name to corresponding pathes"""
    video_pathes = glob(f'{root_folder}/*/*.mp4')
    video_name_to_path = {path.split('/')[-1][:-4]: path for path in video_pathes}
    return video_name_to_path


def get_snapshots_by_name(video_name_to_path,
                          video_name,
                          snapshot_path='aligned_data_mtcnn_112',
                          folder_postfix='_cropped_faces',
                          num_frames=None,
                          skip_non_first=False,
                          return_pathes=False):
    """Read&return frames and their names
    :num_frames - max num frames to return
    :skip_non_first - skip all frames after first
    """
    folder = video_name_to_path[video_name].split('/')[-2]
    snapshots_folder = f'{snapshot_path}/{folder}{folder_postfix}/{video_name}'
    images = []
    pathes = []
    for snapshot_path in sorted(glob(f'{snapshots_folder}/*'), key=lambda x: int(x.split('/')[-1][:-4])):
        img = cv2.imread(snapshot_path)
        if skip_non_first and '_0.jpg' not in snapshot_path:
            continue
        images.append(img)
        pathes.append(snapshot_path.split('/')[-1])
        if num_frames is not None and len(images) == num_frames:
            break
    if not return_pathes:
        return images
    return images, pathes


def count_real_fakes_by_clust_id(lst, CLUSTER_TO_REAL_VIDEOS, CLUSTER_TO_FAKE_VIDEOS):
    """Return overal number of fake and real videos per cluster"""
    real_count = 0
    fake_count = 0
    for t in lst:
        real_count += len(CLUSTER_TO_REAL_VIDEOS[t])
        fake_count += len(CLUSTER_TO_FAKE_VIDEOS[t])
    return real_count, fake_count

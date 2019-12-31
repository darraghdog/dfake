import os
import glob
import pandas as pd

INPATH = '/Users/dhanley2/Documents/Personal/dfake/data'
JSONFILES = glob.glob(os.path.join(INPATH, 'meta/train/dfdc*/*.json'))

def metaload(fname):
    df = pd.read_json(fname).T
    df['folder'] = fname.split('/')[-2]
    return df

# Load clusters - (sh run_face_embout.sh)
clust =  pd.read_csv(os.path.join(INPATH, 'face_clusters.csv.gz'))
clust = clust[['video', 'cluster']].drop_duplicates().reset_index(drop=True)
clust.columns = ['original', 'cluster']
clust.video.value_counts()

# Load data frames
metadf = pd.concat([metaload(i) for i in JSONFILES ], 0)
metadf = metadf.rename_axis('video').reset_index()
metadf['original'] = metadf['original'].fillna(metadf['video'])
metadf = metadf.merge(clust, on ='original', how = 'left')
metadf.cluster = metadf.cluster.fillna(metadf.cluster.max()+1)
metadf.cluster.value_counts()

# Peek at data
metadf['label'].value_counts()
metadf.head(30)

# Assign a fold to each original video
folddf = pd.DataFrame({'cluster' : metadf.cluster.value_counts().index})
folddf['fold'] = folddf.index.values%5
metadf = metadf.merge(folddf, on = 'cluster')
metadf.fold.value_counts()

# Write to disk
metadf.to_csv(os.path.join(INPATH, 'trainmeta.csv.gz'), index = False, compression = 'gzip')

# Peek at data
pd.crosstab(metadf.label, metadf.fold)

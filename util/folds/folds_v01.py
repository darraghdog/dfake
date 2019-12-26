import os
import glob
import pandas as pd

INPATH = '/Users/dhanley2/Documents/Personal/dfake/data'
JSONFILES = glob.glob(os.path.join(INPATH, 'meta/train/dfdc*/*.json'))

def metaload(fname):
    df = pd.read_json(fname).T
    df['folder'] = fname.split('/')[-2]
    return df

# Load data frames
metadf = pd.concat([metaload(i) for i in JSONFILES ], 0)
metadf = metadf.rename_axis('video').reset_index()
metadf['original'] = metadf['original'].fillna(metadf['video'])

# Peek at data
metadf['label'].value_counts()
metadf.head(30)

# Assign a fold to each original video
folddf = pd.DataFrame({'original' : metadf.original.unique()})
folddf['fold'] = folddf.index.values%5
metadf = metadf.merge(folddf, on = 'original')

# Write to disk
metadf.to_csv(os.path.join(INPATH, 'trainmeta.csv.gz'), index = False, compression = 'gzip')

# Peek at data
pd.crosstab(metadf.label, metadf.fold)

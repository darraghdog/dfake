N_GPU=1
WDIR='sppal04'
FOLD=0
SIZE='224'
ACCUM=1
BORDER=10
LR=0.00001



for LR in  0.00001
do
    bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -m dbslp1897  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/scripts/$WDIR  && python3 sppal.py  \
            --wtspath weights  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --metafile trainmeta.csv.gz  \
            --accum $ACCUM --imgpath data/mount/deepfake/prepared_data_v4/cropped_faces_raw/  --size $SIZE --batchsize 32 --lr $LR \
            --border $BORDER --start 0 --epochs 12"
done

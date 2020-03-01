N_GPU=2
WDIR='unet01'
FOLD=0
SIZE='224'
ACCUM=1
BSIZE=16
LR=0.00005

for FOLD in 0 # 1 2 
do
    bsub  -q low2  -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake8 \
            -m dbslp1828  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c \
            "cd /share/dhanley2/dfake/scripts/$WDIR && python3 unet.py  \
            --wtspath weights  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --metafile trainmeta.csv.gz  \
            --accum $ACCUM --imgpath data/mount/npimg15 --size $SIZE --batchsize $BSIZE --lr $LR \
            --arch 'efficientnet-b0' --skip 6 --start 0 --epochs 20"
done

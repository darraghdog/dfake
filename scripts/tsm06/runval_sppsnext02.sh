N_GPU=1
WDIR='tsm06'
FOLD=0
SIZE='224'
ACCUM=1
BSIZE=4
LR=0.00005

for FOLD in 0 1 2 3 4 #  0.00001 #  0.0001 0.00001
do
    bsub  -q normal -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -m dbslp1897 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/scripts/$WDIR  && python3 sppsnext.py  \
            --wtspath weights  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --metafile trainmeta.csv.gz  \
            --accum $ACCUM --imgpath data/mount/npimg08 --size $SIZE --batchsize $BSIZE --lr $LR \
            --infer VAL --maxlen 40 --nsegment 10 --start 0 --epochs 20 --skip 4"
done

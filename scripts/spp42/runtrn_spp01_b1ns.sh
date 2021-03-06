N_GPU=1
WDIR='spp42'
FOLD=0
SIZE='224'
ACCUM=1
BSIZE=4
LR=0.00005
REAL=4
FAKE=1

for FOLD in 0 # 3 4 # 0 1 2
do
    bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake5 \
            -m dbslp1897  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/scripts/$WDIR && python3 spp.py \
            --wtspath weights  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --metafile trainmeta.csv.gz  \
            --accum $ACCUM --imgpath data/mount/npimg15  --size $SIZE --batchsize $BSIZE --lr $LR \
            --skip 4 --start 0 --epochs 20 --arch tf_efficientnet_b1_ns \
            --origsamp $FAKE --fakesamp $FAKE --realsamp $REAL "
done

N_GPU=1
WDIR='spp29'
FOLD=0
SIZE='224'
ACCUM=1
BSIZE=4
LR=0.00005

for FOLD in 0 # 1 2 
do
    bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake4 \
            -m dbslp1827  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/scripts/$WDIR && pip install --root .  timm  && python3 spp.py  \
            --wtspath weights  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --metafile trainmeta.csv.gz  \
            --accum $ACCUM --imgpath data/mount/npimg08 --size $SIZE --batchsize $BSIZE --lr $LR \
            --skip 4 --start 0 --epochs 20"
done

N_GPU=1
WDIR='spp12'
FOLD=0
SIZE='224'
ACCUM=1

for LR in 0.00001  #  0.0001 0.00001
do
    bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -m dbslp1829 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/scripts/$WDIR  && python3 spp.py  \
            --wtspath weights  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --metafile trainmeta.csv.gz  \
            --infer VAL --accum $ACCUM --lrgamma 0.8 --start 0 --imgpath data/mount/npimg08 --size $SIZE --batchsize 32 --lr $LR \
            --start 8 --epochs 15"
done

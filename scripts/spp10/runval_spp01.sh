N_GPU=1
WDIR='spp10'
FOLD=0
SIZE='224'
ACCUM=1

for LR in 0.00001
do
    bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -m dbslp1827 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/scripts/$WDIR  && python3 spp.py  \
            --wtspath weights  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --metafile trainmeta.csv.gz  \
            --accum $ACCUM --lrgamma 0.8 --start 0 --imgpath 'data/mount/npimg08'  \
            --infer VAL --batchsize 16 --lr $LR  --size $SIZE --start 0 --epochs 8"
done

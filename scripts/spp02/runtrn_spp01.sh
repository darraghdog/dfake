N_GPU=1
WDIR='spp02'
FOLD=0
SIZE='224'

for ACCUM in  1 # 4 # 1
do
    bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -m dbslp1897 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/scripts/$WDIR  && python3 spp.py  \
            --wtspath weights  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --metafile trainmeta.csv.gz --lr 0.000005 --lrmult 5  \
            --accum $ACCUM --start 0 --imgpath data/mount/npimg --size $SIZE --batchsize 16 --epochs 8"
done

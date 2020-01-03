N_GPU=1
WDIR='spp04'
FOLD=0
SIZE='224'

for ACCUM in 1 # 4
do
    bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -m dbslp1897 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/scripts/$WDIR  && python3 sppchk.py  \
            --infer VAL --start 1 --wtspath weights  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --metafile trainmeta.csv.gz  \
            --accum $ACCUM --lrgamma 0.8 --start 0 --imgpath data/mount/npimg10 --size $SIZE --batchsize 32 --epochs 5 --lr 0.00001"
done

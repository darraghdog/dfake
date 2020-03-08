N_GPU=1
WDIR='saver15'
FOLD=0
SIZE='224'

for FOLD in 0 1 # 2 3 4 # 0 1 2 3 
do
    bsub  -q normal -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake4 \
            -m dbslp1827 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/util/$WDIR  && python3 vid_saver.py  \
            --batchsize 16 --fold $FOLD  --rootpath /share/dhanley2/dfake/ --vidpath data/mount/train --metafile trainmeta.csv.gz  \
            --loadseconds 12 --imgpath data/mount/npimg15 --size $SIZE --wtspath weights --fps 8"
done

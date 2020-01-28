N_GPU=1
WDIR='saver13'
FOLD=0
SIZE='224'
BSIZE=4

for FOLD in  4 # 0 2 3 #  1
do
    bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -m dbslp1897 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/util/$WDIR  && python3 vid_saver.py  \
            --batchsize $BSIZE  --fold $FOLD  --rootpath /share/dhanley2/dfake/ --vidpath data/mount/train --metafile trainmeta.csv.gz  \
            --loadseconds 12 --imgpath data/mount/npimg13 --size $SIZE --wtspath weights --fps 8"
done

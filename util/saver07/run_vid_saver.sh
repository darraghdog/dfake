N_GPU=1
WDIR='saver07'
FOLD=0
SIZE='224'

for FOLD in 0 # 1 2 3 4
do
    bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfaketf \
            -m dbslp1897 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/util/$WDIR  && python3 vid_saver.py  \
            --fold $FOLD  --rootpath /share/dhanley2/dfake/ --vidpath data/mount/train --metafile trainmeta.csv.gz  \
            --imgpath data/mount/npimg07 --size $SIZE --startsize 640 --wtspath weights --fps 8"
done

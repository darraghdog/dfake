N_GPU=1
WDIR='saver05'
FOLD=0
SIZE='224'
bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/util/$WDIR  && python3 vid_saver.py  \
            --fold $FOLD  --rootpath /share/dhanley2/submit/dfake/ --vidpath mount/train/dfdc_train_part_3  \
            --imgpath mount/npimg --size $SIZE --wtspath weights --fps 8"

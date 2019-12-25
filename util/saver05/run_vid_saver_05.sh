N_GPU=1
WDIR='saver05'
FOLD=0
SIZE='224'
bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -m dbslp1829 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/util/$WDIR  && python3 trainorig.py  \
            --fold $FOLD  --workpath scripts/$WDIR  \
            --imgpath data/mount/512X512X6/ --size $SIZE --weightsname weights/model_512_resnext101$FOLD.bin"

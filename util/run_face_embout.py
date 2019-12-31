N_GPU=1
bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake \
            -m dbslp1897 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c \
            "cd /share/dhanley2/dfake/util/  && python3 face_embout.py"

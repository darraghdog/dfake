N_GPU=1
bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:dfake5 \
            -m dbslp1828  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/datasets/ && \
            python3 forensic_download.py /share/dhanley2/dfake/data/mount/datasets/ -c c23 --type videos -n 1000 --server EU"

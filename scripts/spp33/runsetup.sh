N_GPU=1
WDIR='spp33'
FOLD=0
SIZE='224'
ACCUM=1
BSIZE=4
LR=0.00005

for FOLD in 0 # 1 2 
do
    bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=kaggle/python:latest \
            -m dbslp1896  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/dfake/scripts && \
            export PYTHONPATH='${PYTHONPATH}:/opt/conda/lib/python3.6/site-packages'  && cd pim && python setup.py -d .  install"
done

N_GPU=1
WDIR='spp26'

bsub  -q low2 -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=guestros/deepfake-docker:latest \
            -m dbslp1828  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "ls && python faceswap.py -h"
# "cd /share/dhanley2/dfake/scripts/$WDIR  && python3 spp.py  \

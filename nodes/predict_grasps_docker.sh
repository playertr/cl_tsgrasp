#!/bin/bash

# Run grasp prediction in a Docker image with predict_grasps.py bind-mounted in.
# Note that this script can be invoked with `rosrun cl_tsgrasp 
# predict_grasps_docker.sh`, but passing ROS arguments to the script has no
# effect.

# this script expects the following folder path:
# . (cl_tsgrasp)
# └── nodes
#     ├──predict_grasps.py
#     └── predict_grasps_docker.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run -it \
    --gpus all \
    --network host \
    -v $SCRIPT_DIR/../nodes/predict_grasps.py:/cl_tsgrasp/cl_tsgrasp_ws/src/cl_tsgrasp/nodes/predict_grasps.py \
    playertr/cl_tsgrasp
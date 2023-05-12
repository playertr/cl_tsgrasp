#!/bin/bash

# Script to run grasp inference in a Docker container to simplify setup.
# Invoke this script from the root of a ROS workspace containing src/cl_tsgrasp.

# Optionally, first build playertr/grasp with ./docker/build.sh.
# Or, pull it from Docker Hub

docker run -it \
    -v $(pwd):$(pwd)  \
    -e PYTHONPATH \
    --gpus all \
    --network host \
    --workdir $(pwd)/src/cl_tsgrasp \
    playertr/cl_tsgrasp \
    /cl_tsgrasp/miniconda3/envs/tsgrasp/bin/python -m nodes.predict_grasps "$@"

# bind-mount the workspace containing cl_tsgrasp
# share the location of the Python message files
# enable GPU inference
# share networking for ROS transport
# start the container from this repo
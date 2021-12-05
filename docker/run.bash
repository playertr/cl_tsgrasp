#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <docker image> <cmd (optional)>"
    echo "Example: $0 playertr/clgrasp bash"
    exit 1
fi

IMG=$1
ARGS=("$@")

rocker \
    --env IGN_IP=127.0.0.1 \
    --vol $(pwd):$(pwd) \
    --nvidia \
    --x11 \
    --network host \
    $IMG \
    ${@:2}
    # IGN_IP: needed due to bug in ign_transport
    # https://github.com/AndrejOrsula/drl_grasping/issues/90
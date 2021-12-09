#!/bin/sh -e
# start the xpra server
# source this script if you need DISPLAY set `. start-xpra.sh`
# $XPRASTART is intended for extra setting like `--dpi=108`

export DISPLAY=:100
echo "Set DISPLAY to $DISPLAY."
export XPRAPORT=6100
echo "Set XPRAPORT to $XPRAPORT."

echo "Starting xpra with given DISPLAY=$DISPLAY and on port $XPRAPORT."
xpra start $DISPLAY --bind-tcp=0.0.0.0:$XPRAPORT --pulseaudio=no --notifications=no --quality=99

# cl_tsgrasp

This is a demo of grasping using tsgrasp.

# Installation

1. `git submodule update --init --recursive`
1. Download the model parameter checkpoint.
1. Install dependencies of tsgrasp into a conda environment.
1. Export the conda environment path in your ~/.bashrc: `export NN_CONDA_PATH=/home/playert/miniconda3/envs/tsgrasp/bin/python`
1. `catkin build`

# Usage

## Simulated
`roslaunch cl_tsgrasp testbed_grasp_demo.launch simulated:=true`

## Real
`roslaunch cl_tsgrasp testbed_grasp_demo.launch simulated:=false`
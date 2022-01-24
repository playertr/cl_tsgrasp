#! /usr/bin/bash

cd clgrasping_ws
source devel/setup.bash
# roslaunch cl_tsgrasp sim.launch x:=-0.5 y:=-0.075 \
# 	world:=$(rospack find franka_gazebo)/world/stone.sdf \
# 	controller:=cartesian_impedance_example_controller \
# 	rviz:=true

roslaunch cl_tsgrasp sim_new.launch x:=-0.5 y:=-0.075 \
	world:=$(rospack find cl_tsgrasp)/worlds/no_stone.sdf \
	rviz:=true

	
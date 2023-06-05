# cl_tsgrasp

This is a ROS package to synthesize and execute grasps on the RSA Testbed using tsgrasp and MoveIt.

[paper](https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/icra23_0859_fi_0.pdf) | [ICRA Video](https://youtu.be/E31_ryG2xGY) | [Presentation](https://www.youtube.com/watch?v=JBEWkCMrQKs)

[![Demo Video](https://img.youtube.com/vi/JBEWkCMrQKs/0.jpg)](https://youtu.be/JBEWkCMrQKs)

# Cite
```
@inproceedings{tsgrasp2023,
  title={Real-Time Generative Grasping with Spatio-temporal Sparse Convolution},
  author={Player, Timothy R and Chang, Dongsik and Fuxin, Li and Hollinger, Geoffrey A},
  year = {2023},
  booktitle = {Proc. IEEE International Conference on Robotics and Automation},
  address = {London, UK}
}
```

# About
This repo contains utilities for predicting grasps within a streaming point cloud (`predict_grasps.py`), moving the robot arm through a sequence of pre-grasp poses (`mover.py`), and controlling the grasping process with a graphical interface to a finite state machine (`gui_fsm.py`).

Our documentation focuses on the **grasp prediction** pipeline. This pipeline accepts as input a sensor_msgs/PointCloud2 on the topic `/point_cloud`, and outputs a cl_tsgrasp/Grasps message on the topic `/tsgrasp/grasps_unfiltered`. In our implementation, it must be possible to transform the point cloud into the `world` frame. The contents of a Grasps message are listed below.

```
# Grasps.msg

Header header # sequence number, timestamp,frame ID
geometry_msgs/Pose[] poses # N grasp poses expressed in this frame
geometry_msgs/Pose[] orbital_poses # N orbital_poses, only published occasionally
float32[] confs # N corresponding grasp confidences in [0, 1]
float32[] widths # N corresponding gripper widths in meters
```

# Installation and Usage

1. Install [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).
1. Install [ROS Noetic](http://wiki.ros.org/noetic).
1. Create a catkin workspace.
1. Clone this repo into the `src` directory of your workspace.
1. Install the dependencies using `rosdep install --from-paths src --ignore-src -r -y`
1. Build your catkin workspace.
1. Run `roslaunch cl_tsgrasp predict_grasps_node.launch`.
1. Start a camera node, bag file, or simulator that outputs a PointCloud2 onto the `/point_cloud` topic.

Running `predict_grasps_node.launch` downloads a container from Dockerhub and launches the grasp prediction node in Docker. This means you don't need to manually install difficult dependencies such as Minkowski Engine. If you would like to install these manually, see [docker/Dockerfile](docker/Dockerfile) to view these steps. 

# Switching Between Models
By default, the model performs single-frame inference using the pretrained weights from the Docker image. To switch to four-frame inference, complete the following steps.

1. Copy the configuration string `cfg_str` from `nn/ckpts/tsgrasp_scene_4_random_yaw/README.md` into `nn/load_model.py`. This tells the prediction script where to find the new model weights and how to configure the model.
1. Change `QUEUE_LEN` in `predict_grasps.py` to 4.
1. Run the grasp prediction node again. Your changes will be bind-mounted into the container when Docker is invoked in `nodes/predict_grasps_docker.sh`.

# Configuring the Grasp Prediction Pipeline
The function `find_grasps()` in `nodes/predict_grasps.py` contains several pre- and post-processing steps beyond model inference. In order, the preprocessing steps are:
- Remove points that fall outside a rectangular region in the camera frame.
- Remove points that fall outside a rectangular region in the camera frame.
- Randomly downsample the points with uniform probability.
- Remove points that have few neighbors.
- If multiple point clouds are processed at once, transform all points to the latest camera frame, as done during training.

After inferring the grasp poses, confidences, and grasp widths using TSGrasp, the postprocessing steps are:
- Filter the grasps to select high-confidence, spatially diverse grasps:
    - Remove all grasps whose softmax confidence is less than `CONF_THRESHOLD`.
    - Select the `100 * TOP_K` highest-confidence grasps.
    - Use furthest-point sampling to select the `TOP_K` most spatially diverse grasps from this set.
- Rotate grasps by 180 degrees around the gripper axis if their y-axis is facing downward. This prevented our gripper, which rotated very slowly, from continually rotating during consecutive trials.
- Translate the grasps along the gripper axis to accommodate the the length of the gripper.
- Publish the grasps
- Publish an additional PointCloud2 to visualize the grasp confidence of the point cloud. (***NB!:*** This is slow.)

The recommended way to enable, disable, or change the parameters of any step in this pipeline is to modify `predict_grasps.py`. When you run the grasp synthesis node using `predict_grasps_docker.sh`, the modified file will be used by the container.

# Conducting grasp trials
To perform grasp trials on a Reach Robotics Bravo arm, use in `testbed_grasp_demo.launch`. This launch file requires [software dependencies](https://gitlab.com/apl-ocean-engineering/raven_manipulation/bravo_arm_sw) for the Bravo arm.

If you want to do grasp trials on a different robot arm, many of the nodes (e.g., grasp prediction, motion planning, visualization, kinematic feasibility filtering...) can be modified to work with your MoveIt!-enabled robot. Refer to our reference implementation in `testbed_grasp_demo.launch` and happy hacking!
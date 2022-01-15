Launching the demo

```
cd clgrasping_ws
source devel/setup.bash
roslaunch cl_tsgrasp sim.launch x:=-0.5 y:=-0.075 \
    world:=$(rospack find franka_gazebo)/world/stone.sdf \
    controller:=cartesian_impedance_example_controller \
    rviz:=true
```

```
cd clgrasping_ws
source devel/setup.bash
conda activate tsgrasp
python src/cl_tsgrasp/nodes/predict_grasps.py
```

```
rosrun topic_tools transform /tsgrasp/grasps /tsgrasp/grasps_pose geometry_msgs/PoseArray 'geometry_msgs.msg.PoseArray(header=m.header, poses=m.poses)' --import geometry_msgs
```

Start the panda_simulator and its controllers

```
setup
roslaunch panda_gazebo panda_world.launch
```

```
setup
roslaunch panda_sim_moveit sim_move_group.launch
```

Switch controller to velocity

```
rosservice call /controller_manager/switch_controller "{start_controllers: ['panda_simulator/velocity_joint_velocity_controller'], stop_controllers: ['panda_simulator/effort_joint_torque_controller'], strictness: 2}"
```

Start servoing node
```
setup
roslaunch cl_tsgrasp moveit_servo.launch
```

Publish servo messages
```
setup
rostopic pub -r 100 -s /panda_simulator/servo_server/delta_twist_cmds geometry_msgs/TwistStamped "header: auto
twist:
  linear:
    x: 0.0
    y: 0.01
    z: -0.01
  angular:
    x: 0.0
    y: 0.0
    z: 0.0"
```

Start transformer node
```
rosrun topic_tools transform /joint_group_position_controller/command /panda_simulator/motion_controller/arm/joint_commands franka_core_msgs/JointCommand "franka_core_msgs.msg.JointCommand( \
    header=std_msgs.msg.Header(seq=0,stamp=rospy.Time.now(),frame_id='world'), \
    mode=1, \
    names=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',], \
    position=m.data, \
    velocity=[], \
    acceleration=[], \
    effort=[] \
    )" --import std_msgs rospy franka_core_msgs
```
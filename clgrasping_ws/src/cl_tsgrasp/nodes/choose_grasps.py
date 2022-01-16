#! /usr/bin/python
# Select the best grasp from the published Grasps message.

import rospy
from cl_tsgrasp.msg import Grasps
from geometry_msgs.msg import PoseStamped
from rospy.numpy_msg import numpy_msg
import numpy as np

from utils import TransformFrames

## Start node and publisher
rospy.init_node('grasp_chooser')
final_goal_pose_pub = rospy.Publisher(name='/tsgrasp/final_goal_pose', 
    data_class=numpy_msg(PoseStamped), queue_size=1)

tf = TransformFrames()

final_goal_pose = None

def publish_goal_callback(msg):
    """Identify the best grasp in the Grasps message and publish it."""
    confs = msg.confs
    poses = msg.poses

    ## Latch the goal pose -- do not update after initial update
    # global final_goal_pose
    # if final_goal_pose is not None:
    #     final_goal_pose_pub.publish(final_goal_pose)
    #     return

    # filter grasps (and confs) that are too low
    # note: this way of transforming to the world frame is slow
    world_poses = [tf.pose_transform(PoseStamped(header=msg.header, pose=pose), target_frame='world') for pose in poses]
    valid_idx = [world_poses[i].pose.position.z > 0.42 for i in range(len(world_poses))]
    poses = [poses[i] for i in range(len(poses)) if valid_idx[i]]
    confs = [confs[i] for i in range(len(confs)) if valid_idx[i]]

    best_grasp_idx = np.argmax(np.array(confs))
    try:
        pose = PoseStamped(header=msg.header, pose=poses[best_grasp_idx])
    except:
        pose = PoseStamped(header=msg.header, pose=msg.poses[0])

    final_goal_pose = tf.pose_transform(pose, target_frame='panda_link0')
    final_goal_pose_pub.publish(final_goal_pose)

grasp_sub = rospy.Subscriber(name='/tsgrasp/grasps', 
    data_class=Grasps, callback=publish_goal_callback, queue_size=1)

rospy.spin()
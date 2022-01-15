#! /usr/bin/env python
# Read in the current end effector goal pose from a FrankaState message
# and publish it as a normal PoseStamped
import rospy
from std_msgs.msg import Header
from franka_msgs.msg import FrankaState
from utils import TransformFrames
from geometry_msgs.msg import PoseStamped, Pose, Vector3
import quaternion
import numpy as np
import quaternion


rospy.init_node('publish_ee_pose')
ee_pose_pub = rospy.Publisher('/tsgrasp/ee_pose', PoseStamped, queue_size=1)
tf = TransformFrames()

def update_EE_state(msg):
    """
        Callback function to get current end-point state. From panda_simulator.
    """
    # pose message received is a vectorised column major transformation matrix
    global ee_pose
    cart_pose_trans_mat = np.asarray(msg.O_T_EE).reshape(4,4,order='F')

    pose = PoseStamped(
        header=Header(frame_id='panda_link0'),
        pose=Pose(
            position=Vector3(*cart_pose_trans_mat[:3,3]),
            orientation=quaternion.from_rotation_matrix(cart_pose_trans_mat[:3,:3])
        )
    )
    ee_pose = tf.pose_transform(pose, target_frame='panda_link0')
    ee_pose_pub.publish(ee_pose)

ee_pose_sub = rospy.Subscriber(
    name='panda/franka_state_controller/franka_states',
    data_class=FrankaState, callback=update_EE_state, queue_size=1)

rospy.loginfo('Ready to publish end effector pose.')
rospy.spin()
#! /usr/bin/env python3

import rospy
from utils import TFHelper
from geometry_msgs.msg import PoseStamped, Pose
import tf2_ros

rospy.init_node('publish_ee_pose')
ee_pose_pub = rospy.Publisher('/tsgrasp/ee_pose', PoseStamped, queue_size=1)
tf = TFHelper()

def publish_ee_pose():
    try:
        ee_tf = tf.get_transform(source_frame="ee_link", target_frame="world")
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return # no link yet

    ee_pose = PoseStamped(
        header = ee_tf.header,
        pose = Pose(
            position=ee_tf.transform.translation,
            orientation=ee_tf.transform.rotation
        )
    )

    ee_pose_pub.publish(ee_pose)

rospy.loginfo('Ready to publish end effector pose.')
r = rospy.Rate(50)
while not rospy.is_shutdown():
    publish_ee_pose()
    r.sleep()
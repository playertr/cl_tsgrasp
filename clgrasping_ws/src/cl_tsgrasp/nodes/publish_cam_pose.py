#! /usr/bin/env python3

import rospy
from utils import TransformFrames
from geometry_msgs.msg import PoseStamped, Pose
import tf2_ros


rospy.init_node('publish_cam_pose')
ee_pose_pub = rospy.Publisher('/tsgrasp/cam_pose', PoseStamped, queue_size=1)
tf = TransformFrames()

def publish_cam_pose():
    try:
        cam_tf = tf.get_transform(source_frame="kinect_optical_link", target_frame="world")
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return # no link yet
    
    cam_pose = PoseStamped(
        header = cam_tf.header,
        pose = Pose(
            position=cam_tf.transform.translation,
            orientation=cam_tf.transform.rotation
        )
    )

    ee_pose_pub.publish(cam_pose)

rospy.loginfo('Ready to publish camera pose.')
r = rospy.Rate(50)
while not rospy.is_shutdown():
    publish_cam_pose()
    r.sleep()
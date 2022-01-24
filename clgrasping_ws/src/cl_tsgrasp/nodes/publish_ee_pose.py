#! /usr/bin/env python3

import rospy
from utils import TransformFrames
from geometry_msgs.msg import PoseStamped, Pose


rospy.init_node('publish_ee_pose')
ee_pose_pub = rospy.Publisher('/tsgrasp/ee_pose', PoseStamped, queue_size=1)
tf = TransformFrames()

def publish_ee_pose():
    ee_tf = tf.get_transform(source_frame="panda_link8", target_frame="panda_link0")
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
#! /usr/bin/env python
# Read in the current end effector goal pose from a FrankaState message
# and publish it as a normal PoseStamped
import rospy
from std_msgs.msg import Header
from franka_msgs.msg import FrankaState
from utils import TransformFrames
from geometry_msgs.msg import PoseStamped, Pose, Vector3


rospy.init_node('publish_cam_pose')
ee_pose_pub = rospy.Publisher('/tsgrasp/cam_pose', PoseStamped, queue_size=1)
tf = TransformFrames()

def publish_cam_pose():
    cam_tf = tf.get_transform(source_frame="kinect_optical_link", target_frame="world")
    cam_pose = PoseStamped(
        header = cam_tf.header,
        pose = Pose(
            position=cam_tf.transform.translation,
            orientation=cam_tf.transform.rotation
        )
    )

    ee_pose_pub.publish(cam_pose)

rospy.loginfo('Ready to publish camera pose.')
r = rospy.Rate(100)
while not rospy.is_shutdown():
    publish_cam_pose()
    r.sleep()
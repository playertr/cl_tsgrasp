#! /usr/bin/python3
# Read in the end effector goal pose and publish an "orbital pose",
# offset from the final pose, that the robot arm should achieve first.

import rospy
from geometry_msgs.msg import PoseStamped, Pose, Vector3
import quaternion
import numpy as np

## constants
# TODO change to rosparam
ORBIT_RADIUS = 0.1 # distance to initially orbit object before terminal homing

orbital_pose_pub = rospy.Publisher('/tsgrasp/orbital_pose', PoseStamped, queue_size=1)

def orbital_pose(goal_pose):
    """Return a pose offset from the goal (end effector) pose along the z direction."""
    o = goal_pose.pose.orientation
    q = quaternion.from_float_array(np.array([o.w, o.x, o.y, o.z]))
    rot = quaternion.as_rotation_matrix(q)
    z_hat = rot[:3, 2]
    p = goal_pose.pose.position
    pos = np.array([p.x, p.y, p.z])
    pos -= z_hat * ORBIT_RADIUS
    pos = Vector3(x=pos[0],y=pos[1],z=pos[2])
    orbit_pose = PoseStamped(
        header=goal_pose.header,
        pose=Pose(position=pos, orientation=q)
    )
    return orbit_pose

def goal_pose_cb(msg):
    orbital_pose_pub.publish(orbital_pose(msg))

rospy.init_node('publish_orbital_pose')
rospy.Subscriber('/tsgrasp/final_goal_pose', PoseStamped, goal_pose_cb, queue_size=1)

rospy.loginfo('Ready to publish orbital pose.')
rospy.spin()
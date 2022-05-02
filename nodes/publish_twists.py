#! /usr/bin/python3

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import TwistStamped, Twist


rospy.init_node('publish_twists')
twist_pub = rospy.Publisher('/servo_server/delta_twist_cmds', TwistStamped, queue_size=1)

def publish_twist():
    h = Header()
    h.stamp = rospy.Time.now()
    h.frame_id = 'panda_link0'

    twist = Twist()
    twist.linear.x = 0.6
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0

    ts = TwistStamped(
        header = h,
        twist = twist
    )

    twist_pub.publish(ts)
    rospy.loginfo("Published twist.")

rospy.loginfo('Ready to publish twists.')\

r = rospy.Rate(100)
while not rospy.is_shutdown():
    rospy.loginfo("Publishing twist!")
    publish_twist()
    r.sleep()
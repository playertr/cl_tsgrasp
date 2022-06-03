#! /usr/bin/env python

# motion_server.py
# An actionlib server providing an Action to achieve a pose.
# This server uses the franka_ros example impedance controller, and is very
# dumb. It does not handle joint singularities or obstacles. An approach based
# on MoveIt! would be better.

import rospy
import actionlib
import cl_tsgrasp.msg
from geometry_msgs.msg import PoseStamped
from rospy.numpy_msg import numpy_msg
from utils import TFHelper, se3_dist

SUCCESS_RADIUS = 0.02 # in SE(3) space -- dimensionless

ee_pose = None

def ee_pose_cb(msg):
    global ee_pose
    ee_pose = msg

class MotionAction:
    # create messages that are used to publish feedback/result
    _feedback = cl_tsgrasp.msg.MotionFeedback()
    _result = cl_tsgrasp.msg.MotionResult()

    def __init__(self, name):
        self._action_name=name

        # pose topic that the controller is listening on
        self._goal_pose_pub = rospy.Publisher(name='/panda/cartesian_impedance_example_controller/equilibrium_pose', 
        data_class=numpy_msg(PoseStamped), queue_size=1)

        self._as = actionlib.SimpleActionServer(
            self._action_name,
            cl_tsgrasp.msg.MotionAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self._as.start()
        rospy.loginfo('%s: Server ready.' % self._action_name)
        
    def execute_cb(self, goal):
        global ee_pose

        rospy.loginfo("Starting to move end effector.")

        r = rospy.Rate(20)
        success = True

        while se3_dist(ee_pose.pose, goal.goal_pose.pose) > SUCCESS_RADIUS:

            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                success = False
                break
            
            self._goal_pose_pub.publish(goal.goal_pose)

            self._feedback.current_pose = ee_pose
            self._as.publish_feedback(self._feedback)
            r.sleep()

        if success:
            self._result.achieved_pose = ee_pose
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)

if __name__ == '__main__':
    rospy.init_node('motion_server')

    tf = TFHelper() # wrapper around a tf2_ros Buffer

    pose_sub = rospy.Subscriber( # subscribe to live end effector state
        name='/tsgrasp/ee_pose',
        data_class=PoseStamped, callback=ee_pose_cb, queue_size=1)

    server = MotionAction(rospy.get_name())
    rospy.spin()
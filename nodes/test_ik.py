#! /usr/bin/python3

#! /usr/bin/python3
# Interactive GUI for transitioning FSM states.

import rospy
from motion import Mover
import pickle

rospy.init_node('test_ik')

rospy.loginfo("Waiting for services.")
rospy.wait_for_service("/bravo/plan_kinematic_path") # wait for moveit to be up and running
rospy.wait_for_service('/bravo/apply_planning_scene') # just to make sure we can add to the planningscene in motion.py

# Start end-effector motion client
mover = Mover()

with open('/home/tim/Research/bravo_ws/outputs.pkl', 'rb') as f:
    outputs = pickle.load(f)

for pose, plan in outputs:
    print(pose)
    print(plan)

    input("Press enter to attempt this pose.")
    mover.go_ee_pose(pose)


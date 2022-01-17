#! /usr/bin/env python
# Execute finite state machine for grasping.

import rospy
import smach_ros
import smach
from franka_gripper.msg import MoveAction, GraspAction
import actionlib
import cl_tsgrasp.msg

# main
def main():
    rospy.init_node('smach_state_machine')

    # Start end-effector motion client
    motion_client = actionlib.SimpleActionClient('/motion_server', cl_tsgrasp.msg.MotionAction)
    motion_client.wait_for_server()

    # wait for gripper servers. Hacky and seems wrong.
    actionlib.SimpleActionClient('/panda/franka_gripper/move', MoveAction).wait_for_server()
    actionlib.SimpleActionClient('/panda/franka_gripper/grasp', GraspAction).wait_for_server()

    from states import OpenJaws, GoToOrbitalPose, TerminalHoming, CloseJaws, ResetPos

    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['SUCCESS', 'FAIL'])

    # Open the container
    with sm:

        # sub- state machine for grasping
        grasp_sm = smach.StateMachine(outcomes=['GRASP_SUCCESS', 'GRASP_FAIL'])
        with grasp_sm:
            # Add states to the container
            smach.StateMachine.add(
                'OPEN_GRIPPER',
                OpenJaws,
                transitions={
                    'succeeded': 'GO_TO_ORBITAL_POSE',
                    'aborted': 'GRASP_FAIL',
                    'preempted': 'OPEN_GRIPPER'
                }
            )
            smach.StateMachine.add(
                'GO_TO_ORBITAL_POSE',
                GoToOrbitalPose(motion_client),
                transitions={
                    'in_orbital_pose': 'TERMINAL_HOMING',
                    'not_in_orbital_pose': 'GO_TO_ORBITAL_POSE'
                }
            )
            smach.StateMachine.add(
                'TERMINAL_HOMING',
                TerminalHoming(),
                transitions={
                    'in_grasp_pose': 'CLOSE_GRIPPER',
                    'not_in_grasp_pose': 'GO_TO_ORBITAL_POSE'
                }
            )
            smach.StateMachine.add(
                'CLOSE_GRIPPER',
                CloseJaws,
                transitions={
                    'succeeded': 'RESET_POS',
                    'aborted': 'GRASP_FAIL',
                    'preempted': 'OPEN_GRIPPER'
                }
            )
            smach.StateMachine.add(
                'RESET_POS',
                ResetPos,
                transitions={
                    'succeeded': 'GRASP_SUCCESS',
                    'aborted': 'GRASP_FAIL',
                    'preempted': 'GO_TO_ORBITAL_POSE'
                }
            )
        
        smach.StateMachine.add('GRASP', grasp_sm, 
            transitions={
                'GRASP_SUCCESS':'SUCCESS',
                'GRASP_FAIL':'FAIL'
            }
        )



    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    # Execute the state machine
    outcome = sm.execute()

    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
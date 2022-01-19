#! /usr/bin/env python
# Execute finite state machine for grasping.

from ast import Del
import rospy
import smach_ros
import smach
import actionlib
import cl_tsgrasp.msg
from franka_gripper.msg import MoveAction, GraspAction

# main
def main():
    rospy.init_node('smach_state_machine')

    # wait for services called within States
    # (located here because State.__init__ can't block)
    rospy.loginfo("Waiting for services.")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    
    actionlib.SimpleActionClient('/panda/franka_gripper/move', MoveAction).wait_for_server()
    actionlib.SimpleActionClient('/panda/franka_gripper/grasp', GraspAction).wait_for_server()

    # Start end-effector motion client
    motion_client = actionlib.SimpleActionClient('/motion_server', cl_tsgrasp.msg.MotionAction)
    motion_client.wait_for_server()

    # import states AFTER this node and the services are initialized
    from states import OpenJaws, GoToOrbitalPose, TerminalHoming, CloseJaws, ResetPos, SpawnNewItem, Delay

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
                    'not_in_orbital_pose': 'GRASP_FAIL'
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
        
        
        smach.StateMachine.add('SPAWN_NEW_ITEM', SpawnNewItem(),
            transitions={
                'spawned_new_item':'DELAY'
            }
        )
        smach.StateMachine.add('RESET_INITIAL_POS', ResetPos,
            transitions={
                'succeeded': 'SPAWN_NEW_ITEM',
                'aborted': 'RESET_INITIAL_POS',
                'preempted': 'FAIL'
            }
        )
        smach.StateMachine.add('DELAY', Delay(1),
            transitions={
                'delayed':'SPAWN_NEW_ITEM'
            }
        )
        smach.StateMachine.add('GRASP', grasp_sm, 
            transitions={
                'GRASP_SUCCESS':'RESET_INITIAL_POS',
                'GRASP_FAIL':'RESET_INITIAL_POS'
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
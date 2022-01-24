#! /usr/bin/env python3
# Execute finite state machine for grasping.

import rospy
import smach_ros
import smach
import actionlib
import cl_tsgrasp.msg
from motion import Mover

# main
def main():
    rospy.init_node('smach_state_machine')

    # wait for services called within States
    # (located here because State.__init__ can't block)
    rospy.loginfo("Waiting for services.")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")

    # Start end-effector motion client
    mover = Mover()

    # import states AFTER this node and the services are initialized
    from states import  GoToOrbitalPose, TerminalHoming, SpawnNewItem, Delay, ResetPos, OpenJaws, CloseJaws

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
                OpenJaws(mover),
                transitions={
                    'jaws_open': 'GO_TO_ORBITAL_POSE',
                    'jaws_not_open': 'GRASP_FAIL'
                }
            )
            smach.StateMachine.add(
                'GO_TO_ORBITAL_POSE',
                GoToOrbitalPose(mover),
                transitions={
                    'in_orbital_pose': 'TERMINAL_HOMING',
                    'not_in_orbital_pose': 'GRASP_FAIL'
                }
            )
            smach.StateMachine.add(
                'TERMINAL_HOMING',
                TerminalHoming(),
                transitions={
                    'in_grasp_pose': 'CLOSE_JAWS',
                    'not_in_grasp_pose': 'GRASP_FAIL'
                }
            )
            smach.StateMachine.add(
                'CLOSE_JAWS',
                CloseJaws(mover),
                transitions={
                    'jaws_closed': 'RESET_POS',
                    'jaws_not_closed': 'GRASP_FAIL'
                }
            )
            smach.StateMachine.add(
                'RESET_POS',
                ResetPos(mover),
                transitions={
                    'position_reset': 'GRASP_SUCCESS',
                    'position_not_reset': 'GRASP_FAIL'
                }
            )
        
        smach.StateMachine.add('RESET_INITIAL_POS', ResetPos(mover),
            transitions={
                'position_reset': 'SPAWN_NEW_ITEM',
                'position_not_reset': 'RESET_INITIAL_POS'
            }
        )

        smach.StateMachine.add('SPAWN_NEW_ITEM', SpawnNewItem(),
            transitions={
                'spawned_new_item':'DELAY'
            }
        )
        
        smach.StateMachine.add('DELAY', Delay(3),
            transitions={
                'delayed':'GRASP'
            }
        )
        smach.StateMachine.add('GRASP', grasp_sm, 
            transitions={
                'GRASP_SUCCESS':'SPAWN_NEW_ITEM',
                'GRASP_FAIL':'SPAWN_NEW_ITEM'
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
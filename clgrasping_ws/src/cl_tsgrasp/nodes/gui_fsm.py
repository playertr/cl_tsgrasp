#! /usr/bin/python3
# Interactive GUI for transitioning FSM states.

import rospy
import smach_ros
import smach
import actionlib
import cl_tsgrasp.msg
from motion import Mover
import PySimpleGUI as sg

rospy.init_node('smach_state_machine')

# wait for services called within States
# (located here because State.__init__ can't block)
rospy.loginfo("Waiting for services.")
rospy.wait_for_service("gazebo/delete_model")
rospy.wait_for_service("gazebo/spawn_sdf_model")

# Start end-effector motion client
mover = Mover()

# import states AFTER this node and the services are initialized
from states import  GoToOrbitalPose, ServoToFinalPose, TerminalHoming, SpawnNewItem, Delay, ResetPos, OpenJaws, CloseJaws, GoToFinalPose


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

open_loop_grasp_sm = smach.StateMachine(outcomes=['GRASP_SUCCESS', 'GRASP_FAIL'])
open_loop_grasp_sm.userdata.final_goal_pose_input = None
with open_loop_grasp_sm:
    # Add states to the container
    smach.StateMachine.add(
        'OPEN_JAWS',
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
            'in_orbital_pose': 'GO_TO_FINAL_POSE',
            'not_in_orbital_pose': 'GRASP_FAIL'
        },
        remapping={
            'final_goal_pose':'final_goal_pose'
        }
    )
    smach.StateMachine.add(
        'GO_TO_FINAL_POSE',
        ServoToFinalPose(), #GoToFinalPose(mover),
        transitions={
            'in_grasp_pose': 'CLOSE_JAWS',
            'not_in_grasp_pose': 'GRASP_FAIL'
        },
        remapping={
            'final_goal_pose_input':'final_goal_pose'
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

reset_pos = ResetPos(mover)
spawn_new_item = SpawnNewItem()
delay = Delay(3)
grasp = grasp_sm
open_jaws = OpenJaws(mover)
go_to_orbital_pose = GoToOrbitalPose(mover)
terminal_homing = TerminalHoming()
close_jaws = CloseJaws(mover)
go_to_final_pose = GoToFinalPose(mover)
servo_to_final_pose = ServoToFinalPose()
grasp_open_loop = open_loop_grasp_sm

sg.theme('DarkAmber')   # Add a touch of color

# All the stuff inside your window.
layout = [  [sg.Button('reset_pos')],
            [sg.Button('spawn_new_item')],
            [sg.Button('delay')],
            [sg.Button('grasp')],
            [sg.Button('grasp_open_loop')],
            [sg.Button('open_jaws')],
            [sg.Button('go_to_orbital_pose')],
            [sg.Button('go_to_final_pose')],
            [sg.Button('servo_to_final_pose')],
            [sg.Button('terminal_homing')],
            [sg.Button('close_jaws')], 
            [sg.Text(size=(15,1), key='-OUTPUT-')]]

# Create the Window
window = sg.Window('State Machine', layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
        break
    elif event == 'reset_pos':
        state = reset_pos
    elif event == 'spawn_new_item':
        state = spawn_new_item
    elif event == 'delay':
        state = delay
    elif event == 'grasp':
        state = grasp
    elif event == 'grasp_open_loop':
        state = grasp_open_loop
    elif event == 'open_jaws':
        state = open_jaws
    elif event == 'go_to_orbital_pose':
        state = go_to_orbital_pose
    elif event == 'go_to_final_pose':
        state = go_to_final_pose
    elif event == 'servo_to_final_pose':
        state = servo_to_final_pose
    elif event == 'terminal_homing':
        state = terminal_homing
    elif event == 'close_jaws':
        state = close_jaws
    else:
        raise ValueError
    
    class UserData:
        final_goal_pose_input = None

    window['-OUTPUT-'].update(state.execute(UserData()))

window.close()
#! /usr/bin/python3
# Interactive GUI for transitioning FSM states.

import rospy
from motion import Mover
import PySimpleGUI as sg

rospy.init_node('smach_state_machine')

# wait for services called within States
# (located here because State.__init__ can't block)
rospy.loginfo("Waiting for services.")
rospy.wait_for_service("/bravo/plan_kinematic_path") # wait for moveit to be up and running
# rospy.wait_for_service("/gazebo/delete_model")
# rospy.wait_for_service("/gazebo/spawn_sdf_model")
rospy.wait_for_service('/bravo/apply_planning_scene') # just to make sure we can add to the planningscene in motion.py

# Start end-effector motion client
mover = Mover()

# import states AFTER this node and the services are initialized
from states import  (
    GoToOrbitalPose, GoToRest, ServoToFinalPose, TerminalHoming, 
    SpawnNewItem, Delay, GoToDrawnBack, OpenJaws, CloseJaws, 
    GoToFinalPose, AllowHandCollisions, DisallowHandCollisions,
    ExecuteGraspOpenLoop, ExecuteGraspClosedLoop, ServoToGoalPose
)

execute_ol_grasp = ExecuteGraspOpenLoop(mover)
execute_cl_grasp = ExecuteGraspClosedLoop(mover)
servo_to_goal_pose = ServoToGoalPose(mover)
go_to_drawn_back = GoToDrawnBack(mover)
go_to_rest = GoToRest(mover)
spawn_new_item = SpawnNewItem()
delay = Delay(3)
open_jaws = OpenJaws(mover)
go_to_orbital_pose = GoToOrbitalPose(mover)
terminal_homing = TerminalHoming()
close_jaws = CloseJaws(mover)
go_to_final_pose = GoToFinalPose(mover)
servo_to_final_pose = ServoToFinalPose()
allow_hand_collisions = AllowHandCollisions(mover)
disallow_hand_collisions = DisallowHandCollisions(mover)

sg.theme('DarkAmber')   # Add a touch of color

# All the stuff inside your window.
layout = [  [sg.Button('go_to_drawn_back')],
            [sg.Button('go_to_rest')],
            [sg.Button('spawn_new_item')],
            [sg.Button('grasp_open_loop')],
            [sg.Button('grasp_closed_loop')],
            [sg.Button('servo_to_goal_pose')],
            # [sg.Button('open_jaws')],
            # [sg.Button('go_to_orbital_pose')],
            # [sg.Button('go_to_final_pose')],
            # [sg.Button('servo_to_final_pose')],
            # [sg.Button('terminal_homing')],
            # [sg.Button('close_jaws')], 
            # [sg.Button('allow_hand_collisions')],
            # [sg.Button('disallow_hand_collisions')],  
            [sg.Text(size=(15,1), key='-OUTPUT-')]]

# Create the Window
window = sg.Window('State Machine', layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
        break
    elif event == 'go_to_drawn_back':
        state = go_to_drawn_back
    elif event == 'go_to_rest':
        state = go_to_rest
    elif event == 'grasp_open_loop':
        state = execute_ol_grasp
    elif event == 'grasp_closed_loop':
        state = execute_cl_grasp
    elif event == 'spawn_new_item':
        state = spawn_new_item
    elif event == 'servo_to_goal_pose':
        state = servo_to_goal_pose
    # elif event == 'open_jaws':
    #     state = open_jaws
    # elif event == 'go_to_orbital_pose':
    #     state = go_to_orbital_pose
    # elif event == 'go_to_final_pose':
    #     state = go_to_final_pose
    # elif event == 'servo_to_final_pose':
    #     state = servo_to_final_pose
    # elif event == 'terminal_homing':
    #     state = terminal_homing
    # elif event == 'close_jaws':
    #     state = close_jaws
    # elif event == 'allow_hand_collisions':
    #     state = allow_hand_collisions
    # elif event == 'disallow_hand_collisions':
    #     state = disallow_hand_collisions
    else:
        raise ValueError
    
    class UserData:
        final_goal_pose_input = None

    window['-OUTPUT-'].update(state.execute(UserData()))

window.close()
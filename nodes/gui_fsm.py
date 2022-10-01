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
rospy.wait_for_service('/bravo/apply_planning_scene') # just to make sure we can add to the planningscene in motion.py

# Start end-effector motion client
mover = Mover()

# import states AFTER this node and the services are initialized
from states import  (
    GoToRest, SpawnNewItem, Delay, OpenJaws, CloseJaws, 
    ExecuteGraspOpenLoop
)

go_to_rest = GoToRest(mover)
spawn_new_item = SpawnNewItem()
open_jaws = OpenJaws(mover)
close_jaws = CloseJaws(mover)
execute_ol_grasp = ExecuteGraspOpenLoop(mover)

sg.theme('DarkAmber')   # Add a touch of color

# All the stuff inside your window.
layout = [  [sg.Button('go_to_rest')],
            [sg.Button('spawn_new_item')],
            [sg.Button('open_jaws')],
            [sg.Button('close_jaws')],
            [sg.Button('grasp_open_loop')],
            [sg.Text(size=(15,1), key='-OUTPUT-')]]

# Create the Window
window = sg.Window('State Machine', layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
        break
    elif event == 'go_to_rest':
        state = go_to_rest
    elif event == 'spawn_new_item':
        state = spawn_new_item
    elif event == 'open_jaws':
        state = open_jaws
    elif event == 'close_jaws':
        state = close_jaws
    elif event == 'grasp_open_loop':
        state = execute_ol_grasp
    else:
        raise ValueError
    
    window['-OUTPUT-'].update(state.execute(userdata=None))

window.close()
#! /usr/bin/python3
# Select the best grasp from the published Grasps message.

import rospy
from cl_tsgrasp.msg import Grasps
from geometry_msgs.msg import PoseStamped, Pose
from rospy.numpy_msg import numpy_msg
import numpy as np
from tf.transformations import quaternion_matrix

from utils import TFHelper

import time
class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = True

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))

## Start node and publisher
rospy.init_node('grasp_chooser')
final_goal_pose_pub = rospy.Publisher(name='/tsgrasp/final_goal_pose', 
    data_class=numpy_msg(PoseStamped), queue_size=1)

tf = TFHelper()

final_goal_pose = None
grasp_lpf = None

def pose_to_homo(pose):
    tf = quaternion_matrix(np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]))
    tf[:3,3] = pose.position.x, pose.position.y, pose.position.z
    return tf

"""The (5,3) matrix of control points for a single gripper.
    Note: these may be in the wrong frame.
"""
cps = np.array([
    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
    [ 5.2687433e-02, -5.9955313e-05,  7.5273141e-02],
    [-5.2687433e-02,  5.9955313e-05,  7.5273141e-02],
    [ 5.2687433e-02, -5.9955313e-05,  1.0527314e-01],
    [-5.2687433e-02,  5.9955313e-05,  1.0527314e-01]])
cps = np.hstack([cps, np.ones((len(cps), 1))]) # make homogeneous

def dist(p1, p2):
    """Compare two poses by control point distance."""
    mat1 = pose_to_homo(p1)
    mat2 = pose_to_homo(p2)
    return np.mean(
        (mat1.dot(cps.T) - mat2.dot(cps.T))**2
    )

class GraspPoseLPF:
    """A lowpass filter for the terminal grasp pose.
    On update, this filter picks the BEST grasp that is CLOSEST to the previous selection. This is achieved by selecting the arg max over
    ```math
    scores[i] = closest_coeff * exp(-dists[1]/scale) + best_coeff * confs[i]
    ```
    """
    best_coeff    = 1.0 # type: float
    closest_coeff = 2.0 # type: float
    scale         = 0.01 # type: float

    best_grasp    = None # type: Pose
    def __init__(self, grasps, confs):
        self.reset(grasps, confs)

    def reset(self, grasps, confs):
        best_grasp_idx = np.argmax(confs)
        self.best_grasp = grasps[best_grasp_idx]

    def update(self, grasps, confs):
        if len(confs) == 0: return
        dists = np.array([dist(grasp, self.best_grasp) for grasp in grasps])
        scores = self.closest_coeff * np.exp(-dists/self.scale) + confs
        i = np.argmax(scores)

        print("dists[i]")
        print(dists[i])
        print("confs[i]")
        print(confs[i])
        print("scores[i]")
        print(scores[i])

        self.best_grasp = grasps[i]

from motion import PoseStampedExpoFilter
expo_filter = PoseStampedExpoFilter(0.9)

def publish_goal_callback(msg):
    """Identify the best grasp in the Grasps message and publish it."""

    with TimeIt("Complete Call"):
        global grasp_lpf

        confs = msg.confs
        poses = msg.poses

        # filter the grasps in the world frame (for now)
        world_poses = [tf.transform_pose(PoseStamped(header=msg.header, pose=pose), target_frame='world') for pose in poses]

        # allow only top-down-ish grasps (for now)
        if True:
            z_hat = np.array([0, 0, 1])
            valid_idcs = [
                i for (i, p) in enumerate(world_poses)
                if quaternion_matrix(np.array(
                    [p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w]))[:3,2].dot(z_hat) < -0.2
            ]

            confs = [confs[i] for i in valid_idcs]
            poses = [poses[i] for i in valid_idcs]
            if len(confs) == 0: return

        if grasp_lpf is None:
            grasp_lpf = GraspPoseLPF(poses, confs)
        else:
            grasp_lpf.update(poses, confs)
        
        final_goal_pose = PoseStamped()
        final_goal_pose.pose = grasp_lpf.best_grasp
        final_goal_pose.header = msg.header
        final_goal_pose_pub.publish(final_goal_pose)
    
grasp_sub = rospy.Subscriber(name='/tsgrasp/grasps', 
    data_class=Grasps, callback=publish_goal_callback, queue_size=1)

import PySimpleGUI as sg

layout = [
    [sg.Text('Slider Demonstration'), sg.Text('', key='-OUTPUT-')],
    [sg.Text('best_coeff'), sg.Slider((0,2), key='-BEST_COEFF_SLIDER-', orientation='h', enable_events=True, resolution=0.01)],
    [sg.Text('closest_coeff'), sg.Slider((0, 2), key='-CLOSEST_COEFF_SLIDER-', orientation='h', enable_events=True, resolution=0.01)],
    [sg.Text('scale'), sg.Slider((0, 0.1), key='-SCALE_SLIDER-', orientation='h', enable_events=True, resolution=0.001)],
]

window = sg.Window('Window Title', layout)

while True:             # Event Loop
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    grasp_lpf.best_coeff = float(values['-BEST_COEFF_SLIDER-'])
    grasp_lpf.closest_coeff = float(values['-CLOSEST_COEFF_SLIDER-'])
    grasp_lpf.scale = float(values['-SCALE_SLIDER-'])
    rospy.sleep(0.1)

window.close()

rospy.spin()
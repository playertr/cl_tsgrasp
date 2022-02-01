#! /usr/bin/python3
# Select the best grasp from the published Grasps message.

import rospy
from cl_tsgrasp.msg import Grasps
from geometry_msgs.msg import PoseStamped, Pose
from rospy.numpy_msg import numpy_msg
import numpy as np
from tf.transformations import quaternion_matrix

from utils import TransformFrames

## Start node and publisher
rospy.init_node('grasp_chooser')
final_goal_pose_pub = rospy.Publisher(name='/tsgrasp/final_goal_pose', 
    data_class=numpy_msg(PoseStamped), queue_size=1)

tf = TransformFrames()

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
    mat1 = pose_to_homo(p1.pose)
    mat2 = pose_to_homo(p2.pose)
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
    closest_coeff = 1.0 # type: float
    scale         = 0.03 # type: float

    best_grasp    = None # type: Pose
    def __init__(self, grasps, confs):
        self.reset(grasps, confs)

    def reset(self, grasps, confs):
        best_grasp_idx = np.argmax(confs)
        self.best_grasp = grasps[best_grasp_idx]

    def update(self, grasps, confs):
        dists = np.array([dist(grasp, self.best_grasp) for grasp in grasps])
        scores = self.closest_coeff * np.exp(-dists/self.scale) + confs
        i = np.argmax(scores)
        print("dists[i]")
        print(dists[i])
        print("confs[i]")
        print(confs[i])
        print("scores[i]")
        print(scores[i])
        self.best_grasp = grasps[np.argmax(scores)]

def publish_goal_callback(msg):
    """Identify the best grasp in the Grasps message and publish it."""

    global grasp_lpf

    confs = msg.confs
    poses = msg.poses

    # filter the grasps in the world frame (for now)
    world_poses = [tf.pose_transform(PoseStamped(header=msg.header, pose=pose), target_frame='world') for pose in poses]

    # allow only top-down-ish grasps (for now)
    if True:
        z_hat = np.array([0, 0, 1])
        valid_idcs = [
            i for (i, p) in enumerate(world_poses)
            if quaternion_matrix(np.array(
                [p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w]))[:3,2].dot(z_hat) < -0.6
        ]

        confs = [confs[i] for i in valid_idcs]
        poses = [poses[i] for i in valid_idcs]
        world_poses = [world_poses[i] for i in valid_idcs]
        if len(confs) == 0: return

    if grasp_lpf is None:
        grasp_lpf = GraspPoseLPF(world_poses, confs)
    else:
        grasp_lpf.update(world_poses, confs)

    best_pose_world_frame = grasp_lpf.best_grasp
    final_goal_pose = tf.pose_transform(best_pose_world_frame, target_frame='panda_link0')

    final_goal_pose_pub.publish(final_goal_pose)

grasp_sub = rospy.Subscriber(name='/tsgrasp/grasps', 
    data_class=Grasps, callback=publish_goal_callback, queue_size=1)

rospy.spin()
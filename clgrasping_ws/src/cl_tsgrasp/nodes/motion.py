#! /usr/bin/python3

import moveit_commander

import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
import numpy as np

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of
    their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True

class Mover:
    """Wrapper around MoveIt functionality for the panda arm."""

    arm: moveit_commander.MoveGroupCommander
    gripper: moveit_commander.MoveGroupCommander

    def __init__(self):
        moveit_commander.roscpp_initialize([])
        self.arm = moveit_commander.MoveGroupCommander("panda_arm")
        self.gripper = moveit_commander.MoveGroupCommander("panda_hand")

    def go_joints(self, joints: np.ndarray, wait: bool = True):
        """Move robot to the given 7-element joint configuration.

        Args:
            joints (np.ndarray): joint angles for panda
            wait (bool): whether to block until finished 
        """
        plan = self.arm.go(joints, wait=wait)
        self.arm.stop()
        return plan

    def go_gripper(self, pos: np.ndarray, wait: bool = True):
        """Move the gripper fingers to the two given positions.

        Args:
            pos (np.ndarray): gripper positions, in meters from center
            wait (bool): whether to block until finished
        """
        plan = self.gripper.go(pos, wait=wait)
        self.gripper.stop()
        return plan

    def go_ee_pose(self, pose: geometry_msgs.msg.Pose, wait: bool = True):
        """Move the end effector to the given pose.

        Args:
            pose (geometry_msgs.msg.Pose): desired pose for panda_hand
            wait (bool): whether to block until finished
        """

        self.arm.set_pose_target(pose)
        plan = self.arm.go(wait=wait)
        self.arm.stop()
        self.arm.clear_pose_targets()
        return plan

    def get_ee_pose(self):
        return self.arm.get_current_pose()

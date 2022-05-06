#! /usr/bin/python3

import moveit_commander
import sys

import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
import numpy as np

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        


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
    """Wrapper around MoveIt functionality for the arm."""

    arm_move_group_cmdr: moveit_commander.MoveGroupCommander
    arm_robot_cmdr: moveit_commander.RobotCommander
    arm_group_name: str
    scene: moveit_commander.PlanningSceneInterface
    gripper_pub: rospy.Publisher
    box_name: str = None
    grasping_group_name: str

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.arm_group_name = "arm"
        self.grasping_group_name = "hand"
        self.arm_robot_cmdr = moveit_commander.RobotCommander(robot_description="/bravo/robot_description")
        self.arm_move_group_cmdr = moveit_commander.MoveGroupCommander(self.arm_group_name, robot_description="/bravo/robot_description")

        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True) # might break the non-blocking promise
        self.gripper_pub = rospy.Publisher('/bravo/hand_position_controller/command', data_class=JointTrajectory, queue_size=1)
        self.add_ground_plane_to_planning_scene()

        self.arm_move_group_cmdr.set_planner_id("RRTConnect")

    def add_ground_plane_to_planning_scene(self):
        """Add a box object to the PlanningScene so that collisions with the 
        hand are ignored. Otherwise, no collision-free trajectories can be found 
        after an object is picked up."""

        eef_link = self.arm_move_group_cmdr.get_end_effector_link()

        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.arm_robot_cmdr.get_planning_frame()
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = -0.5  # above the hand frame
        box_name = "ground_plane"
        self.scene.add_box(box_name, box_pose, size=(3, 3, 1))

    def go_joints(self, joints: np.ndarray, wait: bool = True):
        """Move robot to the given 7-element joint configuration.

        Args:
            joints (np.ndarray): joint angles
            wait (bool): whether to block until finished 
        """
        joint_positions = self.arm_move_group_cmdr.get_current_joint_values() # ensure current
        plan = self.arm_move_group_cmdr.go(joints, wait=wait)
        self.arm_move_group_cmdr.stop()
        return plan


    def go_gripper(self, pos: np.ndarray, wait: bool = True):
        """Move the gripper fingers to the two given positions.

        Args:
            pos (np.ndarray): gripper positions, in meters from center
            wait (bool): whether to block until finished
        """
        jt = JointTrajectory()
        jt.joint_names = ['bravo_axis_a']
        jt.header.stamp = rospy.Time.now()

        jtp = JointTrajectoryPoint()
        jtp.positions = pos
        jtp.time_from_start = rospy.Duration(secs=3)

        jt.points.append(jtp)

        self.gripper_pub.publish(jt)
        return True

    def go_ee_pose(self, pose: geometry_msgs.msg.Pose, wait: bool = True):
        """Move the end effector to the given pose.

        Args:
            pose (geometry_msgs.msg.Pose): desired pose for end effector
            wait (bool): whether to block until finished
        """

        self.arm_move_group_cmdr.set_pose_target(pose)
        # self.arm_move_group_cmdr.set_goal_tolerance(0.1)
        plan = self.arm_move_group_cmdr.go(wait=wait)
        self.arm_move_group_cmdr.stop()
        self.arm_move_group_cmdr.clear_pose_targets()
        return plan
    
    def go_named_group_state(self, state: str, wait: bool = True):
        """Move the arm group to a named state from the SRDF.

        Args:
            state (str): the name of the state
            wait (bool, optional): whether to block until finished. Defaults to True.
        """
        self.arm_move_group_cmdr.set_named_target(state)
        plan = self.arm_move_group_cmdr.go(wait=wait)
        self.arm_move_group_cmdr.stop()
        return plan

    def get_ee_pose(self):
        return self.arm_move_group_cmdr.get_current_pose()

    def add_object_for_pickup(self):
        """Add a box object to the PlanningScene so that collisions with the 
        hand are ignored. Otherwise, no collision-free trajectories can be found 
        after an object is picked up."""

        eef_link = self.arm_move_group_cmdr.get_end_effector_link()

        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "ee_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = 0.11  # above the hand frame
        self.box_name = "hand_collision_box"
        self.scene.add_box(self.box_name, box_pose, size=(0.075, 0.075, 0.075))

        touch_links = self.arm_robot_cmdr.get_link_names(group=self.grasping_group_name)
        self.scene.attach_box(eef_link, self.box_name, touch_links=touch_links)


    def remove_object_after_pickup(self):
        """Remove a PlanningScene box after adding it in add_object_for_pickup."""

        eef_link = self.arm_move_group_cmdr.get_end_effector_link()

        if self.box_name is not None:
            self.scene.remove_attached_object(eef_link, name=self.box_name)
            self.scene.remove_world_object(self.box_name)
            self.box_name = None
        else:
            raise ValueError("No box was added to the planning scene. Did you call add_object_for_pickup?")

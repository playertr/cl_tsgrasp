#! /usr/bin/python3

import moveit_commander
import sys
import time

from geometry_msgs.msg import PoseStamped
import numpy as np

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from utils import TFHelper

class Mover:
    """Wrapper around MoveIt functionality for the arm."""

    arm_move_group_cmdr: moveit_commander.MoveGroupCommander
    arm_robot_cmdr: moveit_commander.RobotCommander
    arm_group_name: str
    scene: moveit_commander.PlanningSceneInterface
    gripper_pub: rospy.Publisher
    box_name: str = None
    grasping_group_name: str
    tfh: TFHelper

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.arm_group_name = "arm"
        self.grasping_group_name = "hand"
        self.arm_robot_cmdr = moveit_commander.RobotCommander(robot_description="robot_description")
        self.arm_move_group_cmdr = moveit_commander.MoveGroupCommander(self.arm_group_name, robot_description="robot_description")

        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True) # might break the non-blocking promise
        self.gripper_pub = rospy.Publisher('hand_position_controller/command', data_class=JointTrajectory, queue_size=1)
        self.add_ground_plane_to_planning_scene()

        self.arm_move_group_cmdr.set_planner_id("RRTConnect")

        self.tfh = TFHelper()

        self.grasp_chooser = GraspChooser(self.tfh)

    def add_ground_plane_to_planning_scene(self):
        """Add a box object to the PlanningScene to prevent paths that collide
        with the ground. """

        box_pose = PoseStamped()
        box_pose.header.frame_id = self.arm_robot_cmdr.get_planning_frame()
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = -0.501
        box_name = "ground_plane"
        self.scene.add_box(box_name, box_pose, size=(3, 3, 1))

    def get_ee_pose(self):
        return self.arm_move_group_cmdr.get_current_pose()

    def go_joints(self, joints: np.ndarray, wait: bool = True):
        """Move robot to the given joint configuration.

        Args:
            joints (np.ndarray): joint angles
            wait (bool): whether to block until finished 
        """
        success = self.arm_move_group_cmdr.go(joints, wait=wait)
        self.arm_move_group_cmdr.stop()
        return success

    def go_gripper(self, pos: np.ndarray, wait: bool = True) -> bool:
        """Move the gripper fingers to the given actuator value.

        Args:
            pos (np.ndarray): desired position of `bravo_axis_a`
            wait (bool): whether to wait until it's probably done
        """
        jt = JointTrajectory()
        jt.joint_names = ['bravo_axis_a']
        jt.header.stamp = rospy.Time.now()

        jtp = JointTrajectoryPoint()
        jtp.positions = pos
        jtp.time_from_start = rospy.Duration(secs=3)
        jt.points.append(jtp)

        self.gripper_pub.publish(jt)

        if wait:
            time.sleep(3)

        return True

    def go_ee_pose(self, pose: PoseStamped, wait: bool = True) -> bool:
        """Move the end effector to the given pose.

        Args:
            pose (geometry_msgs.msg.Pose): desired pose for end effector
            wait (bool): whether to block until finished
        """

        pose.header.stamp = rospy.Time.now() # the timestamp may be out of date.
        self.arm_move_group_cmdr.set_pose_target(pose, end_effector_link="ee_link")

        motion_goal_pub = rospy.Publisher('motion_goal', PoseStamped, queue_size=10)
        motion_goal_pub.publish(pose)

        success = self.arm_move_group_cmdr.go(wait=wait)
        self.arm_move_group_cmdr.stop()
        self.arm_move_group_cmdr.clear_pose_targets()

        return success
    
    def go_named_group_state(self, state: str, wait: bool = True) -> bool:
        """Move the arm group to a named state from the SRDF.

        Args:
            state (str): the name of the state
            wait (bool, optional): whether to block until finished. Defaults to True.
        """
        self.arm_move_group_cmdr.set_named_target(state)
        success = self.arm_move_group_cmdr.go(wait=wait)
        self.arm_move_group_cmdr.stop()

        return success

    def add_object_for_pickup(self):
        """Add a box object to the PlanningScene so that collisions with the 
        hand are ignored. Otherwise, no collision-free trajectories can be found 
        after an object is picked up."""

        eef_link = self.arm_move_group_cmdr.get_end_effector_link()

        box_pose = PoseStamped()
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
    
    def execute_grasp_open_loop(self, orbital_pose: PoseStamped, final_pose: PoseStamped) -> bool:
        """Execute an open loop grasp by planning and executing a sequence of motions in turn:
           1) Open the jaws
           2) Move the end effector the the orbital pose
           3) Move the end effector to the final pose
           4) Close the jaws
           5) Move the end effector to the orbital pose
           6) Move the arm to the 'rest' configuration

        Args:
            orbital_pose (PoseStamped): the pre-grasp orbital pose of the end effector
            final_pose (PoseStamped): the end effector pose in which to close the jaws

        Returns:
            bool: whether every step succeeded according to MoveIt
        """

        print("Executing grasp.")

        print("\tOpening jaws.")
        success = self.go_gripper(np.array([0.03]), wait=True)
        if not success: return False

        print("\tMoving end effector to orbital pose.")
        success = self.go_ee_pose(orbital_pose, wait=True)
        if not success: return False

        print("\tMoving end effector to final pose.")
        success = self.go_ee_pose(final_pose, wait=True)
        if not success: return False

        print("\tClosing jaws.")
        success = self.go_gripper(np.array([0.0]), wait=True)
        if not success: return False
        rospy.sleep(2)

        print("\tMoving end effector to orbital pose.")
        success = self.go_ee_pose(orbital_pose, wait=True)
        if not success: return False

        print("\tMoving arm to 'rest' configuration.")
        success = self.go_named_group_state('rest', wait=True)

        print("\tOpening jaws.")
        success = self.go_gripper(np.array([0.03]), wait=True)
        if not success: return False

        return success

from cl_tsgrasp.msg import Grasps
class GraspChooser:

    final_goal_pose = None
    grasp_lpf = None

    def __init__(self, tfh: TFHelper):

        self.tfh = tfh

        grasp_sub = rospy.Subscriber(name='/tsgrasp/grasps', 
            data_class=Grasps, callback=self.grasp_cb, queue_size=1)

        self.best_grasp_pub = rospy.Publisher(name='/tsgrasp/final_goal_pose', 
            data_class=PoseStamped, queue_size=1)

        self.orbital_best_grasp_pub = rospy.Publisher(name='/tsgrasp/orbital_final_goal_pose', 
            data_class=PoseStamped, queue_size=1)

        self.closest_grasp_lpf = None
        self.best_closest_grasp = None
        self.best_grasp = None

        self.grasp_closeness_importance = 0.5

    def grasp_cb(self, msg):
        """Identify the (best) and (best, closest) grasps in the Grasps message and publish them."""

        confs = msg.confs
        poses = msg.poses
        orbital_poses = msg.orbital_poses

        if len(confs) == 0: return

        # Find best (highest confidence) grasp and its orbital grasp
        best_grasp = PoseStamped()
        best_grasp.pose = poses[np.argmax(confs)]
        best_grasp.header = msg.header
        self.best_grasp_pub.publish(best_grasp)
        self.best_grasp = best_grasp

        orbital_best_grasp = PoseStamped()
        orbital_best_grasp.pose = orbital_poses[np.argmax(confs)]
        orbital_best_grasp.header = msg.header
        self.orbital_best_grasp_pub.publish(orbital_best_grasp)
        self.orbital_best_grasp = orbital_best_grasp

    def reset_closest_target(self, posestamped: PoseStamped):
        if posestamped.header.frame_id != self.best_closest_grasp.header.frame_id: raise ValueError
        self.closest_grasp_lpf.best_grasp = posestamped.pose

    def get_best_grasp(self):
        return self.best_grasp
    
    def get_best_closest_grasp(self):
        return self.best_closest_grasp
#! /usr/bin/python3

import moveit_commander
import sys
import time

from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Pose
from tf.transformations import euler_from_quaternion
import numpy as np

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from utils import se3_dist, TFHelper
import copy

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

        self.servo_helper = ServoHelper(
            twist_cmd_topic="/bravo/servo_server/delta_twist_cmds",
            ee_frame_id="ee_link",
            tfh = self.tfh,
            grasp_chooser = self.grasp_chooser
        )


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

    def execute_grasp_closed_loop(self, orbital_pose: PoseStamped, final_pose: PoseStamped) -> bool:
        """Execute an open loop grasp by planning and executing a sequence of motions in turn:
           1) Open the jaws
           2) Move the end effector the the orbital pose
           3) Servo the end effector to the (possibly changing) final pose
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
        success = self.go_gripper(np.array([0.02]), wait=True)
        if not success: return False

        print("\tMoving end effector to orbital pose.")
        success = self.go_ee_pose(orbital_pose, wait=True)
        if not success: return False

        print("\tMoving end effector to final pose.")
        success = self.servo_helper.servo_to_goal_pose(final_pose)
        if not success: return False

        print("\tClosing jaws.")
        success = self.go_gripper(np.array([0.0]), wait=True)
        if not success: return False

        print("\tMoving end effector to orbital pose.")
        success = self.go_ee_pose(orbital_pose, wait=True)
        if not success: return False

        print("\tMoving arm to 'rest' configuration.")
        success = self.go_named_group_state('rest', wait=True)
        return success

    def servo_to_goal_pose(self, final_goal_pose):
        return self.servo_helper.servo_to_goal_pose(final_goal_pose)

class ExpoFilter:
    """Scalar exponential FIR filter."""

    def __init__(self, tau: float = 0):
        self.tau = tau
        self.x = None

    def update(self, x: float) -> float:

        if self.x == None:
            self.x = x
        else:
            self.x = self.tau * self.x + (1 - self.tau) * x

        return self.x

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
class PoseStampedExpoFilter:
    """Exponential filter on a PoseStamped that applies an expo filter to each
    scalar term individually.
    
    Rotation uses spherical linear interpolation."""
    
    def __init__(self, tau=[0.9, 0.9, 0.9, 0.99]):
        self.x_filter = ExpoFilter()
        self.y_filter = ExpoFilter()
        self.z_filter = ExpoFilter()
        self.orientation = R.identity()
        self.set_tau(tau)

    def set_tau(self, tau):
        self.x_filter.tau = tau[0]
        self.y_filter.tau = tau[1]
        self.z_filter.tau = tau[2]
        self.orientation_tau = tau[3]

    def update(self, posestamped: PoseStamped) -> PoseStamped:

        x = self.x_filter.update(posestamped.pose.position.x)
        y = self.y_filter.update(posestamped.pose.position.y)
        z = self.z_filter.update(posestamped.pose.position.z)

        # lowpass filter using spherical linear interpolation
        orn = posestamped.pose.orientation
        orn = R.from_quat([orn.x, orn.y, orn.z, orn.w])
        self.orientation = Slerp(
            times=[0, 1],
            rotations=R.from_quat([self.orientation.as_quat(), orn.as_quat()]) # gross
        )(1 - self.orientation_tau)

        ps = copy.deepcopy(posestamped)
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.x = self.orientation.as_quat()[0]
        ps.pose.orientation.y = self.orientation.as_quat()[1]
        ps.pose.orientation.z = self.orientation.as_quat()[2]
        ps.pose.orientation.w = self.orientation.as_quat()[3]
        return ps

class ClosestBestFilter:
    """A lowpass filter for the terminal grasp pose.
    On update, this filter picks the BEST grasp that is CLOSEST to the previous selection. This is achieved by selecting the arg max over
    ```math
    scores[i] = grasp_closeness_importance * np.exp(-dists[i]/self.scale) + (1 - grasp_closeness_importance) * confs[i]
    ```
    """
    scale         = 0.01 # type: float

    best_grasp    = None # type: Pose
    def __init__(self, grasps, confs):
        self.reset(grasps, confs)

    def reset(self, grasps, confs):
        best_grasp_idx = np.argmax(confs)
        self.best_grasp = grasps[best_grasp_idx]

    def update(self, grasps, confs, grasp_closeness_importance):

        if len(confs) == 0: return
        dists = np.array([dist(grasp, self.best_grasp) for grasp in grasps])
        scores = grasp_closeness_importance * np.exp(-dists/self.scale) +  np.array(confs) * (1 - grasp_closeness_importance)
        i = np.argmax(scores)
        self.best_grasp = grasps[i]

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

from cl_tsgrasp.msg import Grasps
from tf.transformations import quaternion_matrix
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

        # self.best_closest_grasp_pub = rospy.Publisher(name='/tsgrasp/best_closest_grasp', 
        #     data_class=PoseStamped, queue_size=1)

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

        # Find best, closest grasp
        # if self.closest_grasp_lpf is None:
        #     self.closest_grasp_lpf = ClosestBestFilter(poses, confs)
        # else:
        #     self.closest_grasp_lpf.update(poses, confs, self.grasp_closeness_importance)
        
        # best_closest_grasp = PoseStamped()
        # best_closest_grasp.pose = self.closest_grasp_lpf.best_grasp
        # best_closest_grasp.header = msg.header
        # self.best_closest_grasp_pub.publish(best_closest_grasp)
        # self.best_closest_grasp = best_closest_grasp

    def reset_closest_target(self, posestamped: PoseStamped):
        if posestamped.header.frame_id != self.best_closest_grasp.header.frame_id: raise ValueError
        self.closest_grasp_lpf.best_grasp = posestamped.pose

    def get_best_grasp(self):
        return self.best_grasp
    
    def get_best_closest_grasp(self):
        return self.best_closest_grasp

class EndEffectorPID:
    """Controller to issue a Twist driving the end effector to a desired pose. """

    # Only proportional control implemented so far
    LAMBDA = 2.0*np.array([1, 1, 1, 1, 1, 1])

    def compute_twist(self, target_pose: PoseStamped) -> TwistStamped:
        """Generate visual servoing Twist command using proportional control over X,Y,Z,R,P,Y .

        Args:
            target_pose (PoseStamped): the pose of the target in the frame of the end effector

        Returns:
            TwistStamped: the twist to issue to MoveIt servo
        """
        
        err = self._posestamped_to_arr(target_pose.pose)
        efforts = self.LAMBDA * (err)

        twist = TwistStamped()
        twist.header = target_pose.header
        twist.header.stamp = rospy.Time.now() # make current so servo command doesn't time out
        twist.twist.linear = Vector3(*efforts[:3])
        twist.twist.angular = Vector3(*efforts[3:])

        return twist

    @staticmethod
    def _posestamped_to_arr(ps):
        pos, quat = ps.position, ps.orientation
        rpy = euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )
        return np.array([pos.x, pos.y, pos.z, *rpy])

from publish_orbital_pose import orbital_pose
class ServoHelper:
    """Helper class to handle pose-based visual servoing loop."""

    GOAL_PUB_RATE   = 60
    SUCCESS_RADIUS  = 0.03 # in SE(3) space -- dimensionless
    TIMEOUT         = 60
    PECK_SPEED      = 0.005 # m/s speed that gripper pecks forward

    _goal_pose = None
    tfh: TFHelper
    ee_frame_id: str

    def __init__(self, twist_cmd_topic, ee_frame_id, tfh, grasp_chooser: GraspChooser):
        
        self._servo_twist_pub = rospy.Publisher(
            name=twist_cmd_topic,
            data_class=TwistStamped, queue_size=1)
        self.ee_frame_id = ee_frame_id
        self.tfh = tfh

        self.end_effector_pid = EndEffectorPID()
        self.grasp_chooser = grasp_chooser

        self.filtered_pose_sub = rospy.Publisher("/tsgrasp/servoing_pose", PoseStamped, queue_size=1)

    def servo_to_goal_pose(self, final_pose):

        start_time = rospy.Time.now()

        origin = PoseStamped()
        origin.header.frame_id = self.ee_frame_id
        origin.pose.orientation.w = 1

        grasp_pose_filter = PoseStampedExpoFilter()

        orbital_distance = 0.05

        self.grasp_chooser.reset_closest_target(final_pose)

        # update goal pose at fixed rate
        in_grasp_pose = False
        rate = rospy.Rate(self.GOAL_PUB_RATE)
        while rospy.Time.now() - start_time < rospy.Duration(self.TIMEOUT):

            self.grasp_chooser.grasp_closeness_importance = 1.0

            # Apply a low-pass filter to the best grasp in the world frame.
            best_closest_grasp_world = self.tfh.transform_pose(
                self.grasp_chooser.get_best_closest_grasp(),
                "world"
            )

            # Lock the lowpass filter output when within 5 cm
            if orbital_distance < 0.05:
                grasp_pose_filter.set_tau([1]*6)

            filtered_grasp = grasp_pose_filter.update(best_closest_grasp_world)
            target_pose = orbital_pose(filtered_grasp, orbit_radius=orbital_distance)

            # Transform the target pose into the end-effector frame for servoing.
            target_pose = self.tfh.transform_pose(
                target_pose,
                self.ee_frame_id
            )

            self.filtered_pose_sub.publish(target_pose)

            dist = se3_dist(origin.pose, target_pose.pose)
            if  orbital_distance == 0 and dist < self.SUCCESS_RADIUS:
                in_grasp_pose = True
                break

            self._servo_twist_pub.publish(
                self.end_effector_pid.compute_twist(target_pose)
            )
            
            orbital_distance = max(orbital_distance - self.PECK_SPEED / self.GOAL_PUB_RATE, 0)

            rate.sleep()

        # send a zero twist to help moveit_servo finish
        zero_twist = TwistStamped()
        zero_twist.header.frame_id = self.ee_frame_id
        self._servo_twist_pub.publish(zero_twist)

        print("servoing succeeded" if in_grasp_pose else "servoing failed")

        return in_grasp_pose
import rospy
import smach
from geometry_msgs.msg import PoseStamped, Pose, Vector3, Quaternion, Point, TwistStamped
import numpy as np
from rospy.numpy_msg import numpy_msg
from utils import se3_dist
from spawn_model import ObjectDataset
from gazebo_msgs.srv import DeleteModel, SpawnModel
import tf.transformations
from motion import Mover
import os

## constants

# TerminalHoming
GOAL_PUB_RATE   = 30
STEP_SPEED      = 1.0
SUCCESS_RADIUS  = 0.01 # in SE(3) space -- dimensionless
TIMEOUT         = 8
LAMBDA          = 1.5*np.array([1, 1, 1, 1, 1, 1])
MIN_VEL         = 0.1
MIN_DIST        = 0.001

# SpawnNewItem
WORKSPACE_BOUNDS    = [[-0.1, -0.3], [0.1, -0.1]]
TABLE_HEIGHT        = 0.2

# SpawnNewItem state
class SpawnNewItem(smach.State):
    # _cur_idx: int   = 0
    # _cur_name: str  = None
    _cur_idx = 0
    _cur_name = None

    def __init__(self):
        smach.State.__init__(self, outcomes=['spawned_new_item'])

        self.obj_ds = ObjectDataset(
            dataset_dir=os.environ['NN_DATASET_DIR'],
            split="train"
        )
        self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        # reproducibly shuffle the items
        np.random.seed(2022)
        self.idcs = np.random.permutation(len(self.obj_ds))
        self._cur_item = 0

    @staticmethod
    def random_pose_in_workspace():
        x = np.random.uniform(WORKSPACE_BOUNDS[0][0], WORKSPACE_BOUNDS[1][0])
        y = np.random.uniform(WORKSPACE_BOUNDS[0][1], WORKSPACE_BOUNDS[1][1])

        z = TABLE_HEIGHT
        yaw = np.random.uniform(0, 2*np.pi)

        x, y, yaw = 0.4, -0.3, 0 # test

        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        return Pose(
            position=Point(x, y, z),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        )

    def execute(self, userdata):
        """ Delete old item, spawn new item. """
        if self._cur_name is not None:
            self.delete_model(self._cur_name)

        if self._cur_item > len(self.obj_ds): self._cur_item = 0
        obj = self.obj_ds[self.idcs[self._cur_item]]
        self._cur_item += 1
        
        self._cur_name = obj.name
        item_pose = self.random_pose_in_workspace()
        item_pose.position.x -= obj.center_of_mass[0] # correct for strange obj origin
        item_pose.position.y -= obj.center_of_mass[1]
        self.spawn_model(self._cur_name, obj.to_sdf(), "", item_pose, "world")
        print(obj.to_sdf())
        return 'spawned_new_item'

class Delay(smach.State):
    def __init__(self, seconds):
        self.seconds = seconds
        smach.State.__init__(self, outcomes=['delayed'])

    def execute(self, userdata):
        rospy.sleep(self.seconds)
        return 'delayed'

# GoToOrbitalPose state
class GoToOrbitalPose(smach.State):
    _orbital_pose = None 
    _final_goal_pose = None

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['in_orbital_pose', 'not_in_orbital_pose'], output_keys=['final_goal_pose'])
        self.mover = mover
        orbital_pose_sub = rospy.Subscriber(
            name='/tsgrasp/orbital_pose', data_class=PoseStamped, 
            callback=self._orbital_pose_cb, queue_size=1)
        final_goal_pose_sub = rospy.Subscriber(
            name='/tsgrasp/final_goal_pose', data_class=PoseStamped, 
            callback=self._goal_pose_cb, queue_size=1)

    def _orbital_pose_cb(self, msg):
        self._orbital_pose = msg

    def _goal_pose_cb(self, msg):
        self._final_goal_pose = msg

    def execute(self, userdata):
        rospy.loginfo('Executing state GO_TO_ORBITAL_POSE')
        if self._orbital_pose is None:
            rospy.loginfo('Orbital pose has not been initialized.')
            return 'not_in_orbital_pose'
        rospy.loginfo(f"Orbital pose is:\n{self._orbital_pose}")

        userdata.final_goal_pose = self._final_goal_pose
        success = self.mover.go_ee_pose(self._orbital_pose)
        return 'in_orbital_pose' if success else 'not_in_orbital_pose'

# GoToFinalPose state
class GoToFinalPose(smach.State):
    _final_pose = None 

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['in_grasp_pose', 'not_in_grasp_pose'])
        self.mover = mover
        orbital_pose_sub = rospy.Subscriber(
            name='/tsgrasp/final_goal_pose', data_class=PoseStamped, 
            callback=self._goal_pose_cb, queue_size=1)

    def _goal_pose_cb(self, msg):
        self._final_pose = msg

    def execute(self, userdata):
        rospy.loginfo('Executing state GO_TO_FINAL_POSE')
        if self._final_pose is None:
            rospy.loginfo('Final goal pose has not been initialized.')
            return 'not_in_grasp_pose'

        success = self.mover.go_ee_pose(self._final_pose)
        return 'in_grasp_pose' if success else 'not_in_grasp_pose'

# GoToFinalPose state
class GoToOriginalFinalPose(smach.State):
    _final_pose = None 

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['in_grasp_pose', 'not_in_grasp_pose'], input_keys=['final_goal_pose_input'])
        self.mover = mover

    def _goal_pose_cb(self, msg):
        self._final_pose = msg

    def execute(self, userdata):
        rospy.loginfo('Executing state GO_TO_FINAL_POSE')
        if self._final_pose is None:
            rospy.loginfo('Final goal pose has not been initialized.')
            return 'not_in_grasp_pose'

        success = self.mover.go_ee_pose(self._final_pose)
        return 'in_grasp_pose' if success else 'not_in_grasp_pose'

## Terminal homing state
class TerminalHoming(smach.State):
    _ee_pose = None
    _goal_pose = None

    def __init__(self):
        smach.State.__init__(self, outcomes=['in_grasp_pose', 'not_in_grasp_pose'])

        goal_pose_sub = rospy.Subscriber(
            name='/tsgrasp/final_goal_pose', data_class=PoseStamped, 
            callback=self._goal_pose_cb, queue_size=1)
        ee_pose_sub = rospy.Subscriber(
            name='/tsgrasp/ee_pose', 
            data_class=PoseStamped, callback=self._ee_pose_cb, queue_size=1)
        self._servo_twist_pub = rospy.Publisher(
            name='/bravo/servo_server/delta_twist_cmds', 
            data_class=numpy_msg(TwistStamped), queue_size=1)

    def _ee_pose_cb(self, msg):
        self._ee_pose = msg

    def _goal_pose_cb(self, msg):
        self._goal_pose = msg

    @staticmethod
    def _posestamped_to_arr(ps):
        pos, quat = ps.position, ps.orientation
        rpy = tf.transformations.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )
        return np.array([pos.x, pos.y, pos.z, *rpy])

    def _compute_twist(self, ee_pose: PoseStamped, final_goal_pose: PoseStamped) -> TwistStamped:
        """
            Generate visual servoing Twist command using proportional control over X,Y,Z,R,P,Y .
        """
        p1 = self._posestamped_to_arr(ee_pose.pose)
        p2 = self._posestamped_to_arr(final_goal_pose.pose)

        err = p2 - p1
        efforts = LAMBDA * (err) + MIN_VEL * np.sign(err) * (np.abs(err) > MIN_DIST)

        twist = TwistStamped()
        twist.header = ee_pose.header
        twist.header.stamp = rospy.Time.now() # make current so servo command doesn't time out
        twist.twist.linear = Vector3(*efforts[:3])
        twist.twist.angular = Vector3(*efforts[3:])

        rospy.loginfo(f"Final pose \n{final_goal_pose.pose.position}")        
        rospy.loginfo(f"ee pose: \n{ee_pose.pose.position}")
        rospy.loginfo(f"Issuing twist: \n{twist.twist.linear}")
        
        return twist

    def execute(self, userdata):
        rospy.loginfo('Executing state TERMINAL_HOMING')

        while self._ee_pose is None or self._goal_pose is None:
            print(("ee_pose", self._ee_pose))
            print(("goal_pose", self._goal_pose))
            rospy.loginfo('Waiting for final goal pose.')
            rospy.sleep(1)

        start_time = rospy.Time.now()

        # update goal pose at fixed rate
        # note: an asynchronous callback may be better.
        in_grasp_pose = False
        rate = rospy.Rate(GOAL_PUB_RATE)
        while rospy.Time.now() - start_time < rospy.Duration(TIMEOUT):
            in_grasp_pose = se3_dist(self._ee_pose.pose, self._goal_pose.pose) < SUCCESS_RADIUS
            
            if in_grasp_pose:
                break

            self._servo_twist_pub.publish(
                self._compute_twist(self._ee_pose, self._goal_pose)
            )
            rate.sleep()

        # send a zero twist to help moveit_servo finish
        self._servo_twist_pub.publish(
                self._compute_twist(self._ee_pose, self._ee_pose) # zero twist
        )

        return 'in_grasp_pose' if in_grasp_pose else 'not_in_grasp_pose'

class ServoToFinalPose(TerminalHoming):
    """Just like TerminalHoming, but the target never moves after being set. Also accepts the target as a userdata element."""

    def __init__(self):
        smach.State.__init__(self, outcomes=['in_grasp_pose', 'not_in_grasp_pose'], input_keys=['final_goal_pose_input'])
        ee_pose_sub = rospy.Subscriber(
            name='/tsgrasp/ee_pose', 
            data_class=PoseStamped, callback=self._ee_pose_cb, queue_size=1)
        self._servo_twist_pub = rospy.Publisher(
            name='/bravo/servo_server/delta_twist_cmds', 
            data_class=numpy_msg(TwistStamped), queue_size=1)

    def execute(self, userdata):
        if userdata.final_goal_pose_input is not None:
            self._goal_pose = userdata.final_goal_pose_input

        return super().execute(userdata)

class ServoToOrbitalPose(TerminalHoming):
    """Just like TerminalHoming, but the target never moves after being set. Also accepts the target as a userdata element."""

    def __init__(self):
        smach.State.__init__(self, outcomes=['in_grasp_pose', 'not_in_grasp_pose'], input_keys=['final_goal_pose_input'])

        goal_pose_sub = rospy.Subscriber(
            name='/tsgrasp/final_goal_pose', data_class=PoseStamped, 
            callback=self._goal_pose_cb, queue_size=1)
        ee_pose_sub = rospy.Subscriber(
            name='/tsgrasp/ee_pose', 
            data_class=PoseStamped, callback=self._ee_pose_cb, queue_size=1)
        self._servo_twist_pub = rospy.Publisher(
            name='/bravo/servo_server/delta_twist_cmds', 
            data_class=numpy_msg(TwistStamped), queue_size=1)

    def _goal_pose_cb(self, msg):
        if self._goal_pose is None:
            self._goal_pose = msg

    def execute(self, userdata):
        if userdata.final_goal_pose_input is not None:
            self._goal_pose = userdata.final_goal_pose_input

        return super().execute(userdata)

class AllowHandCollisions(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['hand_collisions_allowed', 'hand_collisions_not_allowed'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Allowing hand collisions by adding PlanningScene object.')
        self.mover.add_object_for_pickup()
        return 'hand_collisions_allowed'

class DisallowHandCollisions(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['hand_collisions_disallowed', 'hand_collisions_not_disallowed'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Disallowing hand collisions by removing PlanningScene object.')
        self.mover.remove_object_after_pickup()
        return 'hand_collisions_disallowed'

class OpenJaws(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['jaws_open', 'jaws_not_open'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Executing state OPEN_JAWS')
        success = self.mover.go_gripper(np.array([0.02]))
        return 'jaws_open' if success else 'jaws_not_open'

class CloseJaws(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['jaws_closed', 'jaws_not_closed'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Executing state CLOSE_JAWS')
        success = self.mover.go_gripper(np.array([0.0]))
        return 'jaws_closed' if success else 'jaws_not_closed'

class GoToDrawnBack(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['position_reset', 'position_not_reset'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Executing state GO_TO_DRAWN_BACK')
        success = self.mover.go_named_group_state('drawn_back')
        return 'position_reset' if success else 'position_not_reset'

class GoToRest(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['position_reset', 'position_not_reset'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Executing state GO_TO_REST')
        success = self.mover.go_named_group_state('rest')
        return 'position_reset' if success else 'position_not_reset'

class ServoToGoalPose(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['position_reset', 'position_not_reset'])
        self.mover = mover
        final_goal_pose_sub = rospy.Subscriber(
            name='/tsgrasp/final_goal_pose', data_class=PoseStamped, 
            callback=self._goal_pose_cb, queue_size=1)

    def _goal_pose_cb(self, msg):
        self._final_goal_pose = msg

    def execute(self, userdata):
        rospy.loginfo('Executing state SERVO_TO_GOAL_POSE')
        success = self.mover.servo_to_goal_pose(self._final_goal_pose)
        return 'in_goal_pose' if success else 'not_in_goal_pose'

class ExecuteGraspOpenLoop(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['grasp_executed', 'grasp_not_executed'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Executing state EXECUTE_GRASP_OPEN_LOOP')
        if self.mover.grasp_chooser.best_grasp is None:
            rospy.logerr('Orbital or final pose not initialized.')
            return 'grasp_not_executed'

        success = self.mover.execute_grasp_open_loop(self.mover.grasp_chooser.orbital_best_grasp, self.mover.grasp_chooser.best_grasp)
        return 'grasp_executed' if success else 'grasp_not_executed'

class ExecuteGraspClosedLoop(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['grasp_executed', 'grasp_not_executed'])
        self.mover = mover
        orbital_pose_sub = rospy.Subscriber(
            name='/tsgrasp/orbital_pose', data_class=PoseStamped, 
            callback=self._orbital_pose_cb, queue_size=1)
        final_goal_pose_sub = rospy.Subscriber(
            name='/tsgrasp/final_goal_pose', data_class=PoseStamped, 
            callback=self._goal_pose_cb, queue_size=1)

    def _orbital_pose_cb(self, msg):
        self._orbital_pose = msg

    def _goal_pose_cb(self, msg):
        self._final_goal_pose = msg

    def execute(self, userdata):
        rospy.loginfo('Executing state EXECUTE_GRASP_CLOSED_LOOP')
        if self._orbital_pose is None or self._final_goal_pose is None:
            rospy.logerr('Orbital or final pose not initialized.')
            return 'grasp_not_executed'

        success = self.mover.execute_grasp_closed_loop(self._orbital_pose, self._final_goal_pose)
        return 'grasp_executed' if success else 'grasp_not_executed'
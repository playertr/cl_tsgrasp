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

## constants
# GoToOrbitalPose
ORBIT_RADIUS   = 0.1 # distance to initially orbit object before terminal homing

# TerminalHoming
GOAL_PUB_RATE   = 30
STEP_SPEED      = 1.0
SUCCESS_RADIUS  = 0.02 # in SE(3) space -- dimensionless
TIMEOUT         = 80
LAMBDA          = 0.1*np.array([1, 1, 1, 1, 1, 1])

# SpawnNewItem
WORKSPACE_BOUNDS    = [[-0.1, -0.3], [0.1, -0.1]]
TABLE_HEIGHT        = 0.4

# SpawnNewItem state
class SpawnNewItem(smach.State):
    # _cur_idx: int   = 0
    # _cur_name: str  = None
    _cur_idx = 0
    _cur_name = None

    def __init__(self):
        smach.State.__init__(self, outcomes=['spawned_new_item'])

        self.obj_ds = ObjectDataset(
            dataset_dir="/home/tim/Research/tsgrasp/data/dataset",
            split="train"
        )
        self.delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self.spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        self._cur_idx = np.random.randint(0, len(self.obj_ds))

    @staticmethod
    def random_pose_in_workspace():
        x = np.random.uniform(WORKSPACE_BOUNDS[0][0], WORKSPACE_BOUNDS[1][0])
        y = np.random.uniform(WORKSPACE_BOUNDS[0][1], WORKSPACE_BOUNDS[1][1])

        z = TABLE_HEIGHT
        yaw = np.random.uniform(0, 2*np.pi)

        x, y, yaw = 0, 0, 0 # test

        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        return Pose(
            position=Point(x, y, z),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        )

    def execute(self, userdata):
        """ Delete old item, spawn new item. """
        if self._cur_name is not None:
            self.delete_model(self._cur_name)

        if self._cur_idx > len(self.obj_ds): self._cur_idx = 0
        obj = self.obj_ds[self._cur_idx]
        self._cur_idx += 1
        
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

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['in_orbital_pose', 'not_in_orbital_pose'])
        self.mover = mover
        orbital_pose_sub = rospy.Subscriber(
            name='tsgrasp/orbital_pose', data_class=PoseStamped, 
            callback=self._goal_pose_cb, queue_size=1)

    def _goal_pose_cb(self, msg):
        self._orbital_pose = msg

    def execute(self, userdata):
        rospy.loginfo('Executing state GO_TO_ORBITAL_POSE')
        if self._orbital_pose is None:
            rospy.loginfo('Orbital pose has not been initialized.')
            return 'not_in_orbital_pose'

        success = self.mover.go_ee_pose(self._orbital_pose)
        return 'in_orbital_pose' if success else 'not_in_orbital_pose'

# GoToFinalPose state
class GoToFinalPose(smach.State):
    _final_pose = None 

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['in_grasp_pose', 'not_in_grasp_pose'])
        self.mover = mover
        orbital_pose_sub = rospy.Subscriber(
            name='tsgrasp/final_goal_pose', data_class=PoseStamped, 
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

## Terminal homing state
class TerminalHoming(smach.State):
    _ee_pose = None
    _goal_pose = None

    def __init__(self):
        smach.State.__init__(self, outcomes=['in_grasp_pose', 'not_in_grasp_pose'])

        goal_pose_sub = rospy.Subscriber(
            name='tsgrasp/final_goal_pose', data_class=PoseStamped, 
            callback=self._goal_pose_cb, queue_size=1)
        ee_pose_sub = rospy.Subscriber(
            name='tsgrasp/ee_pose', 
            data_class=PoseStamped, callback=self._ee_pose_cb, queue_size=1)
        self._servo_twist_pub = rospy.Publisher(
            name='/servo_server/delta_twist_cmds', 
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

        efforts = LAMBDA * (p2 - p1)

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
        
        rate = rospy.Rate(GOAL_PUB_RATE)
        while se3_dist(self._ee_pose.pose, self._goal_pose.pose) > SUCCESS_RADIUS:
            if rospy.Time.now() - start_time > rospy.Duration(TIMEOUT):
                return 'not_in_grasp_pose'

            self._servo_twist_pub.publish(
                self._compute_twist(self._ee_pose, self._goal_pose)
            )
            rate.sleep()

        return 'in_grasp_pose'

class ServoToFinalPose(TerminalHoming):
    """Just like TerminalHoming, but the target never moves after being set."""

    def _goal_pose_cb(self, msg):
        if self._goal_pose is None:
            self._goal_pose = msg

## Open Jaws State
class OpenJaws(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['jaws_open', 'jaws_not_open'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Executing state OPEN_JAWS')
        success = self.mover.go_gripper(np.array([0.04, 0.04]))
        return 'jaws_open' if success else 'jaws_not_open'

## Close Jaws State
class CloseJaws(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['jaws_closed', 'jaws_not_closed'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Executing state CLOSE_JAWS')
        success = self.mover.go_gripper(np.array([0.0, 0.0]))
        return 'jaws_closed' if success else 'jaws_not_closed'

## Reset Position State
class ResetPos(smach.State):

    STARTING_POS = np.array([
        0.0,
        -0.9,
        0.0,
        -1.65,
        0.0,
        1.40,
        0.0
    ])

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['position_reset', 'position_not_reset'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Executing state RESET_POSITION')
        success = self.mover.go_joints(self.STARTING_POS)
        return 'position_reset' if success else 'position_not_reset'

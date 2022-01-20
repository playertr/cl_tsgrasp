import rospy
from smach_ros import SimpleActionState
import smach
from franka_gripper.msg import MoveAction, MoveGoal, GraspAction, GraspGoal
from geometry_msgs.msg import PoseStamped, Pose, Vector3, Quaternion, Point
import cl_tsgrasp.msg
import numpy as np
from rospy.numpy_msg import numpy_msg
from utils import se3_dist
from spawn_model import ObjectDataset
from gazebo_msgs.srv import DeleteModel, SpawnModel
import tf.transformations

## constants
# GoToOrbitalPose
ORBIT_RADIUS   = 0.1 # distance to initially orbit object before terminal homing

# TerminalHoming
GOAL_PUB_RATE   = 100.0
STEP_SPEED      = 1.0
STEP_SIZE       = STEP_SPEED / GOAL_PUB_RATE
SUCCESS_RADIUS  = 0.02 # in SE(3) space -- dimensionless
TIMEOUT         = 8

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
            split="test"
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

    def __init__(self, motion_client):
        smach.State.__init__(self, outcomes=['in_orbital_pose', 'not_in_orbital_pose'])
        self._motion_client = motion_client
        orbital_pose_sub = rospy.Subscriber(
            name='tsgrasp/orbital_pose', data_class=PoseStamped, 
            callback=self._goal_pose_cb, queue_size=1)

    def _goal_pose_cb(self, msg):
        self._orbital_pose = msg

    def execute(self, userdata):
        rospy.loginfo('Executing state GO_TO_ORBITAL_POSE')

        while self._orbital_pose is None:
            rospy.loginfo('Waiting for orbital pose.')
            rospy.sleep(1)

        goal = cl_tsgrasp.msg.MotionGoal(goal_pose=self._orbital_pose)
        self._motion_client.send_goal(goal)
        finished = self._motion_client.wait_for_result(rospy.Duration.from_sec(5.0))
        return 'in_orbital_pose' if finished else 'not_in_orbital_pose'

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
        self._eq_pose_pub = rospy.Publisher(name='/tsgrasp/goal_pose', 
            data_class=numpy_msg(PoseStamped), queue_size=1)

    def _ee_pose_cb(self, msg):
        self._ee_pose = msg

    def _goal_pose_cb(self, msg):
        self._goal_pose = msg

    def _intermediate_goal(self, ee_pose, final_goal_pose):
        """
            Set an intermediate goal with a position that is close to the current EE position.
        """

        cp = ee_pose.pose.position
        curr_pos = np.array([cp.x, cp.y, cp.z])
        gp = final_goal_pose.pose.position
        goal_pos = np.array([gp.x, gp.y, gp.z])

        pos_diff = goal_pos - curr_pos
        if np.linalg.norm(pos_diff) < STEP_SIZE:

            rospy.loginfo("Setting goal pose.")
            intermediate_pos = goal_pos
        else:
            rospy.loginfo("Setting intermediate pose.")
            intermediate_pos = curr_pos + STEP_SIZE * pos_diff / np.linalg.norm(pos_diff)

        intermediate_pose = Pose(
            position=Vector3(*intermediate_pos), 
            orientation=final_goal_pose.pose.orientation)
        return PoseStamped(header=final_goal_pose.header, pose=intermediate_pose)

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
            self._eq_pose_pub.publish(self._intermediate_goal(
                self._ee_pose, self._goal_pose))
            rate.sleep()

        return 'in_grasp_pose'

## Open Jaws State
open_goal = MoveGoal(width=0.08, speed=0.1)
OpenJaws = SimpleActionState(
    '/panda/franka_gripper/move',
    MoveAction,
    goal=open_goal
)

## Close Jaws State
close_goal = GraspGoal(
    width=0.00,
    speed=0.05, 
    force=10.0
)
close_goal.epsilon.inner=0.04
close_goal.epsilon.outer=0.04
CloseJaws = SimpleActionState(
    '/panda/franka_gripper/grasp',
    GraspAction,
    goal=close_goal
)

## Reset Position State
q = tf.transformations.quaternion_from_euler(np.pi, -np.pi/3, 0)
reset_pose = PoseStamped(
    pose=Pose(
        position=Vector3(z=0.7),
        orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    )
)
reset_pos_goal = cl_tsgrasp.msg.MotionGoal(goal_pose=reset_pose)
ResetPos = SimpleActionState(
    '/motion_server',
    cl_tsgrasp.msg.MotionAction,
    goal=reset_pos_goal
)

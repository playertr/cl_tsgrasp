import rospy
import smach
from geometry_msgs.msg import Pose, Quaternion, Point
import numpy as np
from spawn_model import ObjectDataset
from gazebo_msgs.srv import DeleteModel, SpawnModel
import tf.transformations
from motion import Mover
import os

# SpawnNewItem constants
WORKSPACE_BOUNDS    = [[-0.1, -0.3], [0.1, -0.1]]
TABLE_HEIGHT        = 0.2

class SpawnNewItem(smach.State):
    """Spawn an item from ShapeNetSem into the Gazebo world"""
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

        x, y, yaw = 0.6, 0.1, 0 # test

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

class GoToRest(smach.State):

    def __init__(self, mover: Mover):
        smach.State.__init__(self, outcomes=['position_reset', 'position_not_reset'])
        self.mover = mover

    def execute(self, userdata):
        rospy.loginfo('Executing state GO_TO_REST')
        success = self.mover.go_named_group_state('rest')
        return 'position_reset' if success else 'position_not_reset'

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
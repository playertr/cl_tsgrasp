import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np

## tf2 abstraction
# https://gitlab.msu.edu/av/av_notes/-/blob/master/ROS/Coordinate_Transforms.md
class TransformFrames():
    def __init__(self):
        ''' Create a buffer of transforms and update it with TransformListener '''
        self.tfBuffer = tf2_ros.Buffer()           # Creates a frame buffer
        tf2_ros.TransformListener(self.tfBuffer)   # TransformListener fills the buffer as background task
    
    def get_transform(self, source_frame, target_frame):
        ''' Lookup latest transform between source_frame and target_frame from the buffer '''
        try:
            trans = self.tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.2) )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f'Cannot find transformation from {source_frame} to {target_frame}.')
            raise e
        return trans     # Type: TransformStamped

    def pose_transform(self, pose_s, target_frame='odom'):
        ''' pose_s: PoseStamped will be transformed to target_frame '''
        trans = self.get_transform( pose_s.header.frame_id, target_frame )
        pose_t = tf2_geometry_msgs.do_transform_pose(pose_s, trans)
        return pose_t

def se3_dist(p1, p2):
    """
        'Distance' between two poses. Presently, just gives R(3) distance.
    """
    return np.linalg.norm(np.array([
        p2.position.x - p1.position.x, p2.position.y - p1.position.y, p2.position.z - p1.position.z]))

        
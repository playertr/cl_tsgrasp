#! /usr/bin/env python3
# Read in the predicted grasp poses and publish markers to visualize.
# Takes in the topic "grasps" and publishes a MarkerArray to "grasp_pose_markers".
# Accepts an optional argument, "--color", documented in argparse help.

import rospy
from cl_tsgrasp.msg import Grasps
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib import cm
import argparse

color = 'confs'
marker_array_pub = rospy.Publisher('grasp_pose_markers', MarkerArray, queue_size=1)
    
def gripper_marker():
    marker = Marker()
    marker.type = marker.MESH_RESOURCE
    marker.action = marker.MODIFY
    marker.mesh_resource = "package://cl_tsgrasp/urdf/gripper_yawed.stl"
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    return marker

def poses_cb(msg):
    global color

    total_markers = 1000
    skip = int(len(msg.poses) / total_markers)
    if skip > 0:
        poses = msg.poses[::skip]
        confs = msg.confs[::skip]
    else:
        poses = msg.poses
        confs = msg.confs

    if len(confs) == 0: return
    
    viridis = cm.get_cmap('viridis', 12)

    _range = max(confs) - min(confs)
    _range = 1 if _range == 0 else _range
    cmap = lambda x: viridis(
        (x - min(confs))/(_range)
    ) # normalize linearly
    
    marker_array = MarkerArray()
    for i, pose in enumerate(poses):
        marker = gripper_marker()
        marker.id = i
        marker.lifetime = rospy.Duration()

        if color == 'confs':
            marker.color.a = 0.5
            this_color = cmap(confs[i])
            marker.color.r = this_color[0]
            marker.color.g = this_color[1]
            marker.color.b = this_color[2]
        else:
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

        marker.header = msg.header
        marker.pose.orientation = pose.orientation
        marker.pose.position = pose.position
        marker_array.markers.append(marker)

    print("Publishing marker array.")
    marker_array_pub.publish(marker_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--color", type=str,
                        help="Which color of marker to make. Setting to 'confs' specifies that markers should be colormapped by grasp confidence. Setting to '(R, G, B, A)' sets markers to the given color. R, G, and B are floats in the range [0, 255], and A is in [0, 1].", default='confs')

    args, unknown = parser.parse_known_args()

    if args.color != 'confs':
        color = eval(args.color) # bit of an antipattern but I'm sure this won't come back to bite me
        
    rospy.init_node('publish_grasp_markers')
    rospy.Subscriber('grasps', Grasps, poses_cb, queue_size=1)

    rospy.loginfo('Ready to grasp markers.')
    rospy.spin()
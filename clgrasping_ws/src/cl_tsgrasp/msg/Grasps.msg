# Grasps.msg
# a message definition containing many filtered, plausible grasps
# identified by the grasp detection network.

Header header # sequence number, timestamp,frame ID
geometry_msgs/Pose[] poses # N grasp poses expressed in this frame
float32[] confs # N corresponding grasp confidences in [0, 1]
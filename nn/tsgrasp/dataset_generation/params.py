"""
params.py
Global parameters for rendering along a trajectory.
"""
TRAJ_PER_OBJECT = 3
FRAMES_PER_TRAJ = 30
GRASPDIR = "/home/playert/Research/GraspRefinement/data/contact_points/"
OBJDIR = "/home/playert/Research/acronym/data/shapenetsem/models-OBJ/simplified/"
# TRAJDIR = "/home/playert/Research/GraspRefinement/data/trajectory_with_pose/train"
TRAJDIR = "/home/playert/Research/GraspRefinement/data/trajectory_with_pose/val"
# TRAJDIR = "/home/playert/Research/GraspRefinement/data/test_labels"
SUCCESS_RADIUS = 0.0001

DEBUG = False

EPSILON = 0.0001 # detection radius for successful grasp contacts
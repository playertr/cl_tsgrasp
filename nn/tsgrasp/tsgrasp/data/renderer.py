# renderer.py
# class for rendering trajectories for training and testing

from dataclasses import dataclass
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' # for headless

import pyrender
import numpy as np
from omegaconf import DictConfig
import sys
import trimesh

class Renderer:
    obj_pose : np.ndarray
    
    def __init__(self, h5_path: str, obj_pose: np.ndarray, cfg:DictConfig):

        
        ## Dynamically import ACRONYM rendering utilities
        if cfg.acronym_repo not in sys.path[0]:
            sys.path.insert(0, cfg.acronym_repo) # :(
        from acronym_tools import load_mesh
        from scripts.acronym_render_observations import SceneRenderer, PyrenderScene
        
        ## Load mesh corresponding to this grasp file
        m = load_mesh(h5_path, mesh_root_dir=cfg.mesh_dir)

        self.scene = PyrenderScene()
        self.renderer = SceneRenderer(self.scene, width=cfg.width, height=cfg.height)

        ## Define floor and object
        support_mesh = trimesh.creation.cylinder(radius=100, height=0.1)

        ## Create Scene object and add the meshes
        self.scene = PyrenderScene()
        floor_pose = np.eye(4)
        floor_pose[2,3] = -0.05 # offset the floor so z=0 is the ground
        self.scene.add_object("support_object", support_mesh, pose=floor_pose, support=True)

        ## Define object location and add object to scene
        self.obj_pose = obj_pose
        self.scene.add_object("obj0", m, pose=self.obj_pose, support=False)

        ## Create rendering machinery: SceneRenderer and camera
        self.renderer = SceneRenderer(self.scene, width=300, height=300)

    def render_trajectory(self, poses: np.ndarray) -> np.ndarray:
        ## Step through trajectory and render images
        depths = []
        for i, camera_pose in enumerate(poses):
            cam_pose_world_frame = self.tf_cam_to_world(camera_pose)

            color, depth = render(self.renderer, camera_pose=cam_pose_world_frame)

            depths.append(depth)

            # import pyrender
            # viewer = pyrender.Viewer(self.scene.as_pyrender_scene())

            # import matplotlib.pyplot as plt
            # plt.imshow(depth)
            # plt.show()

            # ax = plt.axes(projection='3d')
            # ax.scatter3D(pc[:,0], pc[:,1], pc[:,2])
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z')
            # plt.show()

        # Turn lists of point clouds into one ndarray
        depths = np.stack(depths)
        return depths

    def tf_cam_to_world(self, pose: np.ndarray) -> np.ndarray:
        """Transform object_frame camera pose into a world_frame camera pose.

        Args:
            pose (np.ndarray): (4, 4) camera pose in object frame

        Returns:
            np.ndarray: (4, 4) camera pose in world frame
        """
        tf_cam_to_obj = pose
        tf_obj_to_world = self.obj_pose
        tf_cam_to_world = tf_obj_to_world @ tf_cam_to_obj

        return tf_cam_to_world

def render(acronym_r, camera_pose):
    """Render RGB/depth image of the scene.

    Args:
        camera_pose (np.ndarray): Homogenous 4x4 matrix describing the pose of the camera in scene coordinates.

    Returns:
        np.ndarray: Color image.
        np.ndarray: Depth image.
    """
    # Keep local to free OpenGl resources after use
    renderer = pyrender.OffscreenRenderer(
        viewport_width=acronym_r._width, viewport_height=acronym_r._height
    )

    # add camera to scene
    scene = acronym_r._scene.as_pyrender_scene()
    scene.add(acronym_r._camera, pose=camera_pose, name="camera")

    # render the full scene
    color, depth = renderer.render(scene)

    return color, depth

def inverse_homo(tf):
    """Compute inverse of homogeneous transformation matrix.

    The matrix should have entries
    [[R,       Rt]
     [0, 0, 0, 1]].
    """
    R = tf[0:3, 0:3]
    t = R.T @ tf[0:3, 3].reshape(3, 1)
    return np.block([
        [R.T, -t],
        [0, 0, 0, 1]
    ])

if __name__ == "__main__":

    @dataclass
    class Cfg:
        height=300
        width=300
        mesh_dir="/home/playert/Research/tsgrasp/data/obj/"
        acronym_repo="/home/playert/Research/acronym"
    cfg = Cfg()

    r = Renderer('/home/playert/Research/tsgrasp/data/dataset/4Shelves_2e22e4ffe7fa6b9998d5fc0473d00a1c_0.002564156568947413.h5',  obj_loc = np.array([0, 0, 0]), cfg=cfg)

    poses = [
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -5],
            [0, 0, 0, 1]
        ])
    ]
    pts = r.render_trajectory(poses)
    print(pts)
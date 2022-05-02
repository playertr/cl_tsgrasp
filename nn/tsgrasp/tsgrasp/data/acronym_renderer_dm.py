import h5py
import os
import numpy as np
import torch
from omegaconf import DictConfig
from tsgrasp.data.renderer import Renderer
from tsgrasp.data.augmentations import RandomJitter, RandomRotation, RandomRotationAboutZ
import trimesh
import copy

from tsgrasp.utils.utils import transform, compose

class TrajectoryDataset(torch.utils.data.Dataset):
    """
    A Dataset for:
        - loading grasps and meshes
        - rendering trajectories
        - augmenting data
    """

    AVAILABLE_SPLITS = ["train", "val", "test"]

    def __init__(self, cfg : DictConfig, split="train"):
        self.frames_per_traj = cfg.frames_per_traj
        self.root = os.path.join(cfg.dataroot, split)
        self.pts_per_frame = cfg.points_per_frame
        self.renderer_cfg = copy.deepcopy(cfg.renderer)
        self.renderer_cfg.mesh_dir = self.root
        self.min_pitch = cfg.min_pitch
        self.max_pitch = cfg.max_pitch

        self.split = split
        # Find the raw filepaths.
        if split in ["train", "val", "test"]:
            folder = os.path.join(self.root, "h5")
            self._paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        else:
            raise ValueError("Split %s not recognised" % split)

        augmentations = []
        if cfg.augmentations.add_random_jitter:
            augmentations.append(RandomJitter(sigma=cfg.augmentations.random_jitter_sigma))
        
        if cfg.augmentations.add_random_rotations:
            augmentations.append(RandomRotation())
        
        if cfg.augmentations.add_random_rotation_about_z:
            augmentations.append(RandomRotationAboutZ())

        self.augmentations = compose(*augmentations)

    def download(self):
        if len(os.listdir(self.raw_dir)) == 0:
            print(f"No files found in {self.raw_dir}. Please create the dataset by following tsgrasp/data/data_generation/README.md")

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            print(e)
            print("Trying random index.")
            return self.getitem(np.random.randint(low=0, high=self.__len__()))

    def getitem(self, idx):

        """Generate a random trajectory using the grasp information in this .h5 file."""

        path = self._paths[idx]
        with h5py.File(path) as ds:
            success = np.asarray(ds["grasps/qualities/flex/object_in_gripper"])
            grasp_tfs = np.asarray(ds["grasps/transforms"])
            grasp_contact_points = np.asarray(ds["grasps/contact_points"])

        obj_pose = np.eye(4)
        renderer = Renderer(h5_path=path, obj_pose=obj_pose, cfg=self.renderer_cfg)
        
        trajectory = self.make_trajectory(renderer.obj_pose[:3,3], num_frames=self.frames_per_traj, min_pitch=self.min_pitch, max_pitch=self.max_pitch)
        depth_ims = renderer.render_trajectory(trajectory)

        pcs = [depth_to_pointcloud(d) for d in depth_ims]

        if any(pc.shape[-2] != self.pts_per_frame for pc in pcs):
            raise ValueError("Not enough valid points rendered.")

        for i in range(len(pcs)):
            idxs = torch.randperm(len(pcs[i]), dtype=torch.int32, device='cpu')[:self.pts_per_frame].sort()[0].long()
            pcs[i] = pcs[i][idxs]

        positions = torch.Tensor(np.stack(pcs))

        # Get transformation from camera frame to object frame
        tf_from_cam_to_scene = trajectory.dot(
            trimesh.transformations.euler_matrix(np.pi, 0, 0)
        ) # camera_pose has a flipped z axis
        tf_from_scene_to_obj = inverse_homo(obj_pose)
        tfs_from_cam_to_obj = tf_from_scene_to_obj @ tf_from_cam_to_scene

        ## Remove all grasps with nan contact points
        invalid_idxs = np.isnan(grasp_contact_points).any(axis=(1,2))
        pos_grasp_tfs = grasp_tfs[success==1 & ~invalid_idxs]
        success = success[~invalid_idxs]
        grasp_contact_points = grasp_contact_points[~invalid_idxs]

        obj_frame_pos_grasp_tfs = pos_grasp_tfs # (2000, 4, 4)

        # transform object-frame grasp poses into camera frame
        # gross numpy broadcasting
        # https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python
        tfs_from_obj_to_cam = np.array([inverse_homo(t) for t in tfs_from_cam_to_obj])
        cam_frame_pos_grasp_tfs =  np.matmul(tfs_from_obj_to_cam[:,np.newaxis,:,:], obj_frame_pos_grasp_tfs[np.newaxis,:,:,:])
        cam_frame_pos_grasp_tfs = torch.Tensor(cam_frame_pos_grasp_tfs)

        pos_contact_pts_mesh = torch.Tensor(
            grasp_contact_points[np.where(success)].astype(np.float32)
        )
        T = tfs_from_cam_to_obj.shape[0]
        pos_contact_pts_cam = transform(
            pos_contact_pts_mesh.repeat(T, 1, 1, 1),
            torch.Tensor(tfs_from_obj_to_cam)
        )

        ## Transform all "camera" frame poses to be, specifically, in the frame of the most recent camera perspective.

        # Hopefully float precision ought to do ...
        tf_from_cam_i_to_cam_N = torch.Tensor(
            inverse_homo(tfs_from_cam_to_obj[-1]) @ tfs_from_cam_to_obj
        )
        positions = transform(
            positions, 
            tf_from_cam_i_to_cam_N
        )
        cam_frame_pos_grasp_tfs = transform(
            cam_frame_pos_grasp_tfs,
            tf_from_cam_i_to_cam_N
        )
        pos_contact_pts_cam = transform(
            pos_contact_pts_cam,
            tf_from_cam_i_to_cam_N
        )

        # from tsgrasp.utils.viz.viz import draw_grasps
        # draw_grasps(pts=positions[0], grasp_tfs=[], confs=[])
        # tfs = torch.stack([t[:50] for t in cam_frame_pos_grasp_tfs])
        # all_pos = positions.reshape(-1, 3)
        # tfs = tfs.reshape(-1, 4, 4)
        # draw_grasps(all_pos, tfs, confs=[1]*len(tfs))

        data = {
            "positions" : positions,
            "cam_frame_pos_grasp_tfs": cam_frame_pos_grasp_tfs,
            "pos_contact_pts_cam": pos_contact_pts_cam,
            "idx": idx * torch.ones(1)
        }
        return self.augmentations(data)

    @staticmethod
    def make_trajectory(obj_loc: np.ndarray, num_frames : float, min_pitch: float, max_pitch: int, seed=None):
        """Create a "random" trajectory that views the object.
        Initially, these trajectories will be circular orbits that always look directly at the object, at different elevation angles.

        A trajectory consists of camera poses in the object frame, i.e., 4x4 transforms which map from the camera coordinates to object coordinates. The camera coordinate system is defined with +X right, +Y down, +Z out and towards the scene (z > 0 for visible points).

        More creative distributions of trajectories, including ones in which
        the image-frame object center is not in the middle of the path, are 
        future work.

        Args:
            obj_loc (np.ndarray): location of the object. Origin? Center? Not sure.
            num_frames (int): number of positions along trajectory
            min_pitch (float): minimum pitch angle for uniform dist
            max pitch (float): max pitch angle for uniform dist
            seed (int, optional): RNG seed. Defaults to None.

        Returns:
            np.ndarray: array of 4x4 poses
        """
        
        np.random.seed(seed)

        # The pitch angle is a uniform random angle
        pitch = np.random.uniform(min_pitch, max_pitch)
        
        # A full circle is swept out, with a random initial yaw angle.
        yaw0 = np.random.uniform(0, 2*np.pi)
        
        d0 = np.random.uniform(1.5, 2.5) # Orbital distance
        
        yaws = np.linspace(yaw0, yaw0+2*np.pi * num_frames/30, num_frames)
        ds = d0 * np.ones_like(yaws)
        poses = []

        for i, (yaw, d) in enumerate(zip(yaws, ds)):
            camera_pose = look_at(
                loc=obj_loc,
                rotation=trimesh.transformations.euler_matrix(
                    pitch,
                    0,
                    yaw,
                ),
                distance=d,
            )
            poses.append(camera_pose)
        return np.array(poses)
        
    @property
    def raw_dir(self):
        return self.root

def look_at(loc: np.ndarray, rotation: np.ndarray, distance: float) -> np.ndarray:
    """Generate transform for a camera to keep a point in the camera's field of view. This is trimesh.scene.cameras.Camera.look_at, and it uses the pyrender convention for z-axis.

    NB! This function assumes the camera looks along its -z axis.

    Args:
        loc (np.ndarray): (3,) point to look at
        rotation (np.ndarray): (4,4) Rotation matrix for initial rotation
        distance (float): Distance from camera to point

    Returns:
        np.ndarray: (4,4) camera pose in world frame (tf world<--cam)
    """
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = rotation[:3, :3]
    focal_axis = rotation[:3,2] # + z axis. Should be a unit vector.
    
    assert np.abs(1 - np.linalg.norm(focal_axis)) < 0.001, "Rotation not orthonormal"

    pos = loc + distance * focal_axis
    cam_pose[:3, 3] = pos

    return cam_pose

def ragged_collate_fn(list_data):
    """Attempt to stack up each tensor in the dictionary. If they have incompatible sizes, return a list. """
    data = {}
    for key in list_data[0].keys():
        try: 
            data[key] = torch.stack([d[key] for d in list_data])
        except RuntimeError:
            data[key] = [d[key] for d in list_data]
    return data

def depth_to_pointcloud(depth, fov=np.pi/6):
    """Convert depth image to pointcloud given camera intrinsics, from acronym.scripts.acronym_render_observations

    Args:
        depth (np.ndarray): Depth image.

    Returns:
        np.ndarray: Point cloud.
    """
    fy = fx = 0.5 / np.tan(fov * 0.5)  # aspectRatio is one.

    height = depth.shape[0]
    width = depth.shape[1]

    mask = np.where(depth > 0)

    x = mask[1]
    y = mask[0]

    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = (y.astype(np.float32) - height * 0.5) / height

    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = depth[y, x]

    return np.vstack((world_x, world_y, world_z)).T

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
    from dataclasses import dataclass

    @dataclass
    class render_cfg:
        height=300
        width=300
        acronym_repo="/home/playert/Research/acronym"


    @dataclass
    class augment_cfg:
        add_random_jitter = True
        random_jitter_sigma = 0.001
        add_random_rotations = True

    @dataclass
    class Cfg:
        frames_per_traj=4
        dataroot="/home/playert/Research/tsgrasp/data/dataset"
        renderer=render_cfg()
        points_per_frame=45000
        augmentations=augment_cfg()

    cfg = Cfg()

    tds = TrajectoryDataset(cfg, split="train")
    tds2 = TrajectoryDataset(cfg, split="test")

    from torch.utils.data import DataLoader
    dl = DataLoader(tds, 
        batch_size=3, 
        num_workers=0, 
        collate_fn=ragged_collate_fn, persistent_workers=False,
        pin_memory=False, shuffle=True
        # sampler=RandomSampler(self.dataset_train, 
        #     num_samples=int(len(self.dataset_train)*self.data_cfg.data_proportion_per_epoch))
        )

    next(iter(dl))

    for i in range(60, len(tds)):
        item = tds[i]
        print(i)


    print("done")
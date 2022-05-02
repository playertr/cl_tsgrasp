# augmentations.py
# Data augmentations for point clouds and grasp labels

import torch
from tsgrasp.utils.utils import transform_unbatched
import numpy as np
import trimesh.transformations

class RandomJitter:
    """Add Guassian noise to `position` coordinates."""

    def __init__(self, sigma:float):
        self.sigma=sigma

    def __call__(self, data: dict):
        data['positions'] += self.sigma * torch.randn(*data['positions'].shape)
        return data

class RandomRotation:
    """Randomly rotate all spatial inputs about the origin."""

    def __call__(self, data: dict):
        tf = torch.eye(4)
        # https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices
        randrot, _R = torch.linalg.qr(torch.randn(3,3))
        tf[:3,:3] = randrot

        data['positions'] = transform_unbatched(data['positions'], tf)
        data['cam_frame_pos_grasp_tfs'] = transform_unbatched(
            data["cam_frame_pos_grasp_tfs"], tf)
        data["pos_contact_pts_cam"] = transform_unbatched(data["pos_contact_pts_cam"], tf)
        return data

class RandomRotationAboutZ:
    """Randomly yaw all spatial inputs about Z axis (preserving depth)."""

    def __call__(self, data: dict):
        yaw = np.random.uniform(0, 2*np.pi)
        tf = trimesh.transformations.euler_matrix(
            0,
            0,
            yaw,
        )
        tf = torch.Tensor(tf)

        data['positions'] = transform_unbatched(data['positions'], tf)
        data['cam_frame_pos_grasp_tfs'] = transform_unbatched(
            data["cam_frame_pos_grasp_tfs"], tf)
        data["pos_contact_pts_cam"] = transform_unbatched(data["pos_contact_pts_cam"], tf)
        return data
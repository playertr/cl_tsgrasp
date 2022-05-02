import h5py
import os
import numpy as np
import torch
from contact_torchnet.utils.mesh_utils.mesh_utils import create_gripper
from omegaconf import DictConfig
from functools import reduce

class AcronymVidDataset(torch.utils.data.Dataset):
    """
    A torch.geometric.Dataset for loading from files.
    """

    AVAILABLE_SPLITS = ["train", "val", "test"]

    def __init__(self, cfg : DictConfig, split="train"):

        self.root = cfg.dataroot
        self.pts_per_frame = cfg.points_per_frame
        self.grid_size = cfg.grid_size
        self.time_decimation_factor = cfg.time_decimation_factor

        # Find the raw filepaths. For now, we're doing no file-based preprocessing.
        if split in ["train", "val", "test"]:
            folder = os.path.join(self.root, split)
            self._paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')]
        else:
            raise ValueError("Split %s not recognised" % split)

        # Make a list of tuples (name, path) for each trajectory.
        self._trajectories = []
        for path in self._paths:
            with h5py.File(path) as ds:
                keys = {k for k in ds.keys() if k.startswith('pitch')} # a Set
            self._trajectories += [(k, path) for k in keys]

    def download(self):
        if len(os.listdir(self.raw_dir)) == 0:
            print(f"No files found in {self.raw_dir}. Please create the dataset using the scripts in GraspRefinement and ACRONYM.")

    def __len__(self):
        return len(self._trajectories) * 10

    def __getitem__(self, idx):
        traj_name, path = self._trajectories[idx // 10]
        time_step = idx % 10 # find time step

        with h5py.File(path) as ds:
            # `depth` and `labels` are (B, 300,300) arrays.
            # `depth` contains the depth video values, and `labels` is a binary mask indicating
            # whether a given pixel's 3D point is within data_generation.params.EPSILON of a positive grasp contact.
            depth = np.asarray(ds[traj_name]["depth"])
            labels = np.asarray(ds[traj_name]["grasp_labels"])
            # nearest_grasp_idxs = np.asarray(ds[traj_name]["nearest_grasp_idx"])
            success = np.asarray(ds["grasps/qualities/flex/object_in_gripper"])
            pos_grasp_tfs = np.asarray(ds["grasps/transforms"])[success==1]
            tfs_from_cam_to_obj = np.asarray(ds[traj_name]["tf_from_cam_to_obj"])
            grasp_contact_points = np.asarray(ds["grasps/contact_points"])

        ## Make data shorter via temporal decimation
        depth = depth[::self.time_decimation_factor, :, :]
        labels = labels[::self.time_decimation_factor, :, :]
        tfs_from_cam_to_obj = tfs_from_cam_to_obj[::self.time_decimation_factor,:,:]

        pcs = [depth_to_pointcloud(d) for d in depth]
        orig_pcs = torch.Tensor(pcs)
        labels = [label_frame for label_frame in labels]

        ## Downsample points
        b, npoint, dim = orig_pcs.shape
        orig_pcs_sampled = torch.empty(b, self.pts_per_frame, dim)
        for i in range(len(pcs)):
            idxs = torch.randperm(len(pcs[i]), dtype=torch.int32, device='cpu')[:self.pts_per_frame].sort()[0].long()
            
            orig_pcs_sampled[i] = orig_pcs[i][idxs]
            pcs[i] = pcs[i][idxs]
            labels[i] = labels[i].ravel()[idxs]

        ## Quantize points to grid
        positions = torch.Tensor(pcs) # save positions prior to truncation
        pcs = [(pc / self.grid_size).astype(int) for pc in pcs]

        pcs = multi_pointcloud_to_4d_coords(pcs)
        coords4d = torch.Tensor(pcs)

        ## Calculate gripper widths
        cp = grasp_contact_points
        finger_diffs = cp[:,0,:] - cp[:,1,:]
        pos_finger_diffs = finger_diffs[np.where(success)]
        offsets = np.linalg.norm(pos_finger_diffs, axis=-1) / 2

        ## Generate camera-frame grasp poses corresponding to closest points
        obj_frame_pos_grasp_tfs = pos_grasp_tfs # (2000, 4, 4)
        tfs_from_obj_to_cam = np.array([inverse_homo(t) for t in tfs_from_cam_to_obj])

        # gross numpy broadcasting
        # https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python
        cam_frame_pos_grasp_tfs =  np.matmul(tfs_from_obj_to_cam[:,np.newaxis,:,:], obj_frame_pos_grasp_tfs[np.newaxis,:,:,:])
        # (30, 2000, 4, 4)

        ## Generate camera-frame gripper control points
        control_pts, sym_control_pts, single_gripper_control_pts, sym_single_gripper_control_pts = camera_frame_control_pts(pos_grasp_tfs, cam_frame_pos_grasp_tfs, self.root)

        pos_contact_pts_mesh = grasp_contact_points[np.where(success)]
        data = {
            # "coordinates" : coords4d,
            # "positions" : positions,
            # "features" : torch.ones((len(coords4d), 1)),
            "labels" : torch.Tensor(labels)[time_step].view(-1, 1),
            # "pos_control_points" : torch.Tensor(control_pts),
            # "sym_pos_control_points" : torch.Tensor(sym_control_pts),
            "single_gripper_points" : torch.Tensor(single_gripper_control_pts),
            "sym_single_gripper_points": torch.Tensor(sym_single_gripper_control_pts),
            # "depth" : torch.Tensor(depth.astype(np.float32)), # np.float32 for endianness
            "all_pos" : orig_pcs_sampled[time_step],
            "cam_frame_pos_grasp_tfs": torch.Tensor(cam_frame_pos_grasp_tfs)[time_step],
            "pos_contact_pts_mesh": torch.Tensor(pos_contact_pts_mesh.astype(np.float32)),
            "pos_finger_diffs": torch.Tensor(offsets).reshape(-1, 1),
            "camera_pose": torch.Tensor(tfs_from_cam_to_obj.astype(np.float32))[time_step]
        }

        return data

    @property
    def raw_dir(self):
        return self.root

def ctn_collate_fn(list_data):
    """Attempt to stack up each tensor in the dictionary. If they have incompatible sizes,
    return a list. """
    data = {}
    for key in list_data[0].keys():
        try: 
            data[key] = torch.stack([d[key] for d in list_data])
        except RuntimeError:
            data[key] = [d[key] for d in list_data]
    return data

def minkowski_collate_fn(list_data):
    import MinkowskiEngine as ME
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["labels"] for d in list_data],
        dtype=torch.float32,
    )

    # pos_cp_list = [d["pos_control_points"] for d in list_data]
    # padded_stack(pos_cp_list)

    ## Each batch may have different numbers of ground truth grasps, resulting in ragged tensors. We require even, rectangular tensors for calculating the ADD-S loss, so we collate them into rectangular tensors.
    pos_control_points, sym_pos_control_points, gt_grasps_per_batch = \
        collate_control_points(
        batch = torch.arange(len(list_data)),
        time = torch.stack([d["coordinates"][:,0] for d in list_data]),
        pos_cp_list = [d["pos_control_points"] for d in list_data],
        sym_pos_cp_list = [d["sym_pos_control_points"] for d in list_data]
    )

    return {
        "coordinates": coordinates_batch,
        "positions": torch.stack([d["positions"] for d in list_data]),
        "features": features_batch,
        "labels": labels_batch,
        "pos_control_points": pos_control_points,
        "sym_pos_control_points": sym_pos_control_points,
        "single_gripper_points": list_data[0]['single_gripper_points'],
        "gt_grasps_per_batch": gt_grasps_per_batch,
        "cam_frame_pos_grasp_tfs": [d["cam_frame_pos_grasp_tfs"] for d in list_data]
    }

# def padded_stack(t_list) -> torch.Tensor:

#     # In what dimension do the tensors have different shapes?
#     shapes = [torch.Tensor(list(t.shape)).int() for t in t_list]

#     assert all(len(s) == len(shapes[0]) for s in shapes), "All input tensors must have the same number of dimensions."
    
#     dims_equal = [
#         all(s[i] == shapes[0][i] for s in shapes) # All shapes same in this dim
#         for i in range(len(shapes[0]))
#     ]

#     if all(dims_equal):
#         return torch.stack(t_list)

#     assert sum(dims_equal) == len(dims_equal)-1, "Input tensors may only vary in one dimension."

#     unequal_dim = torch.where(~torch.Tensor(dims_equal).bool())[0]

#     def pad(t, shape):
#         idxs = list(range())

#     padded = [pad(t) for t in t_list]

#     return torch.stack(padded)

def collate_control_points(batch, time, pos_cp_list, sym_pos_cp_list):
    """Pack the ground truth control points into a multidimensional dense
    tensor.

    If the control point tensors are not the same shape, then duplicate entries
    from the smaller ones until they are the same shape. Because the ADD-S loss
    considers the minimum distance from a ground truth grasp, duplicating ground
    truth grasps will not affect the grasp.
    """

    n_batch = len(batch.unique())
    n_time = len(time.unique())

    cp_shapes = [cp.shape for cp in pos_cp_list]
    gt_grasps_per_batch = [shape[1] for shape in cp_shapes]
    pos_control_points = torch.empty((
        n_batch,
        n_time,
        max(gt_grasps_per_batch),
        5,
        3
    ))
    sym_pos_control_points = torch.empty((
        n_batch,
        n_time,
        max(gt_grasps_per_batch),
        5,
        3
    ))

    # Pad control point tensors by repeating if different.
    if not reduce(np.array_equal, cp_shapes):
        for i in range(len(pos_cp_list)):
            if gt_grasps_per_batch[i] > 0:
                idxs = list(range(gt_grasps_per_batch[i]))
                idxs = idxs + [0]*(max(gt_grasps_per_batch) - gt_grasps_per_batch[i])
                pos_control_points[i] = pos_cp_list[i][:,idxs,:,:]
                sym_pos_control_points[i] = sym_pos_cp_list[i][:,idxs,:,:]
            # else: leave tensor uninitialized; do not use in ADD-S loss.
    else:
        for i in range(len(pos_cp_list)):
            if gt_grasps_per_batch[i] > 0:
                pos_control_points[i] = pos_cp_list[i]
                sym_pos_control_points[i] = sym_pos_cp_list[i]
            # else: leave tensor uninitialized; do not use in ADD-S loss.
            
    return pos_control_points, sym_pos_control_points, gt_grasps_per_batch

def camera_frame_control_pts(pos_grasp_tfs, cam_frame_grasp_tfs, dataroot):
    """Compute the control points corresponding to ground truth grasps, in the camera frame.

    Args:
        pos_grasp_tfs ([type]): (M, 4, 4) array of object-frame positive ground truth grasp transforms
        cam_frame_grasp_tfs ([type]): (T, 4, 4) array of camera-frame grasp poses
        dataroot (str): folder with gripper control point .npz files

    Returns:
        control_points [type]: camera-frame gournd truth control points
        sym_control_points [type]: camera-frame symmetric (switched 180 deg along wrist) control points
        gripper_control_points [type]: (5,3) array describing the gripper control points in the gripper frame. This is multiplied by predicted grasp transforms to get predicted control points.
    """
   
    ## Create 3D tensor of (num_good_grasps, 5, 3) control points
    gripper = create_gripper('panda', root_folder=dataroot)
    gripper_control_points = gripper.get_control_point_tensor(max(1, pos_grasp_tfs.shape[0]), use_tf=False) # num_gt_grasps x 5 x 3

    gripper_control_points_homog =  np.concatenate([gripper_control_points, np.ones((gripper_control_points.shape[0], gripper_control_points.shape[1], 1))], axis=2)  # b x 5 x 4

    control_points = np.matmul(gripper_control_points_homog, cam_frame_grasp_tfs.transpose((0, 1, 3, 2)))[:,:,:,:3]

    ## Create flipped symmetric control points for symmetric ADD-S loss
    sym_gripper_control_points = gripper.get_control_point_tensor(max(1, pos_grasp_tfs.shape[0]), symmetric=True, use_tf=False) # num_gt_grasps x 5 x 3

    sym_gripper_control_points_homog =  np.concatenate([sym_gripper_control_points, np.ones((sym_gripper_control_points.shape[0], sym_gripper_control_points.shape[1], 1))], axis=2)  # b x 5 x 4

    sym_control_points = np.matmul(sym_gripper_control_points_homog, cam_frame_grasp_tfs.transpose((0, 1, 3, 2)))[:,:,:,:3]

    return control_points, sym_control_points, gripper_control_points[0], sym_gripper_control_points[0]

def multi_pointcloud_to_4d_coords(pcs):
    """Convert a list of N (L, 3) ndarrays into a single (N*L, 4) ndarray, where the first dimension becomes the time dimension."""
    num_pcs = len(pcs)
    time_coords = np.repeat(np.arange(num_pcs), len(pcs[0]))
    
    pcs = np.concatenate(pcs)
    pcs = np.column_stack([time_coords, pcs])

    return pcs

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
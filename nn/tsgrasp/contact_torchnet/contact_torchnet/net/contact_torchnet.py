import torch
import torch.nn as nn
import torch.nn.functional as F
from ...Pointnet_Pointnet2_pytorch.models.pointnet2_utils import (PointNetFeaturePropagation, 
    PointNetSetAbstractionMsg, PointNetSetAbstraction)

from typing import Tuple

class ContactTorchNet(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.global_config = global_config

        model_config = global_config['MODEL']
        data_config = global_config['DATA']

        radius_list_0 = model_config['pointnet_sa_modules_msg'][0]['radius_list']
        radius_list_1 = model_config['pointnet_sa_modules_msg'][1]['radius_list']
        radius_list_2 = model_config['pointnet_sa_modules_msg'][2]['radius_list']
        
        nsample_list_0 = model_config['pointnet_sa_modules_msg'][0]['nsample_list']
        nsample_list_1 = model_config['pointnet_sa_modules_msg'][1]['nsample_list']
        nsample_list_2 = model_config['pointnet_sa_modules_msg'][2]['nsample_list']
        
        mlp_list_0 = model_config['pointnet_sa_modules_msg'][0]['mlp_list']
        mlp_list_1 = model_config['pointnet_sa_modules_msg'][1]['mlp_list']
        mlp_list_2 = model_config['pointnet_sa_modules_msg'][2]['mlp_list']
        
        npoint_0 = model_config['pointnet_sa_modules_msg'][0]['npoint']
        npoint_1 = model_config['pointnet_sa_modules_msg'][1]['npoint']
        npoint_2 = model_config['pointnet_sa_modules_msg'][2]['npoint']
        
        fp_mlp_0 = model_config['pointnet_fp_modules'][0]['mlp']
        fp_mlp_1 = model_config['pointnet_fp_modules'][1]['mlp']
        fp_mlp_2 = model_config['pointnet_fp_modules'][2]['mlp']

        input_normals = data_config['input_normals']
        offset_bins = data_config['labels']['offset_bins']
        joint_heads = model_config['joint_heads']

        if input_normals:
            raise NotImplementedError("Support for input normals not implemented yet.")

        if ('raw_num_points' in data_config and 
            data_config['raw_num_points'] != data_config['ndataset_points']):
            raise NotImplementedError("Farthest point sampling not implemented yet.")

        self.l1_sa = PointNetSetAbstractionMsg(
            npoint=npoint_0,
            radius_list=radius_list_0,
            nsample_list=nsample_list_0,
            in_channel=0,
            mlp_list=mlp_list_0,
        ) # out_channel: 64 + 128 + 128 = 320
        self.l2_sa = PointNetSetAbstractionMsg(
            npoint=npoint_1,
            radius_list=radius_list_1,
            nsample_list=nsample_list_1,
            in_channel=320,
            mlp_list=mlp_list_1,
        ) # out_channel: 640

        if 'asymmetric_model' in model_config and model_config['asymmetric_model']:
            self.l3_sa = PointNetSetAbstractionMsg(
                npoint=npoint_2,
                radius_list=radius_list_2,
                nsample_list=nsample_list_2,
                in_channel=640,
                mlp_list=mlp_list_2,
            ) # out_channel: 128 + 256 + 256 = 640
            self.l4_sa = PointNetSetAbstraction(
                npoint=None, # first three args not used if groupall==True
                radius=None,
                nsample=None,
                in_channel=643, # add 3 bc it concats with xyz
                mlp=model_config['pointnet_sa_module']['mlp'],
                group_all=model_config['pointnet_sa_module']['group_all']
            ) # out_channel: 1024

            # Feature Propagation layers
            # TODO: figure out why the in_channel is growing
            self.fp1 = PointNetFeaturePropagation(
                in_channel=1664, #1024,
                mlp=fp_mlp_0
            ) # out_channel: 256
            self.fp2 = PointNetFeaturePropagation(
                in_channel=896, #256,
                mlp=fp_mlp_1
            ) # out_channel: 128
            self.fp3 = PointNetFeaturePropagation(
                in_channel=448,#128,
                mlp=fp_mlp_2
            ) # out_channel: 128
        else:
            raise NotImplementedError("Symmetric model not implemented yet.")
        
        if joint_heads:
            raise NotImplementedError("Joint heads not implemented yet.")
        else:
            # Head for grasp direction -- normalization done in forward()
            self.baseline_dir_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.3),
                nn.Conv1d(128, 3, 1)
            )

            # Head for grasp approach -- G-S orthonormalization done in forward()
            self.approach_dir_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.3),
                nn.Conv1d(128, 3, 1)
            )

            # Head for grasp width
            if model_config['dir_vec_length_offset']:
                raise NotImplementedError("dir_vec_length_offset not implemented yet.")
            elif model_config['bin_offsets']:
                self.grasp_offset_head = nn.Sequential(
                    nn.Conv1d(128, 128, 1),
                    nn.BatchNorm1d(128),
                    nn.Dropout(p=0.3),
                    nn.Conv1d(128, len(offset_bins)-1, 1)
                )
            else:
                raise NotImplementedError("Only binned grasp offsets are implemented.")

            # Head for contact points
            self.binary_seg_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.5),
                nn.Conv1d(128, 1, 1)
            )

    def forward(self, xyz):
        """Identify grasp poses in a point cloud.

        Args:
            xyz (torch.Tensor): (b, TOTAL_POINTS, 3) point cloud.

        Returns:
            baseline_dir (torch.Tensor): (b, npoints, 3) gripper baseline direction in camera frame.
            binary_seg (torch.Tensor): (b, npoints) pointwise confidence that point is positive.
            grasp_offset (torch.Tensor): (b, npoints) gripper width (or half-width? Unsure.)
            approach_dir (torch.Tensor): (b, npoints, 3) gripper approach direction in camera frame.
            pred_points (torch.Tensor): (b, npoints, 3) positions of inferences (from xyz).

        """
        xyz = xyz.permute(0, 2, 1) # Pytorch_pointnet expects (B, C, N)

        l0_xyz = xyz
        l0_points = None # The "points" (features) are normals, and they're not provided.
        # We retain the xyz/points division from the original CGN in case we want to
        # add support for normals later.
        l1_xyz, l1_points = self.l1_sa(l0_xyz, l0_points)
        l2_xyz, l2_points = self.l2_sa(l1_xyz, l1_points)

        l3_xyz, l3_points = self.l3_sa(l2_xyz, l2_points)
        l4_xyz, l4_points = self.l4_sa(l3_xyz, l3_points)

        # Feature propagation
        l3_points = self.fp1(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp3(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_points = l1_points
        pred_points = l1_xyz

        # Grasp direction
        baseline_dir = self.baseline_dir_head(l0_points)
        baseline_dir = F.normalize(baseline_dir, dim=1)

        # Approach direction
        approach_dir = self.approach_dir_head(l0_points)
        approach_dir = F.normalize( # Gram-Schmidt
            approach_dir -                                          # (b, 3, npoints)
            (
                (baseline_dir * approach_dir).sum(dim=1).unsqueeze(1)  # (b, 1, npoints)
                * baseline_dir                                         # (b, 3, npoints)
            )
        )

        # Grasp width
        grasp_offset = self.grasp_offset_head(l0_points).transpose(1, 2) # (b, npoints, nbins)

        # Contact point classification
        binary_seg = self.binary_seg_head(l0_points).squeeze()

        # Transpose items from (b, 1 or 3, npoints) to (b, npoints, 1 or 3)
        baseline_dir = baseline_dir.permute(0, 2, 1)
        approach_dir = approach_dir.permute(0, 2, 1)
        pred_points = pred_points.permute(0, 2, 1)

        return baseline_dir, binary_seg, grasp_offset, approach_dir, pred_points 

    def loss(self, baseline_dir, binary_seg, grasp_offset, approach_dir, pred_points,
        baseline_dir_labels, offset_labels, approach_dir_labels, point_labels, cp_tensor, sym_cp_tensor):
        """Compute a loss consisting of BCE, ADS, and width losses.
        """
        # IF(USING_BINNED_OFFSETS):
        #   TRANSFORM_PREDS_AND_LABELS_INTO_BINNED_ONES()
        #   MAKE_SURE_TO_APPLY_BIN_WEIGHTS()
        if not self.global_config['MODEL']['bin_offsets']:
            raise NotImplementedError("Only binned offsets supported.")

        bin_vals = get_bin_vals(self.global_config).to(grasp_offset.device)
        thickness_pred_idx = torch.argmax(grasp_offset, dim=-1)
        # logits -> scalar binned width
        thickness_pred = bin_vals[ # for ADD-S loss
            thickness_pred_idx
        ].unsqueeze(-1)

        thickness_gt_idx = torch.argmin(torch.abs(bin_vals - offset_labels), dim=-1)
        # scalar -> one hot
        thickness_gt_one_hot = F.one_hot( # for width loss
            thickness_gt_idx,
            num_classes=10
        )
        # scalar -> scalar binned width
        thickness_gt = bin_vals[ # for ADD-S loss
            thickness_gt_idx
        ].unsqueeze(-1)

        ## BCE Loss
        bin_ce_loss = F.binary_cross_entropy_with_logits(binary_seg, point_labels.squeeze().float())

        ## ADD-S Loss
        adds_loss = add_s_loss(baseline_dir, torch.sigmoid(binary_seg), thickness_pred, approach_dir,
        pred_points, baseline_dir_labels, thickness_gt, approach_dir_labels, 
        point_labels, cp_tensor, sym_cp_tensor)
        
        ## Binned Width Loss
        offset_loss = self.offset_loss(grasp_offset, thickness_gt_one_hot.float())

        total_loss = 0

        if self.global_config['MODEL']['pred_contact_success']:
            total_loss += self.global_config['OPTIMIZER']['score_ce_loss_weight'] * bin_ce_loss
        if self.global_config['MODEL']['pred_contact_offset']:
            total_loss += self.global_config['OPTIMIZER']['offset_loss_weight'] * offset_loss
        if self.global_config['MODEL']['pred_grasps_adds']:
            total_loss += self.global_config['OPTIMIZER']['adds_loss_weight'] * adds_loss

        # Unimplemented losses -- not mentioned in paper
        if self.global_config['MODEL']['pred_contact_approach']:
            raise NotImplementedError
            total_loss += self.global_config['OPTIMIZER']['approach_cosine_loss_weight'] * approach_loss
        if self.global_config['MODEL']['pred_contact_base']:
            raise NotImplementedError
            total_loss += self.global_config['OPTIMIZER']['dir_cosine_loss_weight'] * dir_loss
        if self.global_config['MODEL']['pred_grasps_adds_gt2pred']:
            raise NotImplementedError
            total_loss += self.global_config['OPTIMIZER']['adds_gt2pred_loss_weight'] * adds_loss_gt2pred

        return total_loss

    def offset_loss(self, grasp_offset, offset_labels):
        if not( 
            self.global_config['MODEL']['bin_offsets'] and 
            self.global_config['LOSS']['offset_loss_type'] == 'sigmoid_cross_entropy'
        ):
            raise NotImplementedError("Only softmax cross entropy loss is supported.")

        bin_weights = torch.Tensor(
            self.global_config['DATA']['labels']['bin_weights']
        ).to(grasp_offset.device)
        offset_loss = F.binary_cross_entropy_with_logits(
            grasp_offset, offset_labels, reduction='none')
        offset_loss = torch.mean(
            bin_weights * offset_loss
        )
        return offset_loss
        
def add_s_loss(baseline_dir, binary_seg, grasp_offset, approach_dir,
    pred_points, baseline_dir_labels, offset_labels, approach_dir_labels, point_labels, cp_tensor, sym_cp_tensor):

    # Build 4x4 homogeneous grasp poses from directions
    pred_grasps = build_6dof_grasps(pred_points, baseline_dir, 
        approach_dir, grasp_offset)
    gt_grasps = build_6dof_grasps(pred_points, baseline_dir_labels, 
        approach_dir_labels, offset_labels)

    # Replace all negative grasps with matrices of 1e5 (causing high distance)
    # This should produce the same loss as if we indexed into only the positive
    # grasps, but produces a non-ragged tensor in the event that different
    # numbers of positive grasps are available in each batch.
    gt_grasps = torch.where(
        point_labels.unsqueeze(-1).repeat(1, 1, 4, 4),
        gt_grasps,
        1e5*torch.ones_like(gt_grasps)
    )

    # Fetch normal and reflected control point matrices
    cp_tensor = cp_tensor
    sym_cp_tensor = sym_cp_tensor

    # Transform the prototypical control point tensor into each of the gt grasp poses
    # Reshape arguments and return types to fit our transform API
    nbatch = gt_grasps.shape[0]
    npoints = gt_grasps.shape[1]
    gt_grasp_cp = transform(
        cp_tensor.repeat(nbatch*npoints, 1, 1), 
        gt_grasps.reshape(-1, 4, 4)
    ).reshape(nbatch, -1, 5, 3)

    # Repeat for symmetric control point tensor
    sym_gt_grasp_cp = transform(
        sym_cp_tensor.repeat(nbatch*npoints, 1, 1), 
        gt_grasps.reshape(-1, 4, 4)
    ).reshape(nbatch, -1, 5, 3)

    gt_cps = torch.cat(
        [gt_grasp_cp, sym_gt_grasp_cp], dim=1
    )

    # Transform the prototypical control point tensor into each of the predicted grasp poses
    pred_grasp_cp = transform(
        cp_tensor.repeat(nbatch*npoints, 1, 1), 
        pred_grasps.reshape(-1, 4, 4)
    ).reshape(nbatch, -1, 5, 3)
    # or list(map(lambda tf: transform(cp, tf), gt_grasps)) if dimension requires it

    # Mean-square the difference between the ground truth and predicted control points
    # Rows: predicted grasps. Cols: ground truth grasps.
    sq_dist = torch.mean(
        (
            pred_grasp_cp.unsqueeze(2) - gt_cps.unsqueeze(1) 
        )**2,
        dim=[-1, -2]
    )
    # Find the minimum distance for each predicted grasp
    neg_squared_adds_k, top_idcs = torch.topk(
        -1*sq_dist,
        k=1, sorted=False, dim=-1
    )

    # Filter away grasps with False labels
    min_adds = point_labels * torch.sqrt(-neg_squared_adds_k)

    # Take the mean minimum distance
    return torch.mean(binary_seg * torch.mean(min_adds, dim=-1))

def compute_labels_ragged_grasps(pred_points, gt_grasp_tfs, gt_contact_pts, gt_widths, cam_pose, radius=0.05):
    """Thin wrapper around `compute_labels` allowing for variable numbers of grasps per batch. 
    
    Args:
        pred_points (torch.Tensor): (b, n_pred_grasp, 3) inference positions (camera frame)
        gt_grasp_tfs (List[torch.Tensor]): b * (n_gt_grasp_i, 4, 4) ground truth grasp poses (scene frame)
        gt_contact_pts (List[torch.Tensor]): b * (n_gt_grasp_i, 4, 4) ground truth mesh contact points (scene frame). Left and right gripper contacts are stacked along the second-to-last dimension.
        gt_widths (List[torch.Tensor]): b * (n_gt_grasp_i, 1) ground truth grasp widths
        cam_pose (torch.Tensor): (b, 4, 4) camera poses
    """

    grasp_success_labels = [] 
    approach_dir_labels  = [] 
    baseline_dir_labels  = [] 
    grasp_width_labels   = []

    num_batches = pred_points.shape[0]
    for b in range(num_batches):
        grasp_success_label, approach_dir_label, baseline_dir_label, grasp_width_label = compute_labels(
            pred_points[b].unsqueeze(0), 
            gt_grasp_tfs[b].unsqueeze(0), 
            gt_contact_pts[b].unsqueeze(0), 
            gt_widths[b].unsqueeze(0), 
            cam_pose[b].unsqueeze(0), 
            radius
        )
        grasp_success_labels.append(grasp_success_label)
        approach_dir_labels.append(approach_dir_label)
        baseline_dir_labels.append(baseline_dir_label)
        grasp_width_labels.append(grasp_width_label)

    grasp_success_label = torch.cat(grasp_success_labels)
    approach_dir_label = torch.cat(approach_dir_labels)
    baseline_dir_label = torch.cat(baseline_dir_labels)
    grasp_width_label = torch.cat(grasp_width_labels)
    
    return grasp_success_label, approach_dir_label, baseline_dir_label, grasp_width_label

def compute_labels(pred_points, gt_grasp_tfs_cam, gt_contact_pts, gt_widths, cam_pose, radius=0.05):
    """Determine the single best grasp approach direction, baseline direction,
    and width associated with each pred_point considered during inference.

    Args:
        pred_points (torch.Tensor): (b, n_pred_grasp, 3) inference positions (camera frame)
        gt_grasp_tfs_cam (torch.Tensor): (b, n_gt_grasp, 4, 4) ground truth grasp poses (camera frame)
        gt_contact_pts (torch.Tensor): (b, n_gt_grasp, 2, 3) ground truth mesh contact points (scene frame). Left and right gripper contacts are stacked along the second-to-last dimension.
        gt_widths (torch.Tensor): (b, n_gt_grasp, 1) ground truth grasp widths
        cam_pose (torch.Tensor): (b, 4, 4) camera poses
    """

    # Transform gt grasp contact points into camera frame
    gt_contact_pts_cam = transform(gt_contact_pts, torch.linalg.inv(cam_pose))

    # Unstack gt_contact_pts: (b, n_gt_grasp, 2, 3) -> (b, n_gt_grasp*2, 3)
    b, n_gt_grasp, _2, _3 = gt_contact_pts_cam.shape
    gt_contact_pts_cam = gt_contact_pts_cam.reshape(b, n_gt_grasp*2, 3)

    # If no ground truth grasps are in this batch, then label all points as
    # zero and return nonsense (zero) grasp info labels to be ignored.
    if n_gt_grasp == 0:
        n_pred_grasp = pred_points.shape[1]
        grasp_success_label = torch.zeros((b, n_pred_grasp, 1), dtype=bool, device=pred_points.device)
        approach_dir_label = torch.zeros_like(pred_points)
        baseline_dir_label = torch.zeros_like(pred_points)
        grasp_width_label  = torch.zeros((b, n_pred_grasp, 1),device=pred_points.device)

        return grasp_success_label, approach_dir_label, baseline_dir_label, grasp_width_label

    # Determine the closest ground truth contact points to each pred_point
    sq_dists = torch.sum(
        (pred_points.unsqueeze(2) - gt_contact_pts_cam.unsqueeze(1))**2,
        dim=-1
    )

    # remove nan columns resulting from occasional nan contact points
    sq_dists[sq_dists.isnan()] = float('inf')

    top_ndists, top_idcs = torch.topk(
        -1*sq_dists,
        k=1, sorted=False, dim=-1
    )

    # Convert grasp poses into approach_dir and baseline_dir (and offset, given).
    gt_approach_dir, gt_baseline_dir = unbuild_6dof_grasps(gt_grasp_tfs_cam)

    # Find the approach dir, contact_dir, and width for the second half of the contact points
    gt_approach_dir = torch.cat([gt_approach_dir, gt_approach_dir], dim=-2)
    gt_baseline_dir = torch.cat([gt_baseline_dir, -gt_baseline_dir], dim=-2)
    gt_widths = torch.cat([gt_widths, gt_widths], dim=-2)

    # Select the grasp info from closest ground truth grasps
    approach_dir_label = torch.gather(gt_approach_dir, dim=-2, index=top_idcs.repeat(1, 1, 3))
    baseline_dir_label = torch.gather(gt_baseline_dir, dim=-2, index=top_idcs.repeat(1, 1, 3))
    grasp_width_label = torch.gather(gt_widths, dim=-2, index=top_idcs)

    # Label points as successful if they are close enough to contact points
    grasp_success_label = torch.less(
        -top_ndists,
        radius*radius
    )

    return grasp_success_label, approach_dir_label, baseline_dir_label, grasp_width_label

def transform(m: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """Transform the batched vector or pose matrix `m` by batch transform `tf`.

    Args:
        m (torch.Tensor): (b, ..., 3) or (b, ..., 4, 4) vector or pose.
        tf (torch.Tensor): (b, 4, 4) batched homogeneous transform matrix

    Returns:
        torch.Tensor: (b, ..., 3) or (b, ..., 4, 4) transformed poses or vectors.
    """
    # TODO: refactor batch handling by reshaping the arguments to transform_vec
    if m.shape[-1] == 3:
        return transform_vec(m, tf)
    elif m.shape[-2:] == (4, 4):
        return transform_mat(m, tf)

def transform_mat(pose: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """Transform homogenous transformation `pose` by homogenous transformation `tf`.

    Args:
        x (torch.Tensor): (b, ..., 4, 4) homogeneous pose matrix
        tf (torch.Tensor): (b, 4, 4) homogeneous transform matrix

    Returns:
        torch.Tensor: (b, ..., 4, 4) transformed poses.
    """

    assert all((
        len(tf.shape)==3,           # tf must be (b, 4, 4)
        tf.shape[1:]==(4, 4),       # tf must be (b, 4, 4)
        tf.shape[0]==pose.shape[0], # batch dimension must be same
    )), "Argument shapes are unsupported."
    x_dim = len(pose.shape)
    
    # Pad the dimension of tf for broadcasting.
    # E.g., if pose had shape (2, 3, 7, 4, 4), and tf had shape (2, 4, 4), then
    # we reshape tf to (2, 1, 1, 4, 4)
    tf = tf.reshape(tf.shape[0], *([1]*(x_dim-3)), 4, 4)

    return tf @ pose

def transform_vec(x: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """Transform 3D vector `x` by homogenous transformation `tf`.

    Args:
        x (torch.Tensor): (b, ..., 3) coordinates in R3
        tf (torch.Tensor): (b, 4, 4) homogeneous pose matrix

    Returns:
        torch.Tensor: (b, ..., 3) coordinates of transformed vectors.
    """

    x_dim = len(x.shape)
    assert all((
        len(tf.shape)==3,           # tf must be (b, 4, 4)
        tf.shape[1:]==(4, 4),       # tf must be (b, 4, 4)
        tf.shape[0]==x.shape[0],    # batch dimension must be same
        x_dim > 2                   # x must be a batched matrix/tensor
    )), "Argument shapes are unsupported."

    x_homog = torch.cat(
        [x, torch.ones(*x.shape[:-1], 1, device=x.device)], 
        dim=-1
    )
    
    # Pad the dimension of tf for broadcasting.
    # E.g., if x had shape (2, 3, 7, 3), and tf had shape (2, 4, 4), then
    # we reshape tf to (2, 1, 1, 4, 4)
    tf = tf.reshape(tf.shape[0], *([1]*(x_dim-3)), 4, 4)

    return (x_homog @ tf.transpose(-2, -1))[..., :3]

def build_6dof_grasps(contact_pts, baseline_dir, approach_dir, grasp_width, gripper_depth=0.1034):
    """Calculate the SE(3) transforms corresponding to each predicted
    coord/approach/baseline/grasp_width grasp.

    Args:
        contact_pts (torch.Tensor): (..., 3) contact points predicted
        baseline_dir (torch.Tensor): (..., 3) gripper baseline directions
        approach_dir (torch.Tensor): (..., 3) gripper approach directions
        grasp_width (torch.Tensor): (..., 1) gripper width
        gripper_depth (float, optional): depth of gripper. Defaults to 0.1034.

    Returns:
        pred_grasp_tfs (torch.Tensor): (b, T, n_pred_grasp, 4, 4) homogeneous grasp poses.
    """
    grasps_R = torch.stack([baseline_dir, torch.cross(approach_dir, baseline_dir), approach_dir], axis=-1)
    grasps_t = contact_pts + grasp_width/2 * baseline_dir - gripper_depth * approach_dir
    ones = torch.ones((*contact_pts.shape[:-1], 1, 1), device=contact_pts.device)
    zeros = torch.zeros((*contact_pts.shape[:-1], 1, 3), device=contact_pts.device)
    homog_vec = torch.cat([zeros, ones], dim=-1)

    pred_grasp_tfs = torch.cat([torch.cat([grasps_R, torch.unsqueeze(grasps_t, -1)], dim=-1), homog_vec], dim=-2)
    return pred_grasp_tfs

def unbuild_6dof_grasps(grasp_tfs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the approach direction and baseline directions corrsponding to
    each 6-DOF pose matrix.

    Args: grasp_tfs (torch.Tensor): (..., 4, 4) homogeneous gripper
        transforms

    Returns: Tuple[torch.Tensor, torch.Tensor]: (..., 3) gripper approach
        axis and gripper baseline direction
    """
    approach_dir = grasp_tfs[..., :3, 2] # z-axis of pose orientation
    baseline_dir = grasp_tfs[..., :3, 0] # x-axis of pose orientation
    return approach_dir, baseline_dir


def get_bin_vals(global_config):
    """
    Creates bin values for grasping widths according to bounds defined in config

    Arguments:
        global_config {dict} -- config

    Returns:
        torch.Tensor -- bin value tensor 
    """
    bins_bounds = torch.Tensor(global_config['DATA']['labels']['offset_bins'])
    if global_config['TEST']['bin_vals'] == 'max':
        bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
        bin_vals[-1] = bins_bounds[-1]
    elif global_config['TEST']['bin_vals'] == 'mean':
        bin_vals = bins_bounds[1:]
    else:
        raise NotImplementedError

    if not global_config['TEST']['allow_zero_margin']:
        bin_vals = torch.minimum(
            bin_vals, 
            torch.Tensor([
                global_config['DATA']['gripper_width']-global_config['TEST']['extra_opening']
            ])
        )
        
    return bin_vals
################################################################################
# CGN CODE
################################################################################

# def get_losses(pointclouds_pl, end_points, dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam, global_config):
#     """
#     Computes loss terms from pointclouds, network predictions and labels 

#     Arguments:
#         pointclouds_pl {tf.placeholder} -- bxNx3 input point clouds
#         end_points {dict[str:tf.variable]} -- endpoints of the network containing predictions
#         dir_labels_pc_cam {tf.variable} -- base direction labels in camera coordinates (bxNx3)
#         offset_labels_pc {tf.variable} -- grasp width labels (bxNx1) 
#         grasp_success_labels_pc {tf.variable} -- contact success labels (bxNx1) 
#         approach_labels_pc_cam {tf.variable} -- approach direction labels in camera coordinates (bxNx3)
#         global_config {dict} -- config dict 
        
#     Returns:
#         [dir_cosine_loss, bin_ce_loss, offset_loss, approach_cosine_loss, adds_loss, 
#         adds_loss_gt2pred, gt_control_points, pred_control_points, pos_grasps_in_view] -- All losses (not all are used for training)
#     """

#     grasp_dir_head = end_points['grasp_dir_head']
#     grasp_offset_head = end_points['grasp_offset_head']
#     approach_dir_head = end_points['approach_dir_head']

#     bin_weights = global_config['DATA']['labels']['bin_weights']
#     tf_bin_weights = tf.constant(bin_weights)
    
#     min_geom_loss_divisor = tf.constant(float(global_config['LOSS']['min_geom_loss_divisor'])) if 'min_geom_loss_divisor' in global_config['LOSS'] else tf.constant(1.)
#     pos_grasps_in_view = tf.math.maximum(tf.reduce_sum(grasp_success_labels_pc, axis=1), min_geom_loss_divisor)   

#     ### ADS Gripper PC Loss
#     # Snap each width to its nearest bin balue
#     if global_config['MODEL']['bin_offsets']:
#         thickness_pred = tf.gather_nd(get_bin_vals(global_config), tf.expand_dims(tf.argmax(grasp_offset_head, axis=2), axis=2))
#         thickness_gt = tf.gather_nd(get_bin_vals(global_config), tf.expand_dims(tf.argmax(offset_labels_pc, axis=2), axis=2))
#     else:
#         thickness_pred = grasp_offset_head[:,:,0]
#         thickness_gt = offset_labels_pc[:,:,0]
    
#     # Build 4x4 homogeneous grasp poses from directions
#     pred_grasps = build_6d_grasp(approach_dir_head, grasp_dir_head, pointclouds_pl, thickness_pred, use_tf=True) # b x num_point x 4 x 4
#     gt_grasps_proj = build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pointclouds_pl, thickness_gt, use_tf=True) # b x num_point x 4 x 4
    
#     # Replace all negative grasps with matrices of 1e5 (causing high distance)
#     pos_gt_grasps_proj = tf.where(
#         tf.broadcast_to(
#             tf.expand_dims(
#                 tf.expand_dims(
#                     tf.cast(grasp_success_labels_pc, tf.bool),
#                 2),
#             3), 
#         gt_grasps_proj.shape), 
#         gt_grasps_proj, 
#         tf.ones_like(gt_grasps_proj)*100000
#     )
#     # pos_gt_grasps_proj = tf.reshape(pos_gt_grasps_proj, (global_config['OPTIMIZER']['batch_size'], -1, 4, 4)) 

#     # Fetch normal and reflected control point matrices
#     gripper = mesh_utils.create_gripper('panda')
#     gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size']) # b x 5 x 3
#     sym_gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size'], symmetric=True)

#     # Make the control points into homogenous points by appending 1
#     gripper_control_points_homog =  tf.concat([gripper_control_points, tf.ones((global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], axis=2)  # b x 5 x 4
#     sym_gripper_control_points_homog =  tf.concat([sym_gripper_control_points, tf.ones((global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], axis=2)  # b x 5 x 4
    
#     # Transform the prototypical control point tensor into each of the predicted grasp poses
#     # only use per point pred grasps but not per point gt grasps
#     control_points = tf.keras.backend.repeat_elements(tf.expand_dims(gripper_control_points_homog,1), gt_grasps_proj.shape[1], axis=1)  # b x num_point x 5 x 4
#     sym_control_points = tf.keras.backend.repeat_elements(tf.expand_dims(sym_gripper_control_points_homog,1), gt_grasps_proj.shape[1], axis=1)  # b x num_point x 5 x 4
#     pred_control_points = tf.matmul(control_points, tf.transpose(pred_grasps, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_point x 5 x 3

#     ### Pred Grasp to GT Grasp ADD-S Loss
#     # Transform the prototypical control point tensor into each of the gt grasp poses
#     gt_control_points = tf.matmul(control_points, tf.transpose(pos_gt_grasps_proj, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_pos_grasp_point x 5 x 3
#     sym_gt_control_points = tf.matmul(sym_control_points, tf.transpose(pos_gt_grasps_proj, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_pos_grasp_point x 5 x 3

#     # Mean-square the difference between the ground truth and predicted control points
#     # The shape comment is wrong -- no tensor has shape num_pos_grasp_point, because the shape
#     # of the tensor is not changed by the tf.where command.
#     squared_add = tf.reduce_sum((tf.expand_dims(pred_control_points,2)-tf.expand_dims(gt_control_points,1))**2, axis=(3,4)) # b x num_point x num_pos_grasp_point x ( 5 x 3)
#     sym_squared_add = tf.reduce_sum((tf.expand_dims(pred_control_points,2)-tf.expand_dims(sym_gt_control_points,1))**2, axis=(3,4)) # b x num_point x num_pos_grasp_point x ( 5 x 3)

#     # symmetric ADD-S
#     neg_squared_adds = -tf.concat([squared_add,sym_squared_add], axis=2) # b x num_point x 2num_pos_grasp_point
#     neg_squared_adds_k = tf.math.top_k(neg_squared_adds, k=1, sorted=False)[0] # b x num_point
#     # If any pos grasp exists
#     min_adds = tf.minimum(
#         tf.reduce_sum(grasp_success_labels_pc, axis=1, keepdims=True), 
#         tf.ones_like(neg_squared_adds_k[:,:,0])) * tf.sqrt(-neg_squared_adds_k[:,:,0])
#         #tf.minimum(tf.sqrt(-neg_squared_adds_k), tf.ones_like(neg_squared_adds_k)) # b x num_point

#     adds_loss = tf.reduce_mean(end_points['binary_seg_pred'][:,:,0] * min_adds)

#     ### GT Grasp to pred Grasp ADD-S Loss
#     gt_control_points = tf.matmul(control_points, tf.transpose(gt_grasps_proj, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_pos_grasp_point x 5 x 3
#     sym_gt_control_points = tf.matmul(sym_control_points, tf.transpose(gt_grasps_proj, perm=[0, 1, 3, 2]))[:,:,:,:3] #  b x num_pos_grasp_point x 5 x 3

#     neg_squared_adds = -tf.reduce_sum((tf.expand_dims(pred_control_points,1)-tf.expand_dims(gt_control_points,2))**2, axis=(3,4)) # b x num_point x num_pos_grasp_point x ( 5 x 3)
#     neg_squared_adds_sym = -tf.reduce_sum((tf.expand_dims(pred_control_points,1)-tf.expand_dims(sym_gt_control_points,2))**2, axis=(3,4)) # b x num_point x num_pos_grasp_point x ( 5 x 3)

#     neg_squared_adds_k_gt2pred, pred_grasp_idcs = tf.math.top_k(neg_squared_adds, k=1, sorted=False) # b x num_pos_grasp_point
#     neg_squared_adds_k_sym_gt2pred, pred_grasp_sym_idcs = tf.math.top_k(neg_squared_adds_sym, k=1, sorted=False) # b x num_pos_grasp_point
#     pred_grasp_idcs_joined = tf.where(neg_squared_adds_k_gt2pred<neg_squared_adds_k_sym_gt2pred, pred_grasp_sym_idcs, pred_grasp_idcs)
#     min_adds_gt2pred = tf.minimum(-neg_squared_adds_k_gt2pred, -neg_squared_adds_k_sym_gt2pred) # b x num_pos_grasp_point x 1
#     # min_adds_gt2pred = tf.math.exp(-min_adds_gt2pred)
#     masked_min_adds_gt2pred = tf.multiply(min_adds_gt2pred[:,:,0], grasp_success_labels_pc)
    
#     batch_idcs = tf.meshgrid(tf.range(pred_grasp_idcs_joined.shape[1]), tf.range(pred_grasp_idcs_joined.shape[0]))
#     gather_idcs = tf.stack((batch_idcs[1],pred_grasp_idcs_joined[:,:,0]), axis=2)
#     nearest_pred_grasp_confidence = tf.gather_nd(end_points['binary_seg_pred'][:,:,0], gather_idcs)
#     adds_loss_gt2pred = tf.reduce_mean(tf.reduce_sum(nearest_pred_grasp_confidence*masked_min_adds_gt2pred, axis=1) / pos_grasps_in_view)
 
#     ### Grasp baseline Loss
#     cosine_distance = tf.constant(1.)-tf.reduce_sum(tf.multiply(dir_labels_pc_cam, grasp_dir_head),axis=2)
#     # only pass loss where we have labeled contacts near pc points 
#     masked_cosine_loss = tf.multiply(cosine_distance, grasp_success_labels_pc)
#     dir_cosine_loss = tf.reduce_mean(tf.reduce_sum(masked_cosine_loss, axis=1) / pos_grasps_in_view)

#     ### Grasp Approach Loss
#     approach_labels_orthog = tf.math.l2_normalize(approach_labels_pc_cam - tf.reduce_sum(tf.multiply(grasp_dir_head, approach_labels_pc_cam), axis=2, keepdims=True)*grasp_dir_head, axis=2)
#     cosine_distance_approach = tf.constant(1.)-tf.reduce_sum(tf.multiply(approach_labels_orthog, approach_dir_head), axis=2)
#     masked_approach_loss = tf.multiply(cosine_distance_approach, grasp_success_labels_pc)
#     approach_cosine_loss = tf.reduce_mean(tf.reduce_sum(masked_approach_loss, axis=1) / pos_grasps_in_view)

#     ### Grasp Offset/Thickness Loss
#     if global_config['MODEL']['bin_offsets']:
#         if global_config['LOSS']['offset_loss_type'] == 'softmax_cross_entropy':
#             offset_loss = tf.losses.softmax_cross_entropy(offset_labels_pc, grasp_offset_head)
#         else:
#             offset_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=offset_labels_pc, logits=grasp_offset_head)
            
#             if 'too_small_offset_pred_bin_factor' in global_config['LOSS'] and global_config['LOSS']['too_small_offset_pred_bin_factor']:
#                 too_small_offset_pred_bin_factor = tf.constant(global_config['LOSS']['too_small_offset_pred_bin_factor'], tf.float32)
#                 collision_weight = tf.math.cumsum(offset_labels_pc, axis=2, reverse=True)*too_small_offset_pred_bin_factor + tf.constant(1.)
#                 offset_loss = tf.multiply(collision_weight, offset_loss)

#             offset_loss = tf.reduce_mean(tf.multiply(tf.reshape(tf_bin_weights,(1,1,-1)), offset_loss),axis=2)
#     else:
#         offset_loss = (grasp_offset_head[:,:,0] - offset_labels_pc[:,:,0])**2
#     masked_offset_loss = tf.multiply(offset_loss, grasp_success_labels_pc)        
#     offset_loss = tf.reduce_mean(tf.reduce_sum(masked_offset_loss, axis=1) / pos_grasps_in_view)

#     ### Grasp Confidence Loss
#     bin_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(grasp_success_labels_pc,axis=2), logits=end_points['binary_seg_head'])
#     if 'topk_confidence' in global_config['LOSS'] and global_config['LOSS']['topk_confidence']:
#         bin_ce_loss,_ = tf.math.top_k(tf.squeeze(bin_ce_loss), k=global_config['LOSS']['topk_confidence'])
#     bin_ce_loss = tf.reduce_mean(bin_ce_loss)

#     return dir_cosine_loss, bin_ce_loss, offset_loss, approach_cosine_loss, adds_loss, adds_loss_gt2pred

# def multi_bin_labels(cont_labels, bin_boundaries):
#     """
#     Computes binned grasp width labels from continous labels and bin boundaries

#     Arguments:
#         cont_labels {tf.Variable} -- continouos labels
#         bin_boundaries {list} -- bin boundary values

#     Returns:
#         tf.Variable -- one/multi hot bin labels
#     """
#     bins = []
#     for b in range(len(bin_boundaries)-1):
#         bins.append(tf.math.logical_and(tf.greater_equal(cont_labels, bin_boundaries[b]), tf.less(cont_labels,bin_boundaries[b+1])))
#     multi_hot_labels = tf.concat(bins, axis=2)
#     multi_hot_labels = tf.cast(multi_hot_labels, tf.float32)

#     return multi_hot_labels

# def compute_labels_cgn(pos_contact_pts_mesh, pos_contact_dirs_mesh, pos_contact_approaches_mesh, pos_finger_diffs, pc_cam_pl, camera_pose_pl, global_config):
#     """
#     Project grasp labels defined on meshes onto rendered point cloud from a
#     camera pose via nearest neighbor contacts within a maximum radius. All
#     points without nearby successful grasp contacts are considered negativ
#     contact points.

#     Arguments:
#         pos_contact_pts_mesh {tf.constant} -- positive contact points on the mesh scene (Mx3)
#         pos_contact_dirs_mesh {tf.constant} -- respective contact base directions in the mesh scene (Mx3)
#         pos_contact_approaches_mesh {tf.constant} -- respective contact approach directions in the mesh scene (Mx3)
#         pos_finger_diffs {tf.constant} -- respective grasp widths in the mesh scene (Mx1)
#         pc_cam_pl {tf.placeholder} -- bxNx3 rendered point clouds
#         camera_pose_pl {tf.placeholder} -- bx4x4 camera poses
#         global_config {dict} -- global config

#     Returns:
#         [dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam] -- Per-point contact success labels and per-contact pose labels in rendered point cloud
#     """
#     ## Read projection parameters from config
#     label_config = global_config['DATA']['labels']
#     model_config = global_config['MODEL']

#     nsample = label_config['k']
#     radius = label_config['max_radius']
#     filter_z = label_config['filter_z']
#     z_val = label_config['z_val']

#     ## Transform object point cloud into camera frame
#     xyz_cam = pc_cam_pl[:,:,:3]
#     pad_homog = tf.ones((xyz_cam.shape[0],xyz_cam.shape[1], 1)) 
#     pc_mesh = tf.matmul(
#         tf.concat([xyz_cam, pad_homog], 2), 
#         tf.transpose(tf.linalg.inv(camera_pose_pl),perm=[0, 2, 1])
#     )[:,:,:3]

#     # Repeat the finger widths from (M) to (b, M)
#     contact_point_offsets_batch = tf.keras.backend.repeat_elements(
#         tf.expand_dims(pos_finger_diffs,0), 
#         pc_mesh.shape[0], 
#         axis=0
#     )

#     pad_homog2 = tf.ones(
#         (pc_mesh.shape[0], pos_contact_dirs_mesh.shape[0], 1)
#     ) 
#     # Repeat contact directions from (M, 3) to (b, M, 3)
#     contact_point_dirs_batch = tf.keras.backend.repeat_elements(
#         tf.expand_dims(pos_contact_dirs_mesh,0), 
#         pc_mesh.shape[0], 
#         axis=0
#     )

#     # Transform  contact directions into camera frame
#     contact_point_dirs_batch_cam = tf.matmul(
#         contact_point_dirs_batch, 
#         tf.transpose(camera_pose_pl[:,:3,:3], perm=[0, 2, 1])
#     )[:,:,:3]

#     # Repeat approach directions from (M, 3) to (b, M, 3)
#     pos_contact_approaches_batch = tf.keras.backend.repeat_elements(
#         tf.expand_dims(pos_contact_approaches_mesh,0), 
#         pc_mesh.shape[0], 
#         axis=0
#     )
#     # Transform approach directions into camera frame
#     pos_contact_approaches_batch_cam = tf.matmul(
#         pos_contact_approaches_batch, 
#         tf.transpose(camera_pose_pl[:,:3,:3], 
#         perm=[0, 2, 1])
#     )[:,:,:3]
    
#     # Repeat contact positions from (M, 3) to (b, M, 3)
#     contact_point_batch_mesh = tf.keras.backend.repeat_elements(
#         tf.expand_dims(pos_contact_pts_mesh,0), 
#         pc_mesh.shape[0], 
#         axis=0
#     )

#     # Transform  contact positions into camera frame
#     contact_point_batch_cam = tf.matmul(
#         tf.concat([contact_point_batch_mesh, pad_homog2], 2), 
#         tf.transpose(camera_pose_pl, perm=[0, 2, 1])
#     )[:,:,:3]

#     if filter_z:
#         # Remove contact points greater than max allowable z
#         dir_filter_passed = tf.keras.backend.repeat_elements(
#             tf.math.greater(
#                 contact_point_dirs_batch_cam[:,:,2:3], 
#                 tf.constant([z_val])
#             ), 
#             3, 
#             axis=2
#         )
#         contact_point_batch_mesh = tf.where(
#             dir_filter_passed, 
#             contact_point_batch_mesh, 
#             tf.ones_like(contact_point_batch_mesh)*100000
#         )

#     # Find distance from every point in the camera point cloud (rows) to every
#     # camera-frame contact point (cols)
#     squared_dists_all = tf.reduce_sum(
#         (
#             tf.expand_dims(contact_point_batch_cam,1)
#             - tf.expand_dims(xyz_cam,2)
#         )**2,
#         axis=3
#     )
#     # Identify distances to the top k closest contact points to every camera point
#     neg_squared_dists_k, close_contact_pt_idcs = tf.math.top_k(
#         -squared_dists_all, 
#         k=nsample, 
#         sorted=False
#     ) # (b, N, 1)
#     squared_dists_k = -neg_squared_dists_k

#     # Label every camera point True if it is close enough to a contact point
#     # Nearest neighbor mapping
#     grasp_success_labels_pc = tf.cast(
#         tf.less(tf.reduce_mean(squared_dists_k, axis=2), radius*radius), 
#         tf.float32
#     ) # (batch_size, num_point)

#     # Find the corresponding contact directions
#     grouped_dirs_pc_cam = group_point(
#         contact_point_dirs_batch_cam, 
#         close_contact_pt_idcs
#     )
#     # Find the corresponding approach directions
#     grouped_approaches_pc_cam = group_point(
#         pos_contact_approaches_batch_cam, 
#         close_contact_pt_idcs
#     )

#     # Find the corresponding gripper distances
#     grouped_offsets = group_point(
#         tf.expand_dims(contact_point_offsets_batch,2), 
#         close_contact_pt_idcs
#     )

#     ## Take the average approach direction, contact direction, and offset, if k > 1.
#     dir_labels_pc_cam = tf.math.l2_normalize(
#         tf.reduce_mean(grouped_dirs_pc_cam, axis=2),
#         axis=2
#     ) # (batch_size, num_point, 3)
#     approach_labels_pc_cam = tf.math.l2_normalize(
#         tf.reduce_mean(grouped_approaches_pc_cam, axis=2),
#         axis=2
#     ) # (batch_size, num_point, 3)
#     offset_labels_pc = tf.reduce_mean(grouped_offsets, axis=2)
        
#     return (
#         dir_labels_pc_cam, 
#         offset_labels_pc, 
#         grasp_success_labels_pc, 
#         approach_labels_pc_cam
#     )

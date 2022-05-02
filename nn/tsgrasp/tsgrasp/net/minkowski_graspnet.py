from functools import reduce
import logging
import torch.nn.functional as F
import torch
from omegaconf import DictConfig

from tsgrasp.net.modules.MinkowskiEngine import *

log = logging.getLogger(__name__)

class MinkowskiGraspNet(torch.nn.Module):
    def __init__(self, cfg : DictConfig):
        super().__init__()

        self.use_parallel_add_s = cfg.use_parallel_add_s
        self.add_s_loss_coeff = cfg.add_s_loss_coeff
        self.bce_loss_coeff = cfg.bce_loss_coeff
        self.width_loss_coeff = cfg.width_loss_coeff
        self.top_conf_quantile = cfg.top_confidence_quantile

        self.backbone = initialize_minkowski_unet(
            cfg.backbone_model_name, cfg.feature_dimension, cfg.backbone_out_dim, D=cfg.D
        )
        self.classification_head = nn.Sequential(
            nn.Conv1d(in_channels=cfg.backbone_out_dim, out_channels=128, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1),
        )
        self.baseline_dir_head = nn.Sequential(
            nn.Conv1d(in_channels=cfg.backbone_out_dim, out_channels=128, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1),
        )
        self.approach_dir_head = nn.Sequential(
            nn.Conv1d(in_channels=cfg.backbone_out_dim, out_channels=128, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1),
        )

        self.grasp_offset_head = nn.Sequential(
            nn.Conv1d(in_channels=cfg.backbone_out_dim, out_channels=128, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1),
        )

    def forward(self, sparse_x):
        """Accepts a Minkowski sparse tensor."""

        # B x 10 x ~300 x ~300
        torch.cuda.empty_cache()
        x = self.backbone(sparse_x)
        torch.cuda.empty_cache()

        x = x.slice(sparse_x).F
        torch.cuda.empty_cache()

        class_logits = self.classification_head(x.unsqueeze(-1)).squeeze(dim=-1)

        # Gram-Schmidt normalization
        baseline_dir = self.baseline_dir_head(x.unsqueeze(-1)).squeeze()
        baseline_dir = F.normalize(baseline_dir)

        approach_dir = self.approach_dir_head(x.unsqueeze(-1)).squeeze()
        dot_product =  torch.sum(baseline_dir*approach_dir, dim=-1, keepdim=True)
        approach_dir = F.normalize(approach_dir - dot_product*baseline_dir)

        grasp_offset = F.relu(self.grasp_offset_head(x.unsqueeze(-1)).squeeze(dim=-1))

        return class_logits, baseline_dir, approach_dir, grasp_offset

    def weighted_bce_loss(self, class_logits, labels):
        bce_loss = F.binary_cross_entropy_with_logits(
            class_logits,
            labels, reduction='none'
        ).ravel()

        if self.top_conf_quantile == 1.0:
            bce_loss = torch.mean(bce_loss)
        else:
            bce_loss = torch.mean(
                torch.topk(bce_loss, k=int(len(bce_loss) * self.top_conf_quantile))[0]
            )
    
        return self.bce_loss_coeff*bce_loss

    def weighted_add_s_loss(self, class_logits, labels, baseline_dir, approach_dir, grasp_offset, positions, pos_control_points, sym_pos_control_points, gt_grasps_per_batch, single_gripper_points):
        add_s = parallel_add_s_loss if self.use_parallel_add_s else sequential_add_s_loss
        add_s_loss = add_s(
            approach_dir = approach_dir, 
            baseline_dir = baseline_dir,
            positions = positions,
            pos_control_points = pos_control_points,
            sym_pos_control_points = sym_pos_control_points,
            gt_grasps_per_batch = gt_grasps_per_batch,
            single_gripper_points = single_gripper_points,
            labels = labels,
            logits = class_logits,
            grasp_width = grasp_offset
        )

        return self.add_s_loss_coeff*add_s_loss.squeeze()

    def weighted_width_loss(self, grasp_offset, grasp_offset_label, labels):
        # MSE loss for width
        # Including only non-nan grasp offset labels
        # Including only actual grasps
        offset_label_nan = grasp_offset_label.ravel().isnan()
        grasp_offset_label = torch.nan_to_num(grasp_offset_label)
        width_mse = (grasp_offset.ravel() - grasp_offset_label.ravel())**2
        width_mse = width_mse[labels.bool().ravel() & ~offset_label_nan]
        width_loss = torch.mean(width_mse) if len(width_mse > 0) else 0.0

        return self.width_loss_coeff*width_loss

def parallel_add_s_loss(approach_dir, baseline_dir, positions, pos_control_points, sym_pos_control_points, gt_grasps_per_batch, single_gripper_points, labels, logits, grasp_width) -> torch.Tensor:
    """Compute symmetric ADD-S loss from Contact-GraspNet, finding minimum
    control-point distances for all time and batch values in parallel."""
    
    if min(gt_grasps_per_batch) == 0:
        # one of the batches has no ground truth grasps; fall back to
        # non-parallelized ADD-S computation so that the other batches can still
        # contribute to the loss.
        return sequential_add_s_loss(approach_dir, baseline_dir, positions, pos_control_points, sym_pos_control_points, gt_grasps_per_batch, single_gripper_points, labels, logits, grasp_width)

    ## Package each grasp parameter P into a regular, dense Tensor of shape
    # (BATCH, TIME, N_PRED_GRASP, *P.shape)
    n_batch = pos_control_points.shape[0]
    n_time = pos_control_points.shape[1]
    approach_dir = approach_dir.view(n_batch, n_time, -1, 3)
    baseline_dir = baseline_dir.view(n_batch, n_time, -1, 3)
    positions = positions.view(n_batch, n_time, -1, 3)
    grasp_width = grasp_width.view(n_batch, n_time, -1, 1)

    ## Construct control points for the predicted grasps, where label is True.
    pred_cp = control_point_tensor(
        approach_dir,
        baseline_dir,
        positions,
        grasp_width,
        single_gripper_points
    )

    ## Find the minimum pairwise distance from each predicted grasp to the ground truth grasps.
    dists = torch.minimum(
        approx_min_dists(pred_cp, pos_control_points),
        approx_min_dists(pred_cp, sym_pos_control_points)
    )

    loss = torch.mean(
        torch.sigmoid(logits) * # weight by confidence
        labels *                # only backprop positives
        dists.view(-1)          # weight by distance
    )
    return loss

def sequential_add_s_loss(approach_dir, baseline_dir, positions, pos_control_points, sym_pos_control_points, gt_grasps_per_batch, single_gripper_points, labels, logits, grasp_width) -> torch.Tensor:
    """Un-parallelized implementation of add_s_loss. Uses a loop instead of batch/time parallelization to reduce memory requirements. 

    Args:
        approach_dir (torch.Tensor): (b*T*n_pred_grasp, 3) gripper approach direction
        baseline_dir (torch.Tensor): (b*T*n_pred_grasp, 3) gripper baseline direction
        positions (torch.Tensor): (b, T, n_pred_grasp, 3) position of contact point
        pos_control_points (torch.Tensor): (b, T, n_gt_grasp, 5, 3) control point tensors
        sym_pos_control_points (torch.Tensor): (b, T, n_gt_grasp, 5, 3) symmetric control point tensors
        gt_grasps_per_batch (List[int]): numbers of ground truth grasps in each batch
        single_gripper_points (torch.Tensor): (5, 3) control point tensor for a single grasp
        labels (torch.Tensor): (b*T*n_pred_grasp, 1) ground truth contact point classifications
        logits (torch.Tensor): (b*T*n_pred_grasp, 1) network contact point confidence logits
        grasp_width ([type]): (b*T*n_pred_grasp, 1) network grasp width/offset predictions

    Returns:
        torch.Tensor: [description]
    """

    ## Package each grasp parameter P into a regular, dense Tensor of shape
    # (BATCH, TIME, N_PRED_GRASP, *P.shape)
    n_batch = pos_control_points.shape[0]
    n_time = pos_control_points.shape[1]
    approach_dir = approach_dir.view(n_batch, n_time, -1, 3)
    baseline_dir = baseline_dir.view(n_batch, n_time, -1, 3)
    positions = positions.view(n_batch, n_time, -1, 3)
    grasp_width = grasp_width.view(n_batch, n_time, -1, 1)

    ## Construct control points for the predicted grasps, where label is True.
    pred_cp = control_point_tensor(
        approach_dir,
        baseline_dir,
        positions,
        grasp_width,
        single_gripper_points
    )

    logits = logits.view((n_batch, n_time, -1))
    labels = labels.view((n_batch, n_time, -1))
    loss = torch.zeros(1, device=approach_dir.device)
    for b in range(n_batch):
        
        if gt_grasps_per_batch[b] == 0:
            # This has no ground truth grasps; do not backprop ADD-S loss for this batch.
            loss += 0.0
            break

        for t in range(n_time):
            ## Find the minimum pairwise distance from each predicted grasp to the ground truth grasps.

            dists = torch.minimum(
                approx_min_dists(
                    pred_cp[b][t][None, None, :], 
                    pos_control_points[b][t][None, None, :]
                ),
                approx_min_dists(
                pred_cp[b][t][None, None, :], 
                sym_pos_control_points[b][t][None, None, :]
                )
            )

            loss += torch.mean(
                torch.sigmoid(logits[b][t]) *   # weight by confidence
                labels[b][t] *                  # only backprop positives
                dists.view(-1)                  # weight by distance
            )
    return loss / (n_batch * n_time)

def approx_min_dists(pred_cp, gt_cp):
    """Find the approximate minimum average distance of each
    control-point-tensor in `pred_cp` from any of the control-point-tensors in
    `gt_cp`.

    Approximates distance by finding the mean three-dimensional coordinate of
    each tensor, so that the M x N pairwise lookup takes up one-fifth of the
    memory as comparing all control-point-tensors in full, avoiding the creation
    of a (B, T, N, M, 5, 3) matrix.
    
    Once the mean-closest tensor is found for each point, the full matrix L2
    distance is returned.

    Args:
        pred_cp (torch.Tensor): (B, T, N, 5, 3) Tensor of control points
        gt_cp (torch.Tensor): (B, T, M, 5, 3) Tensor of ground truth control points
    """
    # Take the mean 3-vector from each 5x3 control point Tensor
    m_pred_cp = pred_cp.mean(dim=3) # (B, T, N, 3)
    m_gt_cp = gt_cp.mean(dim=3)     # (B, T, M, 3)

    # Find the squared L2 distance between all N pred and M gt means.
    # Find the index of the ground truth grasp minimizing the L2 distance.
    approx_sq_dists = torch.sum((m_pred_cp.unsqueeze(3) - m_gt_cp.unsqueeze(2))**2, dim=4)  # (B, T, N, M)
    best_idxs = torch.topk(-approx_sq_dists, k=1, sorted=False, dim=3)[1]   # (B, T, N, 1)

    # Select the full 5x3 matrix corresponding to each minimum-distance grasp.
    # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
    closest_gt_cps = torch.gather(gt_cp, dim=2, index=best_idxs.unsqueeze(4).repeat(1, 1, 1, 5, 3)) # (B, T, N, 5, 3)

    # Find the matrix L2 distances
    best_l2_dists = torch.sqrt(torch.sum((pred_cp - closest_gt_cps)**2, dim=(3,4))) # (B, T, N)

    return best_l2_dists

def control_point_tensor(approach_dirs, baseline_dirs, positions, grasp_widths, gripper_pts):
    """Construct an (N, 5, 3) Tensor of "gripper control points". From Contact-GraspNet.

    Each of the N grasps is represented by five 3-D control points.

    Args:
        approach_dirs (torch.Tensor): (B, T, N, 3) set of unit approach directions.
        baseline_dirs (torch.Tensor): (B, T, N, 3) ser of unit gripper axis directions.
        positions (torch.Tensor): (B, T, N, 3) set of gripper contact points.
        grasp_widths (torch.Tensor): (B, T, N) set of grasp widths
        gripper_pts (torch.Tensor): (5,3) Tensor of gripper-frame points
    """
    # Retrieve 6-DOF transformations
    grasp_tfs = build_6dof_grasps(
        contact_pts=positions,
        baseline_dir=baseline_dirs,
        approach_dir=approach_dirs,
        grasp_width=grasp_widths
    )
    # Transform the gripper-frame points into camera frame
    gripper_pts = torch.cat([
        gripper_pts, 
        torch.ones((len(gripper_pts), 1), device=approach_dirs.device)],
        dim=1
    ) # make (5, 4) stack of homogeneous vectors

    pred_cp = torch.matmul(
        gripper_pts, 
        torch.transpose(grasp_tfs, 4, 3)
    )[...,:3]

    return pred_cp

def build_6dof_grasps(contact_pts, baseline_dir, approach_dir, grasp_width, gripper_depth=0.1034):
    """Calculate the SE(3) transforms corresponding to each predicted coord/approach/baseline/grasp_width grasp.

    Args:
        contact_pts ([type]): (b, T, n_pred_grasp, 3) contact points predicted
        baseline_dir ([type]): (b, T, n_pred_grasp, 3) gripper baseline directions
        approach_dir ([type]): (b, T, n_pred_grasp, 3) gripper approach directions
        grasp_width ([type]): (b, T, n_pred_grasp, 3) gripper width
        gripper_depth (float, optional): depth of gripper. Defaults to 0.1034.

    Returns:
        pred_grasp_tfs (torch.Tensor): (b, T, n_pred_grasp, 4, 4) homogeneous grasp poses.
    """
    grasps_R = torch.stack([baseline_dir, torch.cross(approach_dir, baseline_dir), approach_dir], axis=4)
    grasps_t = contact_pts + grasp_width/2 * baseline_dir - gripper_depth * approach_dir
    ones = torch.ones((*contact_pts.shape[:3], 1, 1), device=contact_pts.device)
    zeros = torch.zeros((*contact_pts.shape[:3], 1, 3), device=contact_pts.device)
    homog_vec = torch.cat([zeros, ones], axis=4)

    pred_grasp_tfs = torch.cat([torch.cat([grasps_R, torch.unsqueeze(grasps_t, 4)], dim=4), homog_vec], dim=3)
    return pred_grasp_tfs

import abc
import torch
import torch.nn.functional as F
from typing import Tuple

class TSGraspSuper(abc.ABC, torch.nn.Module):
    """A metaclass for temporal grasping networks.
    
    This class defines functions to calculate labels or losses for a single
    frame from a depth video. These functions are in a metaclass so that the identical functions can be inherited by different implementations of grasp synthesis networks that identify grasp from depth video."""

    @staticmethod
    @torch.inference_mode()
    def closest_points(xyz: torch.Tensor, contact_points: torch.Tensor) -> torch.Tensor:
        """Find point in the set of gripper contact points that is closest to each point in the point cloud `xyz`. Return the distance and index.

        Args:
            xyz (torch.Tensor): (..., V, 3) point cloud from depth cam
            contact_points (torch.Tensor): (..., W, 3) set of gripper contact points from positive grasps

        Returns:
            min_dists (torch.Tensor): (..., V, 1) 
            min_idc (torch.Tensor): the corresponding indices of the closest points
        """
        dists = torch.cdist(xyz, contact_points, p=2, 
            # compute_mode="donot_use_mm_for_euclid_dist" # MUCH slower
        ) # (V, W)
        min_dists, min_idcs = torch.min(dists, dim=-1)
        return min_dists, min_idcs 

    @staticmethod
    @torch.jit.script
    def class_loss(pt_logits: torch.Tensor, pt_labels: torch.Tensor, conf_quantile: float) -> torch.Tensor:
        """Return the pointwise classification loss.

        This loss is implemented as a binary cross-entropy loss, where only the top `conf_quantile` proportion of losses contribute to the loss.

        Args:
            pt_logits (torch.Tensor): (V, 1) vector of float logits in (-inf, inf)
            pt_labels (torch.Tensor): (V, 1) vector of class labels, 0.0 or 1.0.

        Returns:
            bce_loss (torch.Tensor): (1,) scalar top-k BCE loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            pt_logits,
            pt_labels, reduction='none'
        ).ravel()

        if conf_quantile == 1.0:
            bce_loss = torch.mean(bce_loss)
        else:
            bce_loss = torch.mean(
                torch.topk(bce_loss, k=int(len(bce_loss) * conf_quantile))[0]
            )

        return bce_loss

    @staticmethod
    def width_loss(width_preds: torch.Tensor, width_labels: torch.Tensor, pt_labels: torch.Tensor) -> torch.Tensor:
        """Return the regression loss for the grasp width.

        Args:
            width_preds (torch.Tensor): (V, 1) vector of width predictions
            width_labels (torch.Tensor): (V, 1) vector of width labels
            pt_labels (torch.Tensor): (V, 1) vector of contact point labels

        Returns:
            width_loss (torch.Tensor): (1,) scalar width loss
        """
        # MSE loss for width
        # Including only actual grasps
        width_mse = (width_preds.ravel() - width_labels.ravel())**2
        width_mse = width_mse[pt_labels.bool().ravel()]
        width_loss = torch.mean(width_mse) if len(width_mse > 0) else torch.zeros(1, device=width_preds.device).squeeze()

        return width_loss

    @staticmethod
    def add_s_loss(approach_dir, baseline_dir, positions, grasp_width, logits, labels, pos_grasp_tfs) -> torch.Tensor:
        """Computes ADD-S loss: the mean, minimum distance of predicted grasp control point matrices from ground truth control point matrices.

        Reduces memory by identifying the "nearest" grasps using mean R^3 distance as an approximation to R^{5x3} distance.

        Args:
            approach_dir (torch.Tensor): (V, 3) network gripper approach direction
            baseline_dir (torch.Tensor): (V, 3) network gripper baseline direction
            positions (torch.Tensor): (V, 3) position of contact points used by network
            grasp_width (torch.Tensor): (V, 1) network grasp width/offset predictions
            logits (torch.Tensor): (V, 1) network contact point confidence logits
            labels (torch.Tensor): (V, 1) ground truth contact point classifications
            pos_grasp_tfs (torch.Tensor): (W, 4, 4) poses of ground truth positive grasps in the camera frame

        Returns:
            torch.Tensor: (1,) ADD-S scalar loss
        """

        if len(pos_grasp_tfs) == 0: 
            return torch.zeros(1, device=approach_dir.device).squeeze()

        ## Retrieve ground truth control point tensors
        pos_control_points = TSGraspSuper.control_point_tensors(pos_grasp_tfs)
        sym_pos_control_points = TSGraspSuper.control_point_tensors(pos_grasp_tfs, symmetric=True)

        ## Retrieve 6-DOF predicted grasp transformations
        pred_grasp_tfs = TSGraspSuper.build_6dof_grasps(
            contact_pts=positions,
            baseline_dir=baseline_dir,
            approach_dir=approach_dir,
            grasp_width=grasp_width
        )

        ## Construct control points for the predicted grasps, where label is True.
        pred_cp = TSGraspSuper.control_point_tensors(
            pred_grasp_tfs, symmetric=False
        ) # (V, 5, 3)

        ## Find the minimum pairwise distance from each predicted grasp to the ground truth grasps.
        dists = torch.minimum(
            TSGraspSuper.approx_min_dists(
                pred_cp, 
                pos_control_points
            ),
            TSGraspSuper.approx_min_dists(
            pred_cp, 
            sym_pos_control_points
            )
        ) # (V,)

        adds_loss = torch.mean(
            torch.sigmoid(logits) *         # weight by confidence
            labels *                        # only backprop positives
            dists.unsqueeze(-1)             # weight by distance
        )
        return adds_loss

    @staticmethod
    def control_point_tensors(grasp_tfs: torch.Tensor, symmetric: bool=False):
        """Construct an (N, 5, 3) Tensor of "gripper control points" using the provided grasp poses. From Contact-GraspNet.

        Each of the N grasps is represented by five 3-D control points.

        Args:
            grasp_tfs (torch.Tensor): (N, 4, 4) homogeneous grasp poses
            symmetric (bool): whether to use the symmetric (swapped fingers) tensor
        """
        
        # Get control point tensor for single grasp
        if symmetric:
            gripper_pts = TSGraspSuper.sym_single_control_point_tensor(device=grasp_tfs.device)
        else:
            gripper_pts = TSGraspSuper.single_control_point_tensor(device=grasp_tfs.device)

        # make (5, 4) stack of homogeneous vectors
        gripper_pts = torch.cat([
            gripper_pts, 
            torch.ones((len(gripper_pts), 1), device=grasp_tfs.device)],
            dim=1
        )

        # Transform the gripper-frame control points into camera frame
        pred_cp = torch.matmul(
            gripper_pts, 
            torch.transpose(grasp_tfs, -1, -2)
        )[...,:3]

        return pred_cp

    @staticmethod
    def build_6dof_grasps(contact_pts, baseline_dir, approach_dir, grasp_width):
        """Calculate the SE(3) transforms corresponding to each predicted coord/approach/baseline/grasp_width grasp.

        Args:
            contact_pts (torch.Tensor): (..., N, 3) contact points predicted
            baseline_dir (torch.Tensor): (..., N, 3) gripper baseline directions
            approach_dir (torch.Tensor): (..., N, 3) gripper approach directions
            grasp_width (torch.Tensor): (..., N, 3) gripper width

        Returns:
            pred_grasp_tfs (torch.Tensor): (..., N, 4, 4) homogeneous grasp poses.
        """
        shape = contact_pts.shape[:-1]
        gripper_depth = TSGraspSuper.gripper_depth()
        grasps_R = torch.stack([baseline_dir, torch.cross(approach_dir, baseline_dir), approach_dir], axis=-1)
        grasps_t = contact_pts + grasp_width/2 * baseline_dir - gripper_depth * approach_dir
        ones = torch.ones((*shape, 1, 1), device=contact_pts.device)
        zeros = torch.zeros((*shape, 1, 3), device=contact_pts.device)
        homog_vec = torch.cat([zeros, ones], axis=-1)

        pred_grasp_tfs = torch.cat([
            torch.cat([grasps_R, grasps_t.unsqueeze(-1)], dim=-1), 
            homog_vec
        ], dim=-2)
        return pred_grasp_tfs

    @staticmethod
    def unbuild_6dof_grasps(grasp_tfs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the approach direction and baseline directions corrsponding to each homogeneous 6-DOF pose matrix.

        Args: grasp_tfs (torch.Tensor): (N, 4, 4) homogeneous gripper
            transforms

        Returns: Tuple[torch.Tensor, torch.Tensor]: (N, 3) gripper approach
            axis and gripper baseline direction
        """
        approach_dir = grasp_tfs[..., :3, 2] # z-axis of pose orientation
        baseline_dir = grasp_tfs[..., :3, 0] # x-axis of pose orientation
        return approach_dir, baseline_dir


    @staticmethod
    def approx_min_dists(pred_cp: torch.Tensor, gt_cp: torch.Tensor) -> torch.Tensor:
        """Find the approximate minimum average distance of each
        control-point-tensor in `pred_cp` from any of the control-point-tensors in `gt_cp`.

        Approximates distance by finding the mean three-dimensional coordinate of each tensor, so that the M x N pairwise lookup takes up one-fifth of the memory as comparing all control-point-tensors in full, avoiding the creation of a (N, M, 5, 3) matrix.
        
        Once the mean-closest tensor is found for each point, the full matrix L2
        distance is returned.

        Args:
            pred_cp (torch.Tensor): (..., N, 5, 3) Tensor of control points
            gt_cp (torch.Tensor): (..., M, 5, 3) Tensor of ground truth control points

        Returns:
            best_l2_dists (torch.Tensor): (..., N,) matrix of closest matrix-L2 distances
        """
        # Take the mean 3-vector from each 5x3 control point Tensor
        m_pred_cp = pred_cp.mean(dim=-2) # (N, 3)
        m_gt_cp = gt_cp.mean(dim=-2)     # (M, 3)

        # Find the squared L2 distance between all N pred and M gt means.
        # Find the index of the ground truth grasp minimizing the L2 distance.
        dists, best_idxs = TSGraspSuper.closest_points(
            m_pred_cp, m_gt_cp
        ) # (N, 1)

        # Select the full 5x3 matrix corresponding to each minimum-distance grasp.
        closest_gt_cps = multi_dim_index(gt_cp, best_idxs)

        # Find the matrix L2 distances
        best_l2_dists = torch.sqrt(
            torch.sum((pred_cp - closest_gt_cps)**2, 
            dim=(-1, -2))
        ) # (N,)

        return best_l2_dists

    @staticmethod
    def class_width_labels(
        contact_pts: torch.Tensor,
        positions: torch.Tensor,
        pt_radius: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the pointwise class labels and width labels for a "singly batched" set of grasp contact information.

        Args:
            contact_pts (torch.Tensor): (T, N_GT_GRASPS, 2, 3) positive grasp contact points.
            positions (torch.Tensor): (T, N_PTS, 3) camera-frame point cloud
            pt_radius (float): distance threshold for labeling points

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (T, N_PTS, 1) boolean point label and (T, N_PTS, 1) float width label.
        """

        T, N_PTS, _3 = positions.shape
        if contact_pts.shape[1] == 0:
            return torch.zeros((T, N_PTS, 1), dtype=bool, device=contact_pts.device), None

        ## Calculate all grasp widths
        grasp_widths = torch.linalg.norm(
            contact_pts[...,0, :] - contact_pts[...,1, :],
            dim=-1
        ).unsqueeze(-1) # (T, N_GT_GRASPS_i, 1)

        ## Stack contact points
        # Concatenate the left and right contact points
        contact_pts = torch.cat([
            contact_pts[:,:,0,:], contact_pts[:,:,1,:]
            ], dim=-2
        ) # (T, 2N_GT_GRASPS_i, 3)

        ## Class labels
        # Find closest ground truth points for labeling
        dists, min_idcs = TSGraspSuper.closest_points(
            positions,
            contact_pts
        )
        pt_labels = (dists < pt_radius).unsqueeze(-1)

        ## Width labels
        # the minimum index could belong to a right-finger gripper point
        width_labels = grasp_widths.repeat(1, 2, 1)
        width_labels = torch.gather(width_labels, dim=1, index=min_idcs.unsqueeze(-1))

        return pt_labels, width_labels

    @staticmethod
    def losses(
        class_logits: torch.Tensor,
        baseline_dir: torch.Tensor,
        approach_dir: torch.Tensor,
        grasp_offset: torch.Tensor,
        positions: torch.Tensor,
        grasp_tfs: torch.Tensor,
        top_conf_quantile: float,
        pt_labels: torch.Tensor,
        width_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the ADD-S loss, width loss, and classification loss for a set of predicted grasp information and labels.

        Args:
            class_logits (torch.Tensor): (T, N_PTS, 1) network class logit output
            baseline_dir (torch.Tensor): (T, N_PTS, 3) predicted gripper baseline direction
            approach_dir (torch.Tensor): (T, N_PTS, 3) predicted gripper approach direction
            grasp_offset (torch.Tensor): (T, N_PTS, 1) predicted grasp width
            positions (torch.Tensor): (T, N_PTS, 3) contact points used for prediction, from the camera point cloud.
            grasp_tfs (torch.Tensor): (T, N_GT_GRASPS, 4, 4) ground truth grasp poses
            top_conf_quantile (float): proportion of classification losses to backpropagate
            pt_labels (torch.Tensor): (T, N_PTS, 1) contact point labels
            width_labels (torch.Tensor): (T, N_PTS, 1) gripper width labels

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ADD-S loss, width loss, and classification loss (scalars).
        """

        if pt_labels.sum() > 0: # sometimes no grasps
            ## ADD-S Loss
            add_s_loss = TSGraspSuper.add_s_loss(
                approach_dir=approach_dir,
                baseline_dir=baseline_dir,
                positions=positions,
                grasp_width=grasp_offset,
                logits=class_logits,
                labels=pt_labels,
                pos_grasp_tfs=grasp_tfs
            )
            
            ## Width loss
            width_loss = TSGraspSuper.width_loss(
                width_preds=grasp_offset,
                width_labels=width_labels,
                pt_labels=pt_labels
            )

        else:
            add_s_loss = torch.zeros(1, device=class_logits.device).squeeze()
            width_loss = torch.zeros(1, device=class_logits.device).squeeze()

        ## Classification loss
        class_loss = TSGraspSuper.class_loss(
            pt_logits=class_logits,
            pt_labels=pt_labels.float(),
            conf_quantile=top_conf_quantile
        )

        return add_s_loss, width_loss, class_loss



























    ## Hard-coded gripper parameters.
    # These should probably be read in from an external parameter file, but writing them as hard-coded static methods is clear and simple.
    @staticmethod
    def single_control_point_tensor(device: torch.device):
        """Retrieve the (5,3) matrix of control points for a single gripper."""
        return torch.Tensor([[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [ 5.2687433e-02, -5.9955313e-05,  7.5273141e-02],
       [-5.2687433e-02,  5.9955313e-05,  7.5273141e-02],
       [ 5.2687433e-02, -5.9955313e-05,  1.0527314e-01],
       [-5.2687433e-02,  5.9955313e-05,  1.0527314e-01]]).to(device)

    @staticmethod
    def sym_single_control_point_tensor(device: torch.device):
        """Retrieve the (5,3) matrix of control points for a single gripper, with the two fingers switched."""
        return torch.Tensor([[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [-5.2687433e-02,  5.9955313e-05,  7.5273141e-02],
       [ 5.2687433e-02, -5.9955313e-05,  7.5273141e-02],
       [-5.2687433e-02,  5.9955313e-05,  1.0527314e-01],
       [ 5.2687433e-02, -5.9955313e-05,  1.0527314e-01]]).to(device)

    @staticmethod
    def gripper_depth():
        """Retrieve the distance from the contact point to the gripper wrist along the gripper approach direction."""
        return 0.1034

def multi_dim_index(T: torch.Tensor, idx: torch.LongTensor) -> torch.Tensor:
    """Index into tensor `T`  with multi-dimensional index tensor `idx`.
    This operation automates the lengthy `torch.gather` call that is needed to index into a tensor of shape (*shape, B, ...) with an index of shape (*shape, C).

    > foo = torch.rand(1234, 10, 3, 7)
    > baz = torch.randint(10, size=(1234, 2))
    > bar = multi_dim_index(foo, bar)               # (1234, 3, 3, 7)

    Args:
        T (torch.Tensor): (*shape, B, *endshape) input tensor 
        idx (torch.LongTensor): (*shape, C)

    Returns:
        torch.Tensor: [description]
    """
    endshape = T.shape[idx.ndim:]
    return torch.gather(
        T,
        dim=idx.ndim-1,
        index=idx.reshape(*idx.shape, *[1]*(len(endshape))
        ).repeat(*[1]*idx.ndim, *endshape)
    )
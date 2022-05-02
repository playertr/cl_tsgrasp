import torch
import torch.nn as nn
import torch.nn.functional as F
from contact_torchnet.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import (PointNetFeaturePropagation, 
    PointNetSetAbstractionMsg, PointNetSetAbstraction)
from tsgrasp.net.tsgrasp_super import TSGraspSuper

from typing import Tuple

class ContactTorchNet(TSGraspSuper):
    def __init__(self, global_config):
        super().__init__()
        self.pt_radius = global_config.pt_radius
        self.top_conf_quantile = global_config.top_confidence_quantile
        self.add_s_loss_coeff = global_config.add_s_loss_coeff
        self.bce_loss_coeff = global_config.bce_loss_coeff
        self.width_loss_coeff = global_config.width_loss_coeff

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
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.3),
                nn.Conv1d(128, 3, 1)
            )

            # Head for grasp approach -- G-S orthonormalization done in forward()
            self.approach_dir_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.3),
                nn.Conv1d(128, 3, 1)
            )

            # Head for grasp width
            # if model_config['dir_vec_length_offset']:
            #     raise NotImplementedError("dir_vec_length_offset not implemented yet.")
            # elif model_config['bin_offsets']:
            #     self.grasp_offset_head = nn.Sequential(
            #         nn.Conv1d(128, 128, 1),
            #         nn.BatchNorm1d(128),
            #         nn.Dropout(p=0.3),
            #         nn.Conv1d(128, len(offset_bins)-1, 1)
            #     )
            # else:
            #     raise NotImplementedError("Only binned grasp offsets are implemented.")
            self.grasp_offset_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.3),
                nn.Conv1d(128, 1, 1)
            )

            # Head for contact points
            self.binary_seg_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.ReLU(),
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
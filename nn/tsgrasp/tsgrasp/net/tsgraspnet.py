import logging
import torch.nn.functional as F
import torch
from omegaconf import DictConfig

from tsgrasp.net.modules.MinkowskiEngine import *
from tsgrasp.net.tsgrasp_super import TSGraspSuper

log = logging.getLogger(__name__)

class TSGraspNet(TSGraspSuper):
    def __init__(self, cfg : DictConfig):
        super().__init__()

        self.add_s_loss_coeff = cfg.add_s_loss_coeff
        self.bce_loss_coeff = cfg.bce_loss_coeff
        self.width_loss_coeff = cfg.width_loss_coeff
        self.top_conf_quantile = cfg.top_confidence_quantile
        self.pt_radius = cfg.pt_radius
        self.grid_size = cfg.grid_size

        self.backbone = initialize_minkowski_unet(
            cfg.backbone_model_name, cfg.feature_dimension, cfg.backbone_out_dim, D=cfg.D, conv1_kernel_size=cfg.conv1_kernel_size,
            dilations=cfg.dilations
        )
        self.binary_seg_head = nn.Sequential(
                nn.Conv1d(cfg.backbone_out_dim, 128, 1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.5),
                nn.Conv1d(128, 1, 1)
        )
        self.baseline_dir_head = nn.Sequential(
                nn.Conv1d(cfg.backbone_out_dim, 128, 1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.3),
                nn.Conv1d(128, 3, 1)
        )
        self.approach_dir_head = nn.Sequential(
                nn.Conv1d(cfg.backbone_out_dim, 128, 1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.3),
                nn.Conv1d(128, 3, 1)
        )
        self.grasp_offset_head = nn.Sequential(
                nn.Conv1d(cfg.backbone_out_dim, 128, 1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.3),
                nn.Conv1d(128, 1, 1)
        )

    def forward(self, sparse_x):
        """Accepts a Minkowski sparse tensor."""

        # B x 10 x ~300 x ~300
        torch.cuda.empty_cache()
        x = self.backbone(sparse_x)
        torch.cuda.empty_cache()

        x = x.slice(sparse_x).F
        torch.cuda.empty_cache()

        class_logits = self.binary_seg_head(x.unsqueeze(-1)).squeeze(dim=-1)

        # Gram-Schmidt normalization
        baseline_dir = self.baseline_dir_head(x.unsqueeze(-1)).squeeze()
        baseline_dir = F.normalize(baseline_dir)

        approach_dir = self.approach_dir_head(x.unsqueeze(-1)).squeeze()
        dot_product =  torch.sum(baseline_dir*approach_dir, dim=-1, keepdim=True)
        approach_dir = F.normalize(approach_dir - dot_product*baseline_dir)

        grasp_offset = F.relu(self.grasp_offset_head(x.unsqueeze(-1)).squeeze(dim=-1))

        return class_logits, baseline_dir, approach_dir, grasp_offset


def multi_pointcloud_to_4d_coords(pcs):
    """Given a (B, T, N, 3) dense tensor, get the (B*T*N,5) tensor of coordinates across batch, time, x, y, and z dimensions."""
    # this should be true at the end:
    # torch.equal(pcs[coords[:,0], coords[:,1], torch.arange(90_000).repeat(80)], coords[:,-3:].int())
    B, T, N, D = pcs.shape
    batch_coords = torch.arange(B, dtype=pcs.dtype, device=pcs.device).repeat_interleave(T*N)
    time_coords = torch.arange(T, dtype=pcs.dtype, device=pcs.device).repeat_interleave(N).repeat(B)
    coords = torch.column_stack([batch_coords, time_coords, pcs.reshape(-1, D)])

    return coords

def discretize(positions: torch.Tensor, grid_size: float) -> torch.Tensor:
    """Truncate each position to an integer grid."""
    return (positions / grid_size).int()

def unweighted_sum(coords: torch.Tensor):
    """Create a feature vector from a coordinate array, so each 
    row's feature is the number of rows that share that coordinate."""
    
    unique_coords, idcs, counts = coords.unique(dim=0, return_counts=True, return_inverse=True)
    features = counts[idcs]
    return features.reshape(-1, 1).to(torch.float32)

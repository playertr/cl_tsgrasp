import hydra
from omegaconf import DictConfig
import h5py
from tsgrasp.utils.viz.viz import animate_grasps_from_outputs
from tqdm import tqdm
import torch
import os
from tsgrasp.net.tsgrasp_super import TSGraspSuper

@hydra.main(config_path="../conf", config_name="scripts/calculate_temporal_consistency")
def main(cfg : DictConfig):
    with h5py.File(cfg.output_path, 'r') as ds:
        visualize_grasps(ds, cfg)

def visualize_grasps(ds, cfg):
    """Take in the grasps from an h5py output dataset and plot them."""

    for example_num in tqdm(ds['outputs']):

        results = ds['outputs'][example_num]

        device = torch.device('cuda') if cfg.gpus > 0 else torch.device('cpu')
        # Load relevant confidence and position information from each model
        class_logits =torch.Tensor(results['class_logits']).to(device)
        baseline_dir = torch.Tensor(results['baseline_dir']).to(device)
        approach_dir = torch.Tensor(results['approach_dir']).to(device)
        grasp_offset = torch.Tensor(results['grasp_offset']).to(device)
        positions = torch.Tensor(results['positions']).to(device)
        pt_labels = torch.Tensor(results['pt_labels']).to(device)

        gt_contact_pts = torch.Tensor(results['gt_contact_pts']).to(device)
        if gt_contact_pts.dim() != 3:
            continue

        T, N_GT_CP, _3 = gt_contact_pts.shape

        for t in range(T):
            # pairwise distance: every row is a gt contact point,
            # every column is a predicted positive point cloud point
            pos_idcs_t = (class_logits[t] > 0).squeeze()
            pos_pts_t = positions[t][pos_idcs_t]

            if len(pos_pts_t) > 0:
                dists, idcs = TSGraspSuper.closest_points(
                    gt_contact_pts[t], pos_pts_t
                )

                # Find every gt point that actually was covered
                covered_labels = dists < cfg.covered_distance
            else:
                covered_labels = torch.zeros(size=(N_GT_CP,), dtype=bool, device=gt_contact_pts.device)

            if t > 0:
                # If I covered a point last time, am I 
                # more likely to cover the point this time?
                # covered_previously_and_now = covered_labels & prev_covered_labels
                consistency = covered_labels[prev_covered_labels].float().mean()
                print(f"consistency: {consistency}")
            
            prev_covered_labels = covered_labels

if __name__ == "__main__":
    main()
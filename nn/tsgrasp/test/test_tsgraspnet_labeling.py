from omegaconf import OmegaConf
from hydra.utils import instantiate
import pytest
import torch

@pytest.fixture
def cfg():
    s = """
    training:
        gpus: 1
        batch_size: 8
        max_epochs: 100
        optimizer:
            learning_rate: 0.00025
            lr_decay: 0.99
        save_animations: true
        use_wandb: false
        wandb:
            project: TSGrasp
            experiment: test
            notes: null
    model:
        _target_: tsgrasp.net.lit_tsgraspnet.LitTSGraspNet
        model_cfg:
            backbone_model_name: MinkUNet14A
            D: 4
            backbone_out_dim: 128
            add_s_loss_coeff: 10
            bce_loss_coeff: 1
            width_loss_coeff: 1
            top_confidence_quantile: 0.25
            use_parallel_add_s: false
            feature_dimension: 1
            pt_radius: 0.005
            grid_size: 0.005
    data:
        _target_: tsgrasp.data.lit_acronymvid.LitAcronymvidDataset
        data_cfg:
            dataroot: /home/playert/Research/tsgrasp/data/acronymvid
            points_per_frame: 90000
            grid_size: 0.005
            num_workers: 4
            time_decimation_factor: 3
            data_proportion_per_epoch: 1
    """
    return OmegaConf.create(s)

def test_labeling(cfg):
    lds = instantiate(cfg.data, batch_size=1)
    litnet = instantiate(cfg.model, training_cfg=cfg.training)

    lds.setup()
    dl = lds.val_dataloader()

    for batch in dl:
        ## Unpack data from dictionary
        positions = batch['positions'] 
        # (B, T, N_PTS, 3) cam point cloud
        grasp_tfs = batch['cam_frame_pos_grasp_tfs'] 
        # (B,) list of (T, N_GT_GRASPS_i, 4, 4) tensors of homogeneous grasp poses (from positive grasps) in the camera frame
        contact_pts = batch['pos_contact_pts_cam'] 
        # (B,) list of (T, N_GT_GRASPS_i, 2, 3) tensors of gripper contact points (for left and right fingers) in the mesh frame

        grasp_widths = [
            torch.linalg.norm(
                cp[...,0, :] - cp[...,1, :],
                dim=-1
            ).unsqueeze(-1)
            for cp in contact_pts
        ]
        # (B,) list of 10, N_GT_GRASPS_i, 1)

        B, T, N_PTS, D = positions.shape

        for b in range(B):

            ## Compute labels
            pt_labels_b, width_labels_b = litnet.model.class_width_labels(
                contact_pts[b], positions[b], grasp_widths[b], 
                litnet.model.pt_radius
            )

            print(pt_labels_b[0].sum())

            if pt_labels_b[0].sum() > 0:
                plot_point_cloud(positions[b][0], pt_labels_b[0].squeeze().float(), contact_pts[b][0].reshape(-1, 3).numpy())

def plot_point_cloud(pts, confs, gt_cp):
    import trimesh
    pcl = trimesh.points.PointCloud(vertices=pts, r=100)
    cp = trimesh.points.PointCloud(vertices=gt_cp, r=100)
    pcl.visual.vertex_colors = trimesh.visual.interpolate(confs, color_map='viridis') # or color by depth: pts[:,2]

    scene = trimesh.Scene([pcl, cp])

    scene.show(viewer='gl')

    print('bp')

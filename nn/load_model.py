#! /home/playert/miniconda3/envs/tsgrasp/bin/python
# shebang is for the Python3 environment with the network dependencies

import sys, os
from pathlib import Path
pkg_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(pkg_root / 'nn' / 'tsgrasp'))

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

cfg_str = """
training:
  gpus: 1
  batch_size: 12
  max_epochs: 200
  optimizer:
    learning_rate: 0.00025
    lr_decay: 0.99
  animate_outputs: false
  make_sc_curve: false
  use_wandb: true
  wandb:
    project: TSGrasp
    experiment: tsgrasp_scene
    notes: Table scene data with random orbital yaw speed
model:
  _target_: tsgrasp.net.lit_tsgraspnet.LitTSGraspNet
  model_cfg:
    backbone_model_name: MinkUNet14A
    D: 4
    backbone_out_dim: 128
    add_s_loss_coeff: 10
    bce_loss_coeff: 1
    width_loss_coeff: 1
    top_confidence_quantile: 1.0
    feature_dimension: 1
    pt_radius: 0.005
    grid_size: 0.005
    conv1_kernel_size: 3
    dilations:
    - 1 1 1 1
data:
  _target_: tsgrasp.data.lit_scenerenderer_dm.LitTrajectoryDataset
  data_cfg:
    num_workers: 4
    data_proportion_per_epoch: 1
    dataroot: /scratch/playert/workdir/cgn_data
    frames_per_traj: 4
    points_per_frame: 45000
    min_pitch: 0.0
    max_pitch: 1.222
    scene_contacts_path: ${data.data_cfg.dataroot}/scene_contacts
    pc_augm:
      clip: 0.005
      occlusion_dropout_rate: 0.0
      occlusion_nclusters: 0
      sigma: 0.0
    depth_augm:
      clip: 0.005
      gaussian_kernel: 0
      sigma: 0.001

ckpt_path: tsgrasp_scene_4_random_yaw/model.ckpt
"""

def load_model():
    cfg = OmegaConf.create(cfg_str)
    pl_model = instantiate(cfg.model, training_cfg=cfg.training)
    path = pkg_root / 'nn' / 'ckpts' / cfg.ckpt_path
    print(f"Loading weights from {path}")
    pl_model.load_state_dict(torch.load(path)['state_dict'])
    return pl_model
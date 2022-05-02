To run this model, first set the tsgrasp commit via

```
git checkout 987280a1bde69cff1417c6fecee406dc585c4d8b
```

and use
```
cfg_str = """
training:
  gpus: 4
  batch_size: 1
  max_epochs: 100
  optimizer:
    learning_rate: 0.0002
    lr_decay: 0.95
  animate_outputs: false
  make_sc_curve: false
  use_wandb: false
  wandb:
    project: TSGrasp
    experiment: last_loss
    notes: Loss applied only to the last frame. 6 frames per traj.
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
  _target_: tsgrasp.data.lit_acronym_renderer_dm.LitTrajectoryDataset
  data_cfg:
    num_workers: 8
    data_proportion_per_epoch: 1
    dataroot: /scratch/playert/workdir/tsgrasp/data/dataset
    frames_per_traj: 6
    points_per_frame: 45000
    augmentations:
      add_random_jitter: true
      random_jitter_sigma: 0.0001
      add_random_rotations: true
    renderer:
      height: 300
      width: 300
      mesh_dir: ${hydra:runtime.cwd}/data/obj/
      acronym_repo: /scratch/playert/workdir/acronym

ckpt_path: last_loss/model.ckpt
"""
```

artifact playertr/TSGrasp/model-1lu0nng2:v45
To run this model, first set the tsgrasp commit via

```
git checkout 5a469b01682d86d3bdd3db965b672ff806d01e79
```

and use
```
cfg_str = """
training:
  gpus: 4
  batch_size: 3
  max_epochs: 100
  optimizer:
    learning_rate: 0.00025
    lr_decay: 0.99
  animate_outputs: false
  make_sc_curve: false
  use_wandb: false
  wandb:
    project: TSGrasp
    experiment: tsgrasp_1_15
    notes: Jan 15 run with all-frame loss, object frame, and object-category test/train.
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
    num_workers: 4
    data_proportion_per_epoch: 1
    dataroot: ${hydra:runtime.cwd}/data/dataset
    frames_per_traj: 8
    points_per_frame: 45000
    augmentations:
      add_random_jitter: true
      random_jitter_sigma: 0.0001
      add_random_rotations: true
    renderer:
      height: 300
      width: 300
      acronym_repo: /scratch/playert/workdir/acronym
      mesh_dir: ${hydra:runtime.cwd}/data/obj/

ckpt_path: tsgrasp_1_15/model.ckpt
"""
```

artifact playertr/TSGrasp/model-1lu0nng2:v45
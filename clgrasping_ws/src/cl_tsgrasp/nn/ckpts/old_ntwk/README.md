To run this model, first set the tsgrasp commit via

```
git checkout 1ab3f265b3bc29c02d9984117ce114ef5f14273f
```

and use 

```
cfg_str = """
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
training:
  gpus: 1
  batch_size: 8
  optimizer:
    learning_rate: 0.00025
    lr_decay: 0.99

ckpt_path: model.ckpt
"""
```
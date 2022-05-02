To run this model, first set the tsgrasp commit via

```
git checkout e98ef9e672b394da5ca64bc1e34cc3e5a80449dc
```

and use
```
cfg_str = """
training:
  gpus: 1
  batch_size: 3
  max_epochs: 100
  optimizer:
    learning_rate: 0.0002
    lr_decay: 0.95

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

ckpt_path: tsgrasp_otf_v2/model.ckpt
"""
```

artifact playertr/TSGrasp/model-2bq1vhv6:v2
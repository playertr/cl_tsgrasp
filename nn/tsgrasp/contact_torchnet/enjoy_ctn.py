## Load model
# import wandb
# api = wandb.Api()
# artifact = api.artifact('playertr/TSGrasp/model-oq9t9qwj:v65', type='model')
# artifact_dir = artifact.download(root='ckpts/45000_1')

## Load config
# TODO: use same config from wandb run
from hydra import compose, initialize
from omegaconf import open_dict

with initialize(config_path="conf"):
    cfg = compose(config_name="config")

## Override config items to match wandb run
cfg.training.batch_size=1
# ckpt = '/home/playert/Research/tsgrasp/ckpts/45000_1/model.ckpt'
# ckpt = '/home/playert/Research/contact_torchnet/ckpts/0_371b53f6/checkpoints/epoch=99-step=12699.ckpt'
ckpt = '/home/playert/Research/contact_torchnet/outputs/2021-10-27/21-37-11/default/0/checkpoints/epoch=22-step=14535.ckpt'

# # DEBUG -- USE SUBSET OF DATA
cfg.data.data_cfg.subset_factor=512

# DEBUG -- USE CPU
cfg.training.gpus=0



with open_dict(cfg):
    cfg.training.resume_from_checkpoint=ckpt
    # DEBUG -- add figures
    cfg.training.save_figs=True

## Create Trainer
from contact_torchnet.training.trainer import Trainer
trainer = Trainer(cfg)
trainer.pl_model = trainer.pl_model.load_from_checkpoint(ckpt)



trainer.test()
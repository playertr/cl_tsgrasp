from omegaconf import DictConfig
import pytorch_lightning as pl
import os
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
import wandb
from omegaconf import OmegaConf

from hydra.utils import instantiate
import torch
torch.backends.cudnn.benchmark = True

class Trainer:
    def __init__(self, cfg : DictConfig):

        ## Lightning module and datamodule
        self.pl_model   = instantiate(cfg.model, training_cfg=cfg.training)
        self.pl_dataset = instantiate(cfg.data, batch_size=cfg.training.batch_size)

        ## Loggers
        tb_dir    = os.path.join(os.getcwd(), "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        tb_logger = loggers.TensorBoardLogger(tb_dir)
        _loggers  = [tb_logger]
        
        if cfg.training.use_wandb:
            rank = os.getenv('LOCAL_RANK')
            if rank is None or rank == 0:
                wandb.init(
                    project=cfg.training.wandb.project,
                    name=cfg.training.wandb.experiment,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    notes=cfg.training.wandb.notes
                )
                wandb.run.log_code(".")
                wandb_logger = loggers.WandbLogger(
                    project=cfg.training.wandb.project, 
                    log_model="all", 
                    name=cfg.training.wandb.experiment
                )
                wandb_logger.watch(self.pl_model)
                _loggers.append(wandb_logger)
        
        self.pl_dataset.setup()
        example_batch = next(iter(self.pl_dataset.train_dataloader()))
        _callbacks = [
            ModelCheckpoint(every_n_epochs=1), 
            LearningRateMonitor(logging_interval='step'),
        ]

        if cfg.training.animate_outputs:
            from tsgrasp.utils.viz.viz import GraspAnimationLogger
            _callbacks.append(GraspAnimationLogger(cfg.training.viz, example_batch))

        if cfg.training.make_sc_curve:
            from tsgrasp.utils.metric_utils.metrics import SCCurve
            _callbacks.append(SCCurve())

        ## Lightning Trainer
        if "resume_from_checkpoint" in cfg.training:
            ckpt = cfg.training.resume_from_checkpoint
        else:
            ckpt = None

        kwargs = dict(strategy=DDPPlugin(find_unused_parameters=True)) if cfg.training.gpus > 1 else {}

        profiler = None

        # if True:
        #     kwargs.update(dict(overfit_batches=2, check_val_every_n_epoch=100))
        #     from pytorch_lightning.profiler import PyTorchProfiler
        #     profiler = PyTorchProfiler(filename="tsgrasp.prof", with_stack=True, 
        #         with_modules=True, profile_memory=True)

        self.trainer = pl.Trainer(
            gpus=cfg.training.gpus,
            logger=_loggers,
            log_every_n_steps=10,
            callbacks=_callbacks,
            max_epochs=cfg.training.max_epochs,
            resume_from_checkpoint=ckpt,
            # profiler=profiler,
            **kwargs
        )


    def train(self):
        self.trainer.fit(self.pl_model, datamodule=self.pl_dataset)

    def test(self):
        self.trainer.test(self.pl_model, datamodule=self.pl_dataset)
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig
import os
import torch

from tsgrasp.data.acronym_renderer_dm import TrajectoryDataset, ragged_collate_fn

class LitTrajectoryDataset(pl.LightningDataModule):
    def __init__(self, data_cfg : DictConfig, batch_size):
        super().__init__()
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.num_workers = data_cfg.num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = TrajectoryDataset(self.data_cfg, split="train")
            self.dataset_val   = TrajectoryDataset(self.data_cfg, split="test")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, 
        batch_size=self.batch_size, 
        num_workers=self.num_workers, 
        collate_fn=ragged_collate_fn, persistent_workers=False,
        pin_memory=False, shuffle=True
        # sampler=RandomSampler(self.dataset_train, 
        #     num_samples=int(len(self.dataset_train)*self.data_cfg.data_proportion_per_epoch))
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, 
            num_workers=self.num_workers, collate_fn=ragged_collate_fn, pin_memory=False
        # sampler=RandomSampler(self.dataset_val, 
        #     num_samples=min(100, int(len(self.dataset_val)*self.data_cfg.data_proportion_per_epoch)))
        )

    def test_dataloader(self):
        # TODO: DON'T USE VAL DATASET FOR TESTING!
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=ragged_collate_fn
        )

    def prepare_data(self):
        files = os.listdir(self.data_cfg.dataroot)
        if not all( split in files for split in ['test', 'train']):
            raise FileNotFoundError(f"Dataroot <{self.data_cfg.dataroot}> not populated with data files. Download or generate dataset.")

# https://discuss.pytorch.org/t/new-subset-every-epoch/85018
# class RandomSampler(torch.utils.data.Sampler):
#     def __init__(self, data_source, num_samples=None):
#         self.data_source = data_source
#         self._num_samples = num_samples

#         if not isinstance(self.num_samples, int) or self.num_samples <= 0:
#             raise ValueError(
#                 "num_samples should be a positive integer "
#                 "value, but got num_samples={}".format(self.num_samples)
#             )

#     @property
#     def num_samples(self):
#         # dataset size might change at runtime
#         if self._num_samples is None:
#             return len(self.data_source)
#         return self._num_samples

#     def __iter__(self):
#         n = len(self.data_source)
#         return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

#     def __len__(self):
#         return self.num_samples
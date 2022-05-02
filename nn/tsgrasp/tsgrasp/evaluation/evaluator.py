from tsgrasp.net.tsgrasp_super import TSGraspSuper
from typing import List, Tuple
from torch.utils.data import DataLoader

from tqdm import tqdm
import torch

@torch.inference_mode
def evaluate(self, tsgrasp, ctn, dl: DataLoader):

    for batch_idx, batch in enumerate(tqdm(dl)):

        positions = batch['positions']
        tsgrasp_outputs = tsgrasp.forward(positions)
        ctn_outputs = ctn.forward(positions)
        


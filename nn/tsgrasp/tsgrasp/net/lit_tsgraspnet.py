from typing import Tuple
from omegaconf.dictconfig import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
import MinkowskiEngine as ME
from tsgrasp.net.tsgraspnet import multi_pointcloud_to_4d_coords, discretize,TSGraspNet, unweighted_sum
import torch.nn.functional as F

class LitTSGraspNet(pl.LightningModule):
    def __init__(self, model_cfg : DictConfig, training_cfg : DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSGraspNet(model_cfg)
        self.learning_rate = training_cfg.optimizer.learning_rate
        self.lr_decay = training_cfg.optimizer.lr_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{
                'params': [p for p in self.parameters()],
                'name': 'minkowski_graspnet'
            }],
            lr=self.learning_rate
        )
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                self.lr_decay),
            'name': 'learning_rate'
        }
        return [optimizer], [lr_scheduler]

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.TensorType, torch.Tensor, torch.Tensor]:
        """Calculate the grasp parameters from a batched point cloud of positions.

        For the sparse convolution, these positions should already be quantized.

        Args:
            positions (torch.Tensor): (B, T, N_PTS, 3) point cloud

        Returns:
            class_logits (torch.Tensor): (B, T, N_PTS, 1) classification logits
            baseline_dir (torch.Tensor): (B, T, N_PTS, 3) gripper baseline direction
            approach_dir (torch.Tensor): (B, T, N_PTS, 3) gripper approach direction
            grasp_offset (torch.Tensor): (B, T, N_PTS, 1) gripper width
            pred_points  (torch.Tensor): (B, T, N_PTS, 1) contact points that were evaluated. The same as positions, in this implementation.
        """
        B, T, N_PTS, D = positions.shape

        ## Make predictions. Do this in parallel.
        # Convert point positions into a Minkowski SparseTensor
        voxelized_pos = discretize(positions, grid_size=self.model.grid_size)
        coords = multi_pointcloud_to_4d_coords(voxelized_pos)
        feats = unweighted_sum(coords)
        # feats = torch.ones((len(coords), 1), device=coords.device)
        stensor = ME.SparseTensor(
            coordinates = coords,
            features = feats
        )

        # Make predictions
        class_logits, baseline_dir, approach_dir, grasp_offset = self.model.forward(stensor)

        class_logits = class_logits.reshape(B, T, N_PTS, 1)
        baseline_dir = baseline_dir.reshape(B, T, N_PTS, 3)
        approach_dir = approach_dir.reshape(B, T, N_PTS, 3)
        grasp_offset = grasp_offset.reshape(B, T, N_PTS, 1)

        return class_logits, baseline_dir, approach_dir, grasp_offset, positions



    def _step(self, batch: dict,  batch_idx: int, stage: str=None) -> dict:
        """Run inference, compute labels, and compute a loss for a given batch of data.

        Args:
            batch (dict): batch of data containing input and grasp info needed to compute labels.
            batch_idx (int): Index of this batch
            stage (str, optional): "train", "val", or "test". Defaults to None.

        Returns:
            dict: dictionary of tensors, including `loss` (with gradient) and other values for logging or analysis
        """

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache() # recommended for Minkowski

        ## Unpack data from dictionary
        positions = batch['positions'] 
        # (B, T, N_PTS, 3) cam point cloud
        grasp_tfs = batch['cam_frame_pos_grasp_tfs'] 
        # (B,) list of (T, N_GT_GRASPS_i, 4, 4) tensors of homogeneous grasp poses (from positive grasps) in the camera frame
        contact_pts = batch['pos_contact_pts_cam'] 
        # (B,) list of (T, N_GT_GRASPS_i, 2, 3) tensors of gripper contact points (for left and right fingers) in the mesh frame

        B, T, N_PTS, D = positions.shape

        ## Make predictions 
        class_logits, baseline_dir, approach_dir, grasp_offset, _ = self.forward(positions)

        ## Compute labels and losses. Do this in series over batches, because each batch might have different numbers of contact points.
        pt_preds = []
        pt_labels = []
        add_s_losses = []
        width_losses = []
        class_losses = []  
        add_s_loss, class_loss, width_loss = 0., 0., 0.
        for b in range(B):

            ## Compute labels
            pt_labels_b, width_labels_b = self.model.class_width_labels(
                contact_pts[b], positions[b], 
                self.model.pt_radius
            )
            
            ## Compute losses
            add_s_loss_b, width_loss_b, class_loss_b = self.model.losses(
                class_logits[b],
                baseline_dir[b],
                approach_dir[b],
                grasp_offset[b],
                positions[b],
                grasp_tfs[b],
                self.model.top_conf_quantile,
                pt_labels_b,
                width_labels_b
            )  

            add_s_loss += add_s_loss_b / B
            width_loss += width_loss_b / B
            class_loss += class_loss_b / B

            pt_preds.append((class_logits[b] > 0).detach().cpu().ravel())
            pt_labels.append(pt_labels_b.detach().cpu().ravel())
            add_s_losses.append(add_s_loss_b.detach().cpu())
            width_losses.append(width_loss_b.detach().cpu())
            class_losses.append(class_loss_b.detach().cpu())

        ## Combine loss components
        loss = 0.0
        loss += self.model.add_s_loss_coeff * add_s_loss
        loss += self.model.bce_loss_coeff * class_loss
        loss += self.model.width_loss_coeff * width_loss

        ## Log loss components
        self.log(f"{stage}_loss", 
            float(loss), 
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_add_s_loss", 
            float(torch.Tensor(add_s_losses).mean()), 
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_bce_loss", 
            float(torch.Tensor(class_losses).mean()), 
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_width_loss", 
            float(torch.Tensor(width_losses).mean()), 
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_pt_accuracy", 
            float(np.mean([accuracy(pred, label) for pred, label in zip(pt_preds, pt_labels)])), 
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_pt_true_pos", 
            float(np.mean([true_positive(pred, label) for pred, label in zip(pt_preds, pt_labels)])), 
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_pt_false_pos", 
            float(np.mean([false_positive(pred, label) for pred, label in zip(pt_preds, pt_labels)])), 
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_pt_true_neg", 
            float(np.mean([true_negative(pred, label) for pred, label in zip(pt_preds, pt_labels)])), 
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_pt_true_pos", 
            float(np.mean([true_positive(pred, label) for pred, label in zip(pt_preds, pt_labels)])), 
            on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_pt_recall", 
            float(np.mean([recall(pred, label) for pred, label in zip(pt_preds, pt_labels)])), 
            on_step=True, on_epoch=True, sync_dist=True)

        return {
            'loss': loss
        }
    
    # Pytorch Lightning flow of control is as follows.
    # on_train_start()
    # for epoch in epochs:
    #   on_train_epoch_start()
    #   for batch in train_dataloader():
    #       on_train_batch_start()
    #       training_step()
    #       ...
    #       on_train_batch_end()
    #       on_validation_epoch_start()
    #
    #       for batch in val_dataloader():
    #           on_validation_batch_start()
    #           validation_step()
    #           on_validation_batch_end()
    #       on_validation_epoch_end()
    #
    #   on_train_epoch_end()
    # on_train_end

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        # return self._step(batch, batch_idx, "test")
        pass

def accuracy(pred, des):
    return float(torch.mean((pred == des).float()))

def true_positive(pred, des):
    return float(torch.mean((des.bool()[pred.bool()].float())))

def false_positive(pred, des):
    return float(torch.mean(((~des.bool()[pred.bool()]).float())))

def true_negative(pred, des):
    return float(torch.mean((~des.bool()[~pred.bool()]).float()))

def false_negative(pred, des):
    return float(torch.mean((des.bool()[~pred.bool()]).float()))

def recall(pred, label):
    return float(torch.mean(pred.bool()[label.bool()].float()))
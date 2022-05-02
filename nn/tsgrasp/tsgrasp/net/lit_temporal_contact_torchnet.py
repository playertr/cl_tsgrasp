# lit_temporal_contact_torchnet.py
# Tim Player, 2 November 2021, playert@oregonstate.edu
# A Pytorch Lightning module to wrap Contact Torchnet, so that
# the module can train and test using the the identical temporal batching to
# our temperospatial grasp network.

from omegaconf.dictconfig import DictConfig
import torch
from tsgrasp.net.contact_torchnet import ContactTorchNet
import pytorch_lightning as pl
import numpy as np
from typing import Tuple

class LitTemporalContactTorchNet(pl.LightningModule):
    def __init__(self, model_cfg : DictConfig, training_cfg : DictConfig):
        super().__init__()

        self.save_hyperparameters()
        self.model = ContactTorchNet(global_config=model_cfg)

        self.learning_rate = training_cfg.optimizer.learning_rate
        self.lr_decay = training_cfg.optimizer.lr_decay

        # Deactivate the automatic optimization step, so that we can
        # train for multiple timesteps sequentially in step().
        self.automatic_optimization = False

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
            class_logits (torch.Tensor): (B, T, N_PRED_PTS, 1) classification logits
            baseline_dir (torch.Tensor): (B, T, N_PRED_PTS, 3) gripper baseline direction
            approach_dir (torch.Tensor): (B, T, N_PRED_PTS, 3) gripper approach direction
            grasp_offset (torch.Tensor): (B, T, N_PRED_PTS, 1) gripper width
        """
        B, T, N_PTS, D = positions.shape

        # Make predictions

        (baseline_dir, class_logits, grasp_offset, approach_dir, pred_points
        ) = self.model.forward(positions.reshape(-1, N_PTS, 3))

        class_logits = class_logits.reshape(B, T, -1, 1)
        baseline_dir = baseline_dir.reshape(B, T, -1, 3)
        approach_dir = approach_dir.reshape(B, T, -1, 3)
        grasp_offset = grasp_offset.reshape(B, T, -1, 1)

        return class_logits, baseline_dir, approach_dir, grasp_offset, pred_points.unsqueeze(0)

    def _step(self, batch, batch_idx, stage=None):
        opt = self.optimizers()

        ## Unpack data from dictionary
        positions = batch['positions'] 
        # (B, T, N_PTS, 3) cam point cloud
        grasp_tfs = batch['cam_frame_pos_grasp_tfs'] 
        # (B,) list of (T, N_GT_GRASPS_i, 4, 4) tensors of homogeneous grasp poses (from positive grasps) in the camera frame
        contact_pts = batch['pos_contact_pts_cam'] 
        # (B,) list of (T, N_GT_GRASPS_i, 2, 3) tensors of gripper contact points (for left and right fingers) in the mesh frame

        B, T, N_PTS, D = positions.shape
        assert B == 1, "Only batch size of one supported for CTN."


        losses = []   
        pt_preds = []
        pt_labels = []
        add_s_losses = []
        width_losses = []
        class_losses = []  
        ## Compute labels and losses. Do this in series over batches, because each batch might have different numbers of contact points.
        for t in range(T):
            if stage == "train":
                opt.zero_grad()

            (baseline_dir, class_logits, grasp_offset, approach_dir, pred_points
            ) = self.model.forward(positions[0][t].unsqueeze(0))
            # class_logits is (1, N_PRED_PTS)
            # baseline_dir is (1, N_PRED_PTS, 3)
            # approach_dir is (1, N_PRED_PTS, 3)
            # grasp_offset is (1, N_PRED_PTS, 10)
            # pred_points is (1, N_PRED_PTS, 3)

            ## Compute labels
            pt_labels_b, width_labels_b = self.model.class_width_labels(
                contact_pts[0][t].unsqueeze(0), 
                pred_points,
                self.model.pt_radius
            )

            ## Compute losses
            add_s_loss_b, width_loss_b, class_loss_b = self.model.losses(
                class_logits.unsqueeze(-1),
                baseline_dir,
                approach_dir,
                grasp_offset,
                pred_points,
                grasp_tfs[0][t].unsqueeze(0),
                self.model.top_conf_quantile,
                pt_labels_b[0],
                width_labels_b
            ) 

            ## Combine loss components
            loss = 0.0
            loss += self.model.add_s_loss_coeff * add_s_loss_b 
            loss += self.model.bce_loss_coeff * class_loss_b 
            loss += self.model.width_loss_coeff * width_loss_b

            if stage == "train":
                self.manual_backward(loss)
                opt.step()

            losses.append(loss.detach().cpu())
            pt_preds.append((class_logits > 0).detach().cpu().ravel())
            pt_labels.append(pt_labels_b.detach().cpu().ravel())
            add_s_losses.append(add_s_loss_b.detach().cpu())
            width_losses.append(width_loss_b.detach().cpu())
            class_losses.append(class_loss_b.detach().cpu())

        ## Log loss components
        self.log(f"{stage}_loss", 
            float(torch.Tensor(losses).mean()), 
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
        self.log(f"{stage}_accuracy", 
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

        return {"loss": torch.mean(torch.Tensor(losses))}

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

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
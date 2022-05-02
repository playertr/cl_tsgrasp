from omegaconf.dictconfig import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from .contact_torchnet import ContactTorchNet, compute_labels_ragged_grasps

from ..utils.metric_utils.metrics import success_coverage_curve

class LitContactTorchNet(pl.LightningModule):
    def __init__(self, model_cfg : DictConfig, training_cfg : DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = ContactTorchNet(model_cfg)
        self.learning_rate = 0.001 # TODO pass as cfg

        self.train_pt_acc = torchmetrics.Accuracy()
        self.val_pt_acc = torchmetrics.Accuracy()
        self.test_pt_acc = torchmetrics.Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{
                'params': [p for p in self.parameters()],
                'name': 'minkowski_graspnet'
            }],
            lr=self.learning_rate
        )

        lr_scheduler = {
            # TODO make not magic number
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99),
            'name': 'learning_rate'
        }
        return [optimizer], [lr_scheduler]

    def forward(self,xyz):
        return self.model.forward(xyz)

    def _step(self, batch,  batch_idx, stage=None):
        ## Create grasp predictions
        xyz = batch['all_pos']
        (baseline_dir, binary_seg, grasp_offset, approach_dir, pred_points
        ) = self.model.forward(xyz)

        ## Compute grasp labels at the predicted grasp locations
        gt_grasp_tfs_cam = batch['cam_frame_pos_grasp_tfs']
        gt_contact_pts = batch['pos_contact_pts_mesh']
        gt_widths = batch['pos_finger_diffs']
        cam_pose = batch['camera_pose']
        
        grasp_success_label, approach_dir_label, baseline_dir_label, grasp_width_label = \
            compute_labels_ragged_grasps(
            pred_points, gt_grasp_tfs_cam, gt_contact_pts, gt_widths, cam_pose
        )

        ## Compute loss
        cp_tensor = batch['single_gripper_points'][0]
        sym_cp_tensor = batch['sym_single_gripper_points'][0] # TODO -- make symmetric
        loss = self.model.loss(
            baseline_dir=       baseline_dir, 
            binary_seg=         binary_seg,
            grasp_offset=       grasp_offset, 
            approach_dir=       approach_dir, 
            pred_points=        pred_points, 
            baseline_dir_labels=baseline_dir_label, 
            offset_labels=      grasp_width_label, 
            approach_dir_labels=approach_dir_label, 
            point_labels=       grasp_success_label, 
            cp_tensor=          cp_tensor, 
            sym_cp_tensor=      sym_cp_tensor)

        pt_preds = binary_seg > 0
        pt_labels = grasp_success_label
        return {
            'loss': loss, 
            'pt_preds': pt_preds.detach().cpu(), 
            'pt_labels': pt_labels.squeeze().detach().cpu(), 
            'outputs': (
                binary_seg.detach().cpu(), 
                baseline_dir.detach().cpu(), 
                approach_dir.detach().cpu(), 
                grasp_offset.detach().cpu(), 
                pred_points.detach().cpu())
        }

    def training_step_end(self, outputs):
        self.train_pt_acc(outputs['pt_preds'], outputs['pt_labels'].int())
        self.log('train_pt_acc', self.train_pt_acc.compute())
        self.log('training_loss', outputs['loss'])

    def validation_step_end(self, outputs):
        self.val_pt_acc(outputs['pt_preds'], outputs['pt_labels'].int())
        self.log('val_pt_acc', self.val_pt_acc.compute())
        self.log('val_loss', outputs['loss'])

    def test_step_end(self, outputs):
        self.test_pt_acc(outputs['pt_preds'], outputs['pt_labels'].int())
        self.log('test_pt_acc', self.test_pt_acc.compute())
        self.log('test_loss', outputs['loss'])

    def _epoch_end(self, outputs, stage=None):
        if stage:
            loss = np.mean([float(x['loss']) for x in outputs])
            self.logger.log_metrics(
                {f"{stage}/loss": loss}, self.current_epoch + 1)
    
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
        ## Create grasp predictions
        xyz = batch['all_pos']
        (baseline_dir, binary_seg, grasp_offset, approach_dir, pred_points
        ) = self.model.forward(xyz)

        ## Compute grasp labels at the predicted grasp locations
        gt_grasp_tfs = batch['cam_frame_pos_grasp_tfs']
        gt_contact_pts = batch['pos_contact_pts_mesh']
        gt_widths = batch['pos_finger_diffs']
        cam_pose = batch['camera_pose']
        
        grasp_success_label, approach_dir_label, baseline_dir_label, grasp_width_label = \
            compute_labels_ragged_grasps(
            pred_points, gt_grasp_tfs, gt_contact_pts, gt_widths, cam_pose
        )

        ## Compute loss
        cp_tensor = batch['single_gripper_points'][0]
        sym_cp_tensor = batch['sym_single_gripper_points'][0] # TODO -- make symmetric
        loss = self.model.loss(
            baseline_dir=       baseline_dir, 
            binary_seg=         binary_seg, 
            grasp_offset=       grasp_offset, 
            approach_dir=       approach_dir, 
            pred_points=        pred_points, 
            baseline_dir_labels=baseline_dir_label, 
            offset_labels=      grasp_width_label, 
            approach_dir_labels=approach_dir_label, 
            point_labels=       grasp_success_label, 
            cp_tensor=          cp_tensor, 
            sym_cp_tensor=      sym_cp_tensor)

        pt_preds = binary_seg > 0
        pt_labels = grasp_success_label

        # this calc might change with different batch size
        # transform ground truth contact points into camera frame

        from contact_torchnet.net.contact_torchnet import transform

        b = gt_contact_pts.shape[0] 
        pos_gt_grasp_locs = gt_contact_pts.reshape(b, -1, 3)
        pos_gt_grasp_locs = transform(pos_gt_grasp_locs, torch.linalg.inv(cam_pose))
        
        print(f"pt_preds.sum(): {pt_preds.sum()}. pt_labels.sum(): {pt_labels.sum()}")

        sc_table_full = success_coverage_curve(torch.sigmoid(binary_seg), pred_points.squeeze(0), 
            pt_labels.squeeze(), pos_gt_grasp_locs.squeeze(0))

        sc_table_pred_pts = success_coverage_curve(torch.sigmoid(binary_seg), pred_points.squeeze(0), 
            pt_labels.squeeze(), pred_points.squeeze()[pt_labels.squeeze()].squeeze())

        return {
            'loss': loss, 
            'pt_preds': pt_preds.detach().cpu(), 
            'pt_labels': pt_labels.squeeze().detach().cpu(), 
            'outputs': (binary_seg.detach().cpu(), baseline_dir.detach().cpu(), 
                approach_dir.detach().cpu(), grasp_offset.detach().cpu(), pred_points.detach().cpu()),
            'sc_table_full': sc_table_full,
            'sc_table_pred_pts': sc_table_pred_pts,
            'labels': (grasp_success_label.detach().cpu(), approach_dir_label.detach().cpu(), baseline_dir_label.detach().cpu(), grasp_width_label.detach().cpu())
        }

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs)
        import pandas as pd

        ## Full-cp table
        df = pd.concat([d['sc_table_full'] for d  in outputs])
        df = df.groupby('confidence').mean()
        from contact_torchnet.utils.metric_utils.metrics import plot_s_c_curve
        import matplotlib.pyplot as plt
        plot = plot_s_c_curve(df)
        plt.savefig('/home/playert/Research/contact_torchnet/curve_full.png')
        df.to_csv('curve_full.csv')

        # Only pred-cp table
        df = pd.concat([d['sc_table_pred_pts'] for d  in outputs])
        df = df.groupby('confidence').mean()
        plot = plot_s_c_curve(df)
        plt.savefig('/home/playert/Research/contact_torchnet/curve_pred_pts.png')
        df.to_csv('curve_pred_pts.csv')


        print("boo-ya!")
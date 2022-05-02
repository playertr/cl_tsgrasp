from omegaconf.dictconfig import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from tsgrasp.net.minkowski_graspnet import MinkowskiGraspNet
import MinkowskiEngine as ME

# import tracemalloc

class LitMinkowskiGraspNet(pl.LightningModule):
    def __init__(self, model_cfg : DictConfig, training_cfg : DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = MinkowskiGraspNet(model_cfg)
        self.learning_rate = training_cfg.optimizer.learning_rate
        self.lr_decay = training_cfg.optimizer.lr_decay

        # tracemalloc.start()
        # self.snapshot = tracemalloc.take_snapshot()

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

    def forward(self,x):
        return self.model.forward(x)

    def _step(self, batch,  batch_idx, stage=None):
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.compare_to(self.snapshot, 'traceback')
        # pick the biggest memory block
        # for i in range(5):
        #     stat = top_stats[i]
        #     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        #     for line in stat.traceback.format():
        #         print(line)

        # import os, psutil
        # process = psutil.Process(os.getpid())
        # print(f"Total memory used: {process.memory_info().rss / 1e6} MB")  # in bytes 

        # self.snapshot = snapshot

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache() # recommended for Minkowski

        stensor = ME.SparseTensor(
            coordinates = batch['coordinates'],
            features = batch['features']
        )

        class_logits, baseline_dir, approach_dir, grasp_offset = self.model.forward(stensor)

        pt_labels = batch['labels']

        # Each 'weighted' loss has been multiplied by its loss coefficient
        weighted_add_s_loss = self.model.weighted_add_s_loss(
            class_logits, pt_labels,
            baseline_dir, approach_dir, grasp_offset,
            positions = batch['positions'],
            pos_control_points = batch['pos_control_points'], 
            sym_pos_control_points = batch['sym_pos_control_points'],
            gt_grasps_per_batch = batch['gt_grasps_per_batch'], 
            single_gripper_points = batch['single_gripper_points']
        )

        weighted_width_loss = self.model.weighted_width_loss(
            grasp_offset,
            grasp_offset_label= batch["pos_finger_diffs"],
            labels=pt_labels
        )

        weighted_bce_loss = self.model.weighted_bce_loss(
            class_logits, pt_labels
        )

        self.log(f"{stage}_add_s_loss", weighted_add_s_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_bce_loss", weighted_bce_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_width_loss", weighted_width_loss, on_step=True, on_epoch=True, sync_dist=True)

        loss = weighted_add_s_loss + weighted_bce_loss + weighted_width_loss

        pt_preds = class_logits > 0

        return {
            'loss': loss, 
            'pt_preds': pt_preds.detach().cpu(), 
            'pt_labels': pt_labels.detach().cpu(), 
            # 'outputs': (
            #     class_logits.detach().cpu(), 
            #     baseline_dir.detach().cpu(), 
            #     approach_dir.detach().cpu(), 
            #     grasp_offset.detach().cpu())
        }

    def training_step_end(self, outputs):
        self.log('train_pt_acc', accuracy(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_pt_true_pos', true_positive(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_pt_false_pos', false_positive(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_pt_true_neg', true_negative(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_pt_false_neg', false_negative(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('training_loss', float(outputs['loss']), on_step=True, on_epoch=True, sync_dist=True)

    def validation_step_end(self, outputs):
        self.log('val_pt_acc', accuracy(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_pt_true_pos', true_positive(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_pt_false_pos', false_positive(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_pt_true_neg', true_negative(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_pt_false_neg', false_negative(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_loss', float(outputs['loss']), on_step=True, on_epoch=True, sync_dist=True)

    def test_step_end(self, outputs):
        self.log('test_pt_acc', accuracy(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_pt_true_pos', true_positive(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_pt_false_pos', false_positive(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_pt_true_neg', true_negative(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_pt_false_neg', false_negative(
            outputs['pt_preds'], outputs['pt_labels']), on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_loss', float(outputs['loss']), on_step=True, on_epoch=True, sync_dist=True)

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
        return self._step(batch, batch_idx, "test")

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "test")

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
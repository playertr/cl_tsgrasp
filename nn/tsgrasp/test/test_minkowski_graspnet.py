from test.fixtures import *

from tsgrasp.net.minkowski_graspnet import MinkowskiGraspNet

import MinkowskiEngine as ME
import torch

def test_minkowski_graspnet_init(minkowski_graspnet):
    assert isinstance(minkowski_graspnet, MinkowskiGraspNet)

def test_minkowski_graspnet_forward(minkowski_graspnet, acronymvid_dataloader):

    batch = next(iter(acronymvid_dataloader))

    stensor = ME.SparseTensor(
        coordinates = batch['coordinates'],
        features = batch['features']
    )

    class_logits, baseline_dir, approach_dir, grasp_offset = minkowski_graspnet.forward(stensor)

    assert all( isinstance(t, torch.Tensor) for t in [
        class_logits,baseline_dir, approach_dir, grasp_offset])
    

def test_minkowski_graspnet_loss(minkowski_graspnet, acronymvid_dataloader):
    
    batch = next(iter(acronymvid_dataloader))

    stensor = ME.SparseTensor(
        coordinates = batch['coordinates'],
        features = batch['features']
    )

    class_logits, baseline_dir, approach_dir, grasp_offset = minkowski_graspnet.forward(stensor)

    labels = batch['labels']

    loss = minkowski_graspnet.loss(
        class_logits, labels,
        baseline_dir, approach_dir, grasp_offset,
        positions = batch['positions'],
        pos_control_points = batch['pos_control_points'], 
        sym_pos_control_points = batch['sym_pos_control_points'],
        gt_grasps_per_batch = batch['gt_grasps_per_batch'], 
        single_gripper_points = batch['single_gripper_points']
    )

    assert loss > 0, "Loss should be a real positive number."
import torch
import pandas as pd

def success_from_labels(pred_classes, actual_classes):
    """Return success, as proportion, from ground-truth point labels.
    
    (correct positive predictions) / (total positive predictions)"""
    return torch.mean(actual_classes[pred_classes==1].float())

def coverage(pos_pred_grasp_locs, pos_gt_grasp_locs, radius=0.005):
    """Get the proportion of ground truth grasps within epsilon of predicted.
    
    (gt close to predicted) / (total gt)
    """
    if pos_pred_grasp_locs.shape[0] == 0: return torch.Tensor([0.0])

    # for each grasp coordinate in gt_grasps, find the distance to every grasp grasp_coordinate in pred_grasps
    dists = distances(pos_gt_grasp_locs, pos_pred_grasp_locs) # (n_gt, n_pred)

    closest_dists, idxs = dists.min(axis=1)
    return torch.mean((closest_dists < radius).float())

# @TensorMemoize
def distances(grasps1, grasps2):
    """Finds the L2 distances from 3D points in grasps1 to points in grasps2.

    Returns a (grasp1.shape[0], grasps3.shape[0]) ndarray."""

    diffs = torch.unsqueeze(grasps1, -2) - torch.unsqueeze(grasps2, -3)
    return torch.linalg.norm(diffs, axis=-1)

# from collections import deque
# class TensorMemoize:
#     """Decorator to memoize a function with Tensor arguments.
    
#     Arguments are not hashed, so this is slow and memory intensive."""

#     # I haven't figured out how to pass parameters to decorators.
#     def __init__(self, f, maxlen=1):
#         self.f = f
#         self.memo = deque(maxlen=maxlen)
    
#     def __call__(self, *args):
#         for (cargs, val) in self.memo:
#             # assumes len(args) == len(cargs)
#             if all(torch.equal(cargs[i], args[i]) for i in range(len(args))):
#                 return val

#         val = self.f(*args)
#         self.memo.appendleft((args, val))
#         return val

def success_coverage_curve(confs, pred_grasp_locs, gt_labels, pos_gt_grasp_locs):
    """Determine the success and coverage at various threshold confidence values."""

    res = []
    thresholds = torch.linspace(0, 1, 10000)

    # quantiles = torch.linspace(0, 1, 1000)
    # thresholds = [torch.quantile(confs, q) for q in quantiles]

    for t in thresholds:
        pred_classes = confs > t
        pos_pred_grasp_locs = pred_grasp_locs[pred_classes == 1, :]
        res.append({
            "confidence": t,
            "success": success_from_labels(pred_classes, gt_labels),
            "coverage": coverage(pos_pred_grasp_locs, pos_gt_grasp_locs)
        })
    
    return pd.DataFrame(res).astype(float)

import matplotlib.pyplot as plt
def plot_s_c_curve(df, ax=None, title="Coverage vs. Success"):
    if ax is None:
        fig, ax = plt.subplots()
    
    plot = ax.plot(df['coverage'], df['success'])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Success")
    ax.set_title(title)

    return plot

import hydra
from omegaconf import DictConfig
import h5py
import os
from tqdm import tqdm
import torch
from tsgrasp.utils.metric_utils.metrics import framewise_sc_curve, precision_recall_curve
from tsgrasp.utils.metric_utils.metrics import plot_sc_curve, plot_pr_curve
import matplotlib.pyplot as plt

import pandas as pd

@hydra.main(config_path="../conf", config_name="scripts/calculate_comparison_metrics")
def main(cfg : DictConfig):
    with h5py.File(cfg.tsgrasp_output_path, 'r') as ts_ds:
        with h5py.File(cfg.ctn_output_path, 'r') as ctn_ds:
            s_c_curves = make_s_c_curves(ts_ds, ctn_ds, cfg)

def make_s_c_curves(ts_ds, ctn_ds, cfg):

    ts_curves = []
    ctn_curves = []
    ts_pr_curves = []
    ctn_pr_curves = []
    i = 0
    for example_num in tqdm(ts_ds['outputs']):
        i += 1
        if i > 5 and cfg.test_run:
            break 
        ts_results = ts_ds['outputs'][example_num]
        ctn_results = ctn_ds['outputs'][example_num]

        device = torch.device('cuda') if cfg.gpus > 0 else torch.device('cpu')
        # Load relevant confidence and position information from each model
        ts_confs = torch.sigmoid(torch.Tensor(ts_results['class_logits'])).to(device)
        ts_pos = torch.Tensor(ts_results['positions']).to(device)
        ts_gt_contact_pts = torch.Tensor(ts_results['gt_contact_pts']).to(device)
        ts_labels = torch.Tensor(ts_results['pt_labels']).to(device)

        ctn_confs = torch.sigmoid(torch.Tensor(ctn_results['class_logits'])).to(device)
        ctn_pos = torch.Tensor(ctn_results['positions']).to(device)
        ctn_gt_contact_pts = torch.Tensor(ctn_results['gt_contact_pts']).to(device)
        ctn_labels = torch.Tensor(ctn_results['pt_labels']).to(device)

        ## Make success-coverage curves
        ts_curves.append(
            framewise_sc_curve(ts_confs, ts_pos, ts_labels, ts_gt_contact_pts, radius=cfg.success_radius)
        )
        ctn_curves.append(
            framewise_sc_curve(ctn_confs, ctn_pos, ctn_labels, ctn_gt_contact_pts, radius=cfg.success_radius)
        )

        ## Make precision-recall curves
        ts_pr_curves.append(
            precision_recall_curve(ts_confs, ts_labels)
        )
        ctn_pr_curves.append(
            precision_recall_curve(ctn_confs, ctn_labels)
        )

    ## Concatenate the outputs from every example and compute aggregated curves
    # Success-coverage curves
    examples = range(len(ts_curves))
    ts_curves = pd.concat(ts_curves, keys=[f"ex_{ex}" for ex in examples])
    ts_curve_gb = ts_curves.groupby('confidence') # DataFrameGroupBy object
    ts_curve = pd.DataFrame()
    ts_curve['success'] = ts_curve_gb['n_correctly_pred_positive'].sum() / ts_curve_gb['n_pred_positive'].sum()
    ts_curve['coverage'] = ts_curve_gb['n_covered_gt_points'].sum() / ts_curve_gb['n_gt_points'].sum()

    ctn_curves = pd.concat(ctn_curves, keys=[f"ex_{ex}" for ex in examples])
    ctn_curve_gb = ctn_curves.groupby('confidence') # DataFrameGroupBy object
    ctn_curve = pd.DataFrame()
    ctn_curve['success'] = ctn_curve_gb['n_correctly_pred_positive'].sum() / ctn_curve_gb['n_pred_positive'].sum()
    ctn_curve['coverage'] = ctn_curve_gb['n_covered_gt_points'].sum() / ctn_curve_gb['n_gt_points'].sum()


    # Precision-recall curves
    ts_pr_curves = pd.concat(ts_pr_curves, keys=[f"ex_{ex}" for ex in examples])
    ts_pr_curve_gb = ts_pr_curves.groupby('confidence') # DataFrameGroupBy object
    ts_pr_curve = pd.DataFrame()
    ts_pr_curve['precision'] = ts_pr_curve_gb['n_correctly_pred_positive'].sum() / ts_pr_curve_gb['n_pred_positive'].sum()
    ts_pr_curve['recall'] = ts_pr_curve_gb['n_correctly_pred_positive'].sum() / ts_pr_curve_gb['n_actual_positive'].sum()


    ctn_pr_curves = pd.concat(ctn_pr_curves, keys=[f"ex_{ex}" for ex in examples])
    ctn_pr_curve_gb = ctn_pr_curves.groupby('confidence') # DataFrameGroupBy object
    ctn_pr_curve = pd.DataFrame()
    ctn_pr_curve['precision'] = ctn_pr_curve_gb['n_correctly_pred_positive'].sum() / ctn_pr_curve_gb['n_pred_positive'].sum()
    ctn_pr_curve['recall'] = ctn_pr_curve_gb['n_correctly_pred_positive'].sum() / ctn_pr_curve_gb['n_actual_positive'].sum()

    ## Write dataframes to disk
    # Success-coverage curves
    ts_curves.to_csv(cfg.tsgrasp_sc_csv_path)
    ctn_curves.to_csv(cfg.ctn_sc_csv_path)

    # Precision-recall curves
    ts_pr_curves.to_csv(cfg.tsgrasp_pr_csv_path)
    ctn_pr_curves.to_csv(cfg.ctn_pr_csv_path)

    ## Plot curves
    # Success-coverage curves
    fig, ax = plt.subplots()
    plot_sc_curve(ts_curve, ax=ax, label='tsgraspnet')
    plot_sc_curve(ctn_curve, ax=ax, label='CTN')
    fig.legend()
    plt.savefig(cfg.sc_png_path)

    # Precision-recall curves
    fig, ax = plt.subplots()
    plot_pr_curve(ts_pr_curve, ax=ax, label='tsgraspnet')
    plot_pr_curve(ctn_pr_curve, ax=ax, label='CTN')
    fig.legend()

    plt.savefig(cfg.pr_png_path)


if __name__ == "__main__":
    main()
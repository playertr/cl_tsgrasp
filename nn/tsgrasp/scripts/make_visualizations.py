import hydra
from omegaconf import DictConfig
import h5py
from tsgrasp.utils.viz.viz import animate_grasps_from_outputs
from tqdm import tqdm
import torch
import os

@hydra.main(config_path="../conf", config_name="scripts/make_visualizations")
def main(cfg : DictConfig):
    with h5py.File(cfg.output_path, 'r') as ds:
        visualize_grasps(ds, cfg)

def visualize_grasps(ds, cfg):
    """Take in the grasps from an h5py output dataset and plot them."""

    os.makedirs(cfg.gif_path, exist_ok=True)
    for example_num in tqdm(ds['outputs']):

        results = ds['outputs'][example_num]

        device = torch.device('cuda') if cfg.gpus > 0 else torch.device('cpu')
        # Load relevant confidence and position information from each model
        class_logits =torch.Tensor(results['class_logits']).to(device)
        baseline_dir = torch.Tensor(results['baseline_dir']).to(device)
        approach_dir = torch.Tensor(results['approach_dir']).to(device)
        grasp_offset = torch.Tensor(results['grasp_offset']).to(device)
        pts = torch.Tensor(results['positions']).to(device)

        outputs = (
            class_logits.unsqueeze(0), 
            baseline_dir.unsqueeze(0), 
            approach_dir.unsqueeze(0), 
            grasp_offset.unsqueeze(0), 
            pts.unsqueeze(0)
        )
        animate_grasps_from_outputs(outputs, name=f"{cfg.gif_path}/{example_num}")

if __name__ == "__main__":
    main()
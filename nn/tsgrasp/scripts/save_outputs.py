import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import h5py
import os
from tqdm import tqdm
import torch

@hydra.main(config_path="../conf", config_name="scripts/save_outputs")
def main(cfg : DictConfig):
        
    pl_model = instantiate(cfg.model, training_cfg=cfg.training)
    pl_model.load_state_dict(torch.load(cfg.ckpt_path)['state_dict'])
    pl_model.eval()

    pl_data = instantiate(cfg.data, batch_size=cfg.training.batch_size)
    pl_data.setup()

    if cfg.training.gpus > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    pl_model = pl_model.to(device)

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    ds = h5py.File(cfg.output_path,'w')
    ds["description"] = cfg.description
    ds.create_group("outputs")
    evaluate(pl_model, pl_data.test_dataloader(), ds)
    ds.close()

@torch.inference_mode()
def evaluate(pl_model, dl, h5_ds):
    """Iterate through the dataloader, saving outputs."""

    example_num = 0
    for batch in tqdm(dl):

        ## Run inference on this batch
        (
            class_logits,
            baseline_dir,
            approach_dir,
            grasp_offset,
            positions
        ) = pl_model.forward(batch['positions'].to(pl_model.device))

        ## Save each example within the batch into its own group
        for b in range(len(class_logits)):
            h5_ds['outputs'].create_group(str(example_num))
            grp = h5_ds['outputs'][str(example_num)]
            grp.create_dataset(
                "class_logits", class_logits[b].shape, data=class_logits[b].detach().cpu(), compression="gzip", compression_opts=9
            )
            grp.create_dataset(
                "baseline_dir", baseline_dir[b].shape, data=baseline_dir[b].detach().cpu(), compression="gzip", compression_opts=9
            )
            grp.create_dataset(
                "approach_dir", approach_dir[b].shape, data=approach_dir[b].detach().cpu(), compression="gzip", compression_opts=9
            )
            grp.create_dataset(
                "grasp_offset", grasp_offset[b].shape, data=grasp_offset[b].detach().cpu(), compression="gzip", compression_opts=9
            )
            grp.create_dataset(
                "positions", positions[b].shape, data=positions[b].detach().cpu(), compression="gzip", compression_opts=9
            )

            ## Retrieve contact points
            cp = unstack_contact_points(batch['pos_contact_pts_cam'][b])
            T, N, _3 = cp.shape
            cp = cp[~cp.isnan()].reshape(T, -1, 3).to(pl_model.device)

            ## Compute distance-based point labels
            if cp.shape[1] > 0:
                dists, _ = pl_model.model.closest_points(
                    positions[b],
                    cp
                )
                pt_labels = (dists < pl_model.model.pt_radius).unsqueeze(-1)
            else:
                pt_labels = torch.zeros_like(class_logits[b], dtype=bool)

            grp.create_dataset(
                "gt_contact_pts", cp.shape, data=cp.cpu(), compression="gzip", compression_opts=9
            )

            grp.create_dataset(
                "pt_labels", pt_labels.shape, data=pt_labels.cpu(), compression="gzip", compression_opts=9
            )
            
            example_num += 1

def unstack_contact_points(cp):
    """Change contact point tensor from (T, N_GT_GRASPS, 2, 3) to (T, 2*G_GT_GRASPS, 3"""
    return torch.cat([cp[...,0,:], cp[...,1,:]], dim=-2)

if __name__ == "__main__":
    main()
## Load model
import wandb
import hydra
import os

@hydra.main(config_path="../conf", config_name="scripts/download_model")
def main(cfg):
    api = wandb.Api()
    artifact = api.artifact(cfg.artifact_name, type='model')
    os.makedirs(cfg.download_dir, exist_ok=True)
    artifact_dir = artifact.download(root=cfg.download_dir)

if __name__ == "__main__":
    main()
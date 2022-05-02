import hydra
from omegaconf import DictConfig
from tsgrasp.training.trainer import Trainer

@hydra.main(config_path="conf", config_name="config")
def train(cfg : DictConfig):
        
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    train()
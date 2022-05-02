import hydra
import torch

@hydra.main(config_path="../conf", config_name="scripts/enjoy")
def main(cfg):

    ## Create Trainer
    from tsgrasp.training.trainer import Trainer
    trainer = Trainer(cfg)

    trainer.pl_model.load_state_dict(torch.load(cfg.training.resume_from_checkpoint)['state_dict'])

    # trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()
from tsgrasp.training.trainer import Trainer
from test.fixtures import cfg

def test_trainer_construction(cfg):
    trainer = Trainer(cfg)
    assert isinstance(trainer,  Trainer)

# Takes forever -- only run in debugger
# def test_trainer_training(cfg):
#     trainer = Trainer(cfg)
#     trainer.train()